import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.text import Text

from agent.config import get_config


def fmt_ms(ms: float) -> str:
    secs = ms / 1000
    if secs >= 60:
        m = int(secs // 60)
        s = secs % 60
        return f"{m}m {s:.1f}s"
    return f"{secs:.2f}s"


@dataclass
class LogEvent:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"]
    module: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)
    duration_ms: float | None = None


log_queue: asyncio.Queue[LogEvent] = asyncio.Queue()
_console = Console(highlight=False, file=sys.stderr)
_highlighter = ReprHighlighter()

log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "SUCCESS": logging.INFO,
}

log_level_styles = {
    "DEBUG": "dim white",
    "INFO": "bright_cyan",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "SUCCESS": "bold green",
}


async def emit(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"],
    module: str,
    message: str,
    data: dict | None = None,
    duration_ms: float | None = None,
) -> None:
    config = get_config()

    event = LogEvent(
        level=level,
        module=module,
        message=message,
        data=data or {},
        duration_ms=duration_ms,
    )
    await log_queue.put(event)

    if log_level_map[level] < log_level_map[config.log.log_level.upper()]:
        return

    timestamp = event.timestamp.strftime("%H:%M:%S")
    level_text = Text(f" {level.ljust(7)}", style=log_level_styles[level])
    module_text = Text(f" {module.ljust(12)}", style="bold magenta")
    message_text = Text(f" {_highlighter(message)}", style=log_level_styles[level])

    log_line = Text.assemble(
        Text(f"[{timestamp}] ", style="dim white"),
        level_text,
        module_text,
        " ",
        message_text,
    )

    if duration_ms is not None:
        log_line.append(f" ({fmt_ms(duration_ms)})", style="dim white")

    _console.print(log_line)


async def log_info(module: str, message: str, **kwargs) -> None:
    await emit("INFO", module, message, **kwargs)


async def log_error(module: str, message: str, **kwargs) -> None:
    await emit("ERROR", module, message, **kwargs)


async def log_success(module: str, message: str, **kwargs) -> None:
    await emit("SUCCESS", module, message, **kwargs)


async def log_warning(module: str, message: str, **kwargs) -> None:
    await emit("WARNING", module, message, **kwargs)


async def log_debug(module: str, message: str, **kwargs) -> None:
    await emit("DEBUG", module, message, **kwargs)


class FileLogWriter:
    def __init__(self, log_dir: str = "./logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._task: asyncio.Task | None = None
        self._running = False

    def _current_path(self) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"agent_{date_str}.log"

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _drain(self) -> None:
        global log_queue
        while self._running:
            try:
                event = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                self._write_event(event)
            except TimeoutError:
                continue
            except Exception:
                pass

    def _write_event(self, event: LogEvent) -> None:
        record = {
            "timestamp": event.timestamp.isoformat(),
            "level": event.level,
            "module": event.module,
            "message": event.message,
            "data": event.data,
            "duration_ms": event.duration_ms,
        }
        line = json.dumps(record, ensure_ascii=False)
        path = self._current_path()
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(line + "\n")
