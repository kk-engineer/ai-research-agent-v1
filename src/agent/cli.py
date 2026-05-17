import asyncio
import os
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.cache import AsyncCache
from agent.config import load_config
from agent.logger import FileLogWriter
from agent.pipeline import Pipeline
from agent.startup import validate_connectivity

console = Console()

_shutdown_flag = False


def _setup_signal_handlers():
    global _shutdown_flag

    def handle_sigint(signum, frame):
        global _shutdown_flag
        if _shutdown_flag:
            console.print("\n[red]Forced exit[/red]")
            os._exit(1)
        _shutdown_flag = True
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")

    signal.signal(signal.SIGINT, handle_sigint)


class InteractiveShell:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self.config = None
        self.pipeline = None
        self.file_logger = None
        self._shutdown = False

    async def _initialize(self):
        self.config = load_config(Path(self.config_path))
        self.file_logger = FileLogWriter()
        await self.file_logger.start()
        await validate_connectivity(self.config)
        self.pipeline = Pipeline(self.config)

    def _show_banner(self):
        console.print()
        console.print(Panel.fit(
            "[bold blue]AI Research Agent[/bold blue]\n"
            "[dim]Interactive Shell[/dim]\n\n"
            "Type a research query or use [bold]/help[/bold] for commands.\n"
            "Press [bold]Ctrl+C[/bold] to exit gracefully.",
            border_style="blue",
        ))
        console.print()

    async def _run_pipeline(self, query: str, no_cache: bool = False, export_json: bool = False):
        if not self.pipeline:
            await self._initialize()

        try:
            report = await self.pipeline.run(query)

            console.print()
            console.print(Panel.fit(
                "[bold green]Research Report[/bold green]",
                border_style="green",
            ))
            console.print()
            console.print(Markdown(report.markdown))

            stats = report.stats
            stats_dict = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)

            table = Table(title="Pipeline Statistics")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="white")

            skip_keys = {"retriever_latencies"}
            for key, value in stats_dict.items():
                if key in skip_keys or isinstance(value, dict):
                    continue
                label = key.replace("_", " ").title()
                if key.endswith("_ms") and isinstance(value, (int, float)):
                    seconds = value / 1000
                    table.add_row(label.replace(" Ms", ""), f"{seconds:.2f}s")
                elif isinstance(value, float):
                    table.add_row(label, f"{value:,.2f}")
                else:
                    table.add_row(label, str(value))
            console.print(table)

            report_path = report.save(Path(self.config.output.reports_dir))
            console.print(f"\n  [green]✔ Report saved →[/green] {report_path}")

            if export_json:
                json_path = report.to_json(Path(self.config.output.reports_dir))
                console.print(f"  [green]✔ JSON exported →[/green] {json_path}")

            return report

        except Exception as e:
            console.print(f"\n  [red]✖ Error:[/red] {e}")
            raise

    async def _clear_cache(self):
        if not self.config:
            await self._initialize()
        cache = AsyncCache(self.config)
        await cache.clear()
        console.print("  [green]✔ Cache cleared[/green]")

    async def _cache_stats(self):
        if not self.config:
            await self._initialize()
        cache = AsyncCache(self.config)
        stats = await cache.stats()
        table = Table(title="Cache Statistics")
        table.add_column("Level", style="bold")
        table.add_column("Entries", style="white")
        table.add_row("L1 (chunks)", str(stats['l1_entries']))
        table.add_row("L2 (reports)", str(stats['l2_entries']))
        console.print(table)

    def _show_help(self):
        help_panel = Panel.fit(
            "[bold]Available Commands[/bold]\n\n"
            "[bold]/help[/bold]      — Show this help message\n"
            "[bold]/clear[/bold]     — Clear the cache\n"
            "[bold]/stats[/bold]     — Show cache statistics\n"
            "[bold]/config[/bold]    — Show current configuration\n"
            "[bold]/exit[/bold]      — Exit the application\n\n"
            "[bold]Query Prefixes[/bold]\n\n"
            "[bold]--no-cache[/bold] <query>     — Bypass cache for this query\n"
            "[bold]--export-json[/bold] <query>  — Also export as JSON\n\n"
            "Simply type a [bold]natural language query[/bold] to start research.\n"
            "Pipeline progress is streamed in real-time via the integrated logger.",
            border_style="blue",
        )
        console.print(help_panel)

    async def _show_config(self):
        if not self.config:
            await self._initialize()
        dump = self.config.model_dump()
        if "api_keys" in dump:
            masked = {}
            for k, v in dump["api_keys"].items():
                masked[k] = v[:4] + "****" if v and len(v) > 4 else "****" if v else ""
            dump["api_keys"] = masked
        import json
        console.print_json(json.dumps(dump, indent=2, default=str))

    async def run(self):
        await self._initialize()
        self._show_banner()

        while not self._shutdown and not _shutdown_flag:
            try:
                raw = input(">>> ").strip()
            except EOFError:
                console.print("\nExiting...")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                cmd = raw[1:].lower()
                if cmd == "exit":
                    console.print("Exiting...")
                    break
                elif cmd == "help":
                    self._show_help()
                elif cmd == "clear":
                    await self._clear_cache()
                elif cmd == "stats":
                    await self._cache_stats()
                elif cmd == "config":
                    await self._show_config()
                else:
                    console.print(f"  [yellow]⚠ Unknown command:[/yellow] {cmd}")
                continue

            no_cache = False
            export_json = False

            if raw.startswith("--no-cache "):
                raw = raw[11:]
                no_cache = True
            if raw.startswith("--export-json "):
                raw = raw[16:]
                export_json = True

            try:
                await self._run_pipeline(raw, no_cache=no_cache, export_json=export_json)
            except asyncio.CancelledError:
                console.print("\nOperation cancelled.")
                break
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        if self.file_logger:
            await self.file_logger.stop()


def main():
    _setup_signal_handlers()
    shell = InteractiveShell()

    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except SystemExit:
        pass
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
