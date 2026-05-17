import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Literal

from agent.logger import log_warning
from agent.models.result import RawResult


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True
    retry_on_status: list[int] = field(default_factory=lambda: [429, 500, 502, 503])


@dataclass
class CircuitBreakerState:
    state: Literal["closed", "open", "half_open"] = "closed"
    failure_count: int = 0
    last_failure_time: float = 0.0
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0


class BaseRetriever(ABC):
    name: str = ""
    supports_modes: list[str] = []
    _circuit: CircuitBreakerState

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._circuit = CircuitBreakerState()

    @abstractmethod
    async def fetch(self, queries: list[str], max_results: int) -> list[RawResult]:
        ...

    async def health_check(self) -> bool:
        return True

    def _is_circuit_open(self) -> bool:
        if self._circuit.state == "closed":
            return False
        if self._circuit.state == "open":
            elapsed = time.time() - self._circuit.last_failure_time
            if elapsed > self._circuit.recovery_timeout_s:
                self._circuit.state = "half_open"
                return False
            return True
        return False

    def _record_failure(self) -> None:
        self._circuit.failure_count += 1
        self._circuit.last_failure_time = time.time()
        if self._circuit.failure_count >= self._circuit.failure_threshold:
            self._circuit.state = "open"

    def _record_success(self) -> None:
        self._circuit.failure_count = 1
        self._circuit.state = "closed"


def with_retry(config: RetryConfig | None = None) -> Callable:
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    status = getattr(e, "response", None)
                    status_code = getattr(status, "status_code", 0) if status else 0
                    if status_code in (400, 401, 403, 404):
                        raise
                    if status_code and status_code not in config.retry_on_status:
                        raise
                    if attempt == config.max_retries:
                        raise
                    delay = config.base_delay_s * (2 ** attempt)
                    delay = min(delay, config.max_delay_s)
                    if config.jitter:
                        delay *= 0.5 + random.random() * 0.5
                    await log_warning(
                        "retry",
                        (
                            f"Attempt {attempt + 1}/{config.max_retries + 1} "
                            f"failed for {func.__name__}: {e}. Retrying in {delay:.1f}s"
                        ),
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
