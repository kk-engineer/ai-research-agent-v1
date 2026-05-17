import asyncio
import time

from ddgs import DDGS

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry

_TIME_MAP = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
    "all": None,
}


class DuckDuckGoRetriever(BaseRetriever):
    name = "duckduckgo"
    supports_modes = ["general", "hybrid"]

    @staticmethod
    def _time_window_to_timelimit(time_window: str) -> str | None:
        return _TIME_MAP.get(time_window, "y")

    async def fetch(
        self, queries: list[str], max_results: int = 10, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("duckduckgo", "Circuit breaker open, skipping")
            return []

        timelimit = self._time_window_to_timelimit(time_window)
        per_query = max(2, max_results // max(len(queries), 1))

        await log_info("duckduckgo", f"timelimit={timelimit} | sub-queries: {len(queries)}")

        tasks = [self._search(q, per_query, timelimit) for q in queries]
        nested = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)

        self._record_success()
        return deduped[:max_results]

    @with_retry()
    async def _search(self, query: str, max_results: int, timelimit: str | None) -> list[RawResult]:
        t0 = time.perf_counter()

        def _run() -> list[RawResult]:
            batch: list[RawResult] = []
            try:
                kw: dict[str, str | int] = {"max_results": max_results}
                if timelimit:
                    kw["timelimit"] = timelimit
                with DDGS() as ddgs:
                    for i, result in enumerate(ddgs.text(query, **kw)):
                        if i >= max_results:
                            break
                        url = result.get("href", "")
                        snippet = result.get("body", "")[:500]
                        title = result.get("title", "")
                        batch.append(
                            RawResult(
                                id=RawResult.make_id(url),
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="duckduckgo",
                            )
                        )
                        log_info(
                            "duckduckgo",
                            f"Result #{i+1}: title=\"{title[:60]}\" url=\"{url[:60]}\"",
                        )
            except Exception as e:
                log_info("duckduckgo", f"Search error: {e}")
            return batch

        loop = asyncio.get_event_loop()
        batch = await loop.run_in_executor(None, _run)

        latency = (time.perf_counter() - t0) * 1000
        await log_info(
            "duckduckgo",
            f"\"{query[:80]}\" → {len(batch)} results timelimit={timelimit} in {fmt_ms(latency)}",
        )
        return batch
