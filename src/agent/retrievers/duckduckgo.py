import asyncio
import time
from datetime import datetime

from ddgs import DDGS

from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry
from agent.logger import fmt_ms, log_info


TEMPORAL_KEYWORDS = {
    "last", "recent", "latest", "this week", "this month", "this year",
    "past week", "past month", "past year", "new", "upcoming",
}


class DuckDuckGoRetriever(BaseRetriever):
    name = "duckduckgo"
    supports_modes = ["general", "hybrid"]

    async def fetch(self, queries: list[str], max_results: int = 10, **kwargs) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("duckduckgo", "Circuit breaker open, skipping")
            return []

        timelimit = kwargs.get("timelimit")
        if timelimit is None:
            all_text = " ".join(queries).lower()
            if any(kw in all_text for kw in TEMPORAL_KEYWORDS):
                timelimit = "m"
            else:
                timelimit = "y"

        await log_info("duckduckgo", f"timelimit={timelimit} | sub-queries: {len(queries)}")

        seen: set[str] = set()
        results: list[RawResult] = []

        for query in queries:
            batch = await self._search(query, max_results, timelimit)
            for r in batch:
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)

        self._record_success()
        return results

    @with_retry()
    async def _search(self, query: str, max_results: int, timelimit: str) -> list[RawResult]:
        t0 = time.perf_counter()
        current_year = datetime.now().year

        def _run() -> list[RawResult]:
            batch: list[RawResult] = []
            try:
                with DDGS() as ddgs:
                    for i, result in enumerate(ddgs.text(query, max_results=max_results, timelimit=timelimit)):
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
                        result_preview = f"title=\"{title[:60]}\" url=\"{url[:60]}\""
                        log_info("duckduckgo", f"Result #{i+1}: {result_preview}")
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
