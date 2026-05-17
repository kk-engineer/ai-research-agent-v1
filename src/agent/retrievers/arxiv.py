import asyncio
import time
from typing import Any

import arxiv

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry


class ArxivRetriever(BaseRetriever):
    name = "arxiv"
    supports_modes = ["academic", "hybrid"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    async def fetch(
        self, queries: list[str], max_results: int = 15, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("arxiv", "Circuit breaker open, skipping")
            return []

        per_query = max(2, max_results // max(len(queries), 1))
        tasks = [self._search(q, per_query) for q in queries]
        nested = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)

        self._record_success()
        return deduped[:max_results]

    @with_retry()
    async def _search(self, query: str, max_results: int) -> list[RawResult]:
        t0 = time.perf_counter()

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        def _run() -> list[RawResult]:
            raw_results: list[RawResult] = []
            for paper in self._client.results(search):
                raw_results.append(
                    RawResult(
                        id=RawResult.make_id(paper.entry_id),
                        title=paper.title,
                        url=paper.entry_id,
                        snippet=paper.summary[:500],
                        source="arxiv",
                        published_at=paper.published,
                        authors=[str(a) for a in paper.authors],
                        categories=[t for t in paper.categories],
                    )
                )
            return raw_results

        loop = asyncio.get_event_loop()
        batch = await loop.run_in_executor(None, _run)

        latency = (time.perf_counter() - t0) * 1000
        await log_info("arxiv", f"Query: \"{query[:80]}\"")
        for i, r in enumerate(batch):
            await log_info("arxiv", f"  #{i+1}: \"{r.title[:70]}\" | {r.published_at}")
        await log_info("arxiv", f"Total: {len(batch)} papers in {fmt_ms(latency)}")
        return batch
