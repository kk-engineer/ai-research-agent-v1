import asyncio
import time
from typing import Any

import semanticscholar as sch

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry


class SemanticScholarRetriever(BaseRetriever):
    name = "semantic_scholar"
    supports_modes = ["academic", "hybrid"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = sch.SemanticScholar()

    async def fetch(
        self, queries: list[str], max_results: int = 15, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("semantic_scholar", "Circuit breaker open, skipping")
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

        def _run() -> list[RawResult]:
            raw_results: list[RawResult] = []
            search_results = self._client.search_paper(query, limit=max_results)
            for paper in search_results or []:
                raw_results.append(
                    RawResult(
                        id=RawResult.make_id(paper.paperId or paper.title),
                        title=paper.title or "",
                        url=paper.url or f"https://www.semanticscholar.org/paper/{paper.paperId}",
                        snippet=(paper.tldr or {}).get("text", paper.abstract or "")[:500],
                        source="semantic_scholar",
                        published_at=getattr(paper, "publicationDate", None),
                        authors=paper.authors or [],
                        categories=[],
                    )
                )
            return raw_results

        loop = asyncio.get_event_loop()
        batch = await loop.run_in_executor(None, _run)

        latency = (time.perf_counter() - t0) * 1000
        await log_info("semantic_scholar", f"Query: \"{query[:80]}\"")
        for i, r in enumerate(batch):
            await log_info(
                "semantic_scholar",
                f"  #{i+1}: \"{r.title[:70]}\" | published={r.published_at}",
            )
        await log_info("semantic_scholar", f"Total: {len(batch)} papers in {fmt_ms(latency)}")
        return batch
