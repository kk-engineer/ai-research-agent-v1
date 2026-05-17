import asyncio
import time

import httpx

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry

_TIME_SECONDS = {
    "day": 86400,
    "week": 604800,
    "month": 2_592_000,
    "year": 31_536_000,
}


class HackerNewsRetriever(BaseRetriever):
    name = "hackernews"
    supports_modes = ["general", "hybrid"]
    BASE_URL = "https://hn.algolia.com/api/v1/search"

    async def fetch(
        self, queries: list[str], max_results: int = 10, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("hackernews", "Circuit breaker open, skipping")
            return []

        cutoff = self._cutoff_timestamp(time_window)
        per_query = max(2, max_results // max(len(queries), 1))

        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = [self._search(client, q, per_query, cutoff) for q in queries]
            nested = await asyncio.gather(*tasks, return_exceptions=True)

        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)

        self._record_success()
        return deduped[:max_results]

    @staticmethod
    def _cutoff_timestamp(time_window: str) -> int | None:
        if time_window == "all":
            return None
        seconds = _TIME_SECONDS.get(time_window)
        if seconds is None:
            return None
        return int(time.time() - seconds)

    @with_retry()
    async def _search(
        self, client: httpx.AsyncClient, query: str, max_results: int, cutoff: int | None
    ) -> list[RawResult]:
        t0 = time.perf_counter()
        results: list[RawResult] = []

        params: dict = {
            "query": query,
            "tags": "story",
            "hitsPerPage": max_results,
        }
        if cutoff is not None:
            params["numericFilters"] = f"created_at_i>{cutoff}"

        resp = await client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        for hit in data.get("hits", []):
            object_id = hit.get("objectID", "")
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
            created_at = hit.get("created_at", None)
            results.append(
                RawResult(
                    id=RawResult.make_id(url),
                    title=hit.get("title", ""),
                    url=url,
                    snippet=hit.get("title", ""),
                    source="hackernews",
                    published_at=created_at,
                    authors=[hit.get("author", "")] if hit.get("author") else [],
                    categories=[],
                )
            )

        latency = (time.perf_counter() - t0) * 1000
        await log_info("hackernews", f"Query: \"{query[:80]}\"")
        for i, r in enumerate(results):
            author = r.authors[0] if r.authors else "?"
            await log_info(
                "hackernews",
                f"  #{i+1}: \"{r.title[:70]}\" | author={author} | date={r.published_at}",
            )
        await log_info("hackernews", f"Total: {len(results)} stories in {fmt_ms(latency)}")
        return results
