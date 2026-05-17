import time

import httpx

from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry
from agent.logger import fmt_ms, log_info


class HackerNewsRetriever(BaseRetriever):
    name = "hackernews"
    supports_modes = ["general", "hybrid"]
    BASE_URL = "https://hn.algolia.com/api/v1/search"

    async def fetch(self, queries: list[str], max_results: int = 10) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("hackernews", "Circuit breaker open, skipping")
            return []

        seen: set[str] = set()
        results: list[RawResult] = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for query in queries:
                batch = await self._search(client, query, max_results)
                for r in batch:
                    if r.id not in seen:
                        seen.add(r.id)
                        results.append(r)

        self._record_success()
        return results

    @with_retry()
    async def _search(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[RawResult]:
        t0 = time.perf_counter()
        results: list[RawResult] = []

        params = {
            "query": query,
            "tags": "story",
            "hitsPerPage": max_results,
        }
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
            await log_info("hackernews", f"  #{i+1}: \"{r.title[:70]}\" | author={r.authors[0] if r.authors else '?'} | date={r.published_at}")
        await log_info("hackernews", f"Total: {len(results)} stories in {fmt_ms(latency)}")
        return results
