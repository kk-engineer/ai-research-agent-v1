import asyncio
import time

import httpx

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry


class WikipediaRetriever(BaseRetriever):
    name = "wikipedia"
    supports_modes = ["general", "hybrid", "academic"]
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    async def fetch(
        self, queries: list[str], max_results: int = 5, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("wikipedia", "Circuit breaker open, skipping")
            return []

        per_query = max(1, max_results // max(len(queries), 1))

        headers = {"User-Agent": "AIResearchAgent/1.0 (research agent; karan@example.com)"}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            tasks = [self._search(client, q, per_query) for q in queries]
            nested = await asyncio.gather(*tasks, return_exceptions=True)

        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)

        self._record_success()
        return deduped[:max_results]

    @with_retry()
    async def _search(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[RawResult]:
        t0 = time.perf_counter()
        results: list[RawResult] = []

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }
        resp = await client.get(self.SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        search_results = data.get("query", {}).get("search", [])
        for entry in search_results[:max_results]:
            title = entry.get("title", "")
            summary = await self._get_summary(client, title)

            results.append(
                RawResult(
                    id=RawResult.make_id(f"https://en.wikipedia.org/wiki/{title}"),
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title}",
                    snippet=(entry.get("snippet", "") or summary)[:500],
                    source="wikipedia",
                    categories=[],
                )
            )

        latency = (time.perf_counter() - t0) * 1000
        await log_info("wikipedia", f"Query: \"{query[:80]}\"")
        for i, r in enumerate(results):
            await log_info("wikipedia", f"  #{i+1}: \"{r.title[:70]}\"")
        await log_info("wikipedia", f"Total: {len(results)} articles in {fmt_ms(latency)}")
        return results

    async def _get_summary(self, client: httpx.AsyncClient, title: str) -> str:
        try:
            url = self.SUMMARY_URL.format(title=title)
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data.get("extract", "")[:500]
        except Exception:
            return ""
