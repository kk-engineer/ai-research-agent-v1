import asyncio
import re
import time

from httpx import AsyncClient

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry

_GITHUB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_REPO_PATTERN = re.compile(r"^/[^/]+/[^/]+$")


class GitHubSearchRetriever(BaseRetriever):
    name = "github_search"
    supports_modes = ["general", "hybrid"]

    BASE_URL = "https://github.com/search"

    async def fetch(
        self, queries: list[str], max_results: int = 15, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("github_search", "Circuit breaker open, skipping")
            return []

        per_query = max(2, max_results // max(len(queries), 1))

        async with AsyncClient(timeout=15, follow_redirects=True) as client:
            tasks = [self._search(client, q, per_query) for q in queries]
            nested = await asyncio.gather(*tasks, return_exceptions=True)

        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)
        return deduped[:max_results]

    @with_retry()
    async def _search(
        self, client: AsyncClient, query: str, limit: int
    ) -> list[RawResult]:
        t0 = time.perf_counter()
        params = {"q": query, "type": "repositories"}
        resp = await client.get(self.BASE_URL, params=params, headers=_GITHUB_HEADERS)
        resp.raise_for_status()
        chunk = self._parse_html(resp.text, limit)
        latency = (time.perf_counter() - t0) * 1000
        await log_info(
            self.name,
            f"q=\"{query[:60]}\" → {len(chunk)} results in {fmt_ms(latency)}",
        )
        return chunk

    def _parse_html(self, html: str, limit: int) -> list[RawResult]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        results: list[RawResult] = []

        items = soup.select('[data-testid="results-list"] > div')
        for item in items[:limit]:
            name_el = item.select_one(".search-title a")
            if not name_el:
                continue

            href = name_el.get("href", "")
            if not _REPO_PATTERN.match(href):
                continue

            title = name_el.text.strip()
            url = "https://github.com" + href

            desc_el = item.select_one("h3 + div span")
            desc = desc_el.text.strip() if desc_el else ""

            lang_el = item.select_one("li span[aria-label]")
            language = lang_el.text.strip() if lang_el else ""

            topics = [
                a.text.strip()
                for a in item.select('a[href^="/topics/"]')
            ]

            snippet_parts = [desc] if desc else []
            if language:
                snippet_parts.append(f"Language: {language}")
            if topics:
                snippet_parts.append(f"Topics: {', '.join(topics[:5])}")
            snippet = " | ".join(snippet_parts)

            results.append(
                RawResult(
                    id=RawResult.make_id(url),
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                )
            )

        return results
