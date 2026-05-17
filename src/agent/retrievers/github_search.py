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

    async def fetch(self, queries: list[str], max_results: int) -> list[RawResult]:
        results: list[RawResult] = []
        seen: set[str] = set()
        per_query = max(2, max_results // max(len(queries), 1))

        async with AsyncClient(timeout=15, follow_redirects=True) as client:
            for query in queries:
                t0 = time.perf_counter()
                try:
                    chunk = await self._search(client, query, per_query)
                    latency = (time.perf_counter() - t0) * 1000
                    await log_info(
                        self.name,
                        f"q=\"{query[:60]}\" → {len(chunk)} results in {fmt_ms(latency)}",
                    )
                    for r in chunk:
                        if r.id not in seen:
                            seen.add(r.id)
                            results.append(r)
                except Exception as e:
                    latency = (time.perf_counter() - t0) * 1000
                    await log_info(
                        self.name,
                        f"q=\"{query[:60]}\" failed in {fmt_ms(latency)}: {e}",
                    )
                    continue

        return results[:max_results]

    @with_retry()
    async def _search(
        self, client: AsyncClient, query: str, limit: int
    ) -> list[RawResult]:
        params = {"q": query, "type": "repositories"}
        resp = await client.get(self.BASE_URL, params=params, headers=_GITHUB_HEADERS)
        resp.raise_for_status()
        return self._parse_html(resp.text, limit)

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
