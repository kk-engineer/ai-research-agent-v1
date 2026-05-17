import asyncio

from httpx import AsyncClient

from agent.logger import log_info
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

_LANGUAGES = {
    "python", "javascript", "typescript", "rust", "go", "java",
    "c++", "c", "csharp", "ruby", "swift", "kotlin", "scala",
    "php", "perl", "r", "dart", "lua", "haskell", "elixir",
    "clojure", "solidity", "zig", "mojo",
}


class GitHubTrendingRetriever(BaseRetriever):
    name = "github_trending"
    supports_modes = ["general", "hybrid"]

    BASE_URL = "https://github.com/trending"

    async def fetch(
        self, queries: list[str], max_results: int = 15, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("github_trending", "Circuit breaker open, skipping")
            return []

        urls = self._urls_to_fetch(queries)

        async with AsyncClient(timeout=15, follow_redirects=True) as client:
            tasks = [self._fetch_page(client, url, max_results) for url in urls]
            nested = await asyncio.gather(*tasks, return_exceptions=True)

        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)

        self._record_success()
        return deduped[:max_results]

    def _urls_to_fetch(self, queries: list[str]) -> list[str]:
        urls = [self.BASE_URL]
        terms = " ".join(queries).lower().split()
        matched = {t for t in terms if t in _LANGUAGES}
        for lang in matched:
            urls.append(f"{self.BASE_URL}/{lang}")
        return urls

    @with_retry()
    async def _fetch_page(
        self, client: AsyncClient, url: str, limit: int
    ) -> list[RawResult]:
        resp = await client.get(url, headers=_GITHUB_HEADERS)
        resp.raise_for_status()
        chunk = self._parse_html(resp.text, limit)
        await log_info("github_trending", f"Fetched {len(chunk)} from {url}")
        return chunk

    def _parse_html(self, html: str, limit: int) -> list[RawResult]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        results: list[RawResult] = []

        for article in soup.select("article.Box-row")[:limit]:
            h2_el = article.select_one("h2 a")
            if not h2_el:
                continue

            full_name = h2_el["href"].strip("/")
            href = h2_el.get("href", "")
            url = "https://github.com" + href

            desc_el = article.select_one("p")
            desc = desc_el.text.strip() if desc_el else ""

            lang_el = article.select_one("span[itemprop='programmingLanguage']")
            language = lang_el.text.strip() if lang_el else ""

            stars_today_el = article.select_one(".d-inline-block.float-sm-right")
            stars_today = stars_today_el.text.strip() if stars_today_el else ""

            meta_parts = []
            if language:
                meta_parts.append(f"Language: {language}")
            if stars_today:
                meta_parts.append(f"Stars today: {stars_today}")

            snippet = desc
            if meta_parts:
                snippet = desc + " | " + " | ".join(meta_parts) if desc else " | ".join(meta_parts)

            results.append(
                RawResult(
                    id=RawResult.make_id(url),
                    title=full_name,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                )
            )

        return results
