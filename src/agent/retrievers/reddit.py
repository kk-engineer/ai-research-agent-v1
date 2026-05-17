import time
from datetime import UTC, datetime

from httpx import AsyncClient

from agent.config import AppConfig
from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry


class RedditRetriever(BaseRetriever):
    name = "reddit"
    supports_modes = ["general", "hybrid"]

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.subreddits = config.retrievers.reddit.subreddits
        self.feed_type = config.retrievers.reddit.feed_type

    async def fetch(self, queries: list[str], max_results: int) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("reddit", "Circuit breaker open, skipping")
            return []

        seen: set[str] = set()
        results: list[RawResult] = []
        per_sub = max(5, max_results // max(len(self.subreddits), 1))

        async with AsyncClient(timeout=15, follow_redirects=True) as client:
            for sub in self.subreddits:
                t0 = time.perf_counter()
                try:
                    chunk = await self._fetch_subreddit(client, sub, per_sub)
                    latency = (time.perf_counter() - t0) * 1000
                    await log_info(
                        self.name,
                        f"r/{sub} ({self.feed_type}) → {len(chunk)} posts in {fmt_ms(latency)}",
                    )
                    for r in chunk:
                        if r.id not in seen:
                            seen.add(r.id)
                            results.append(r)
                except Exception as e:
                    latency = (time.perf_counter() - t0) * 1000
                    await log_info(
                        self.name,
                        f"r/{sub} failed in {fmt_ms(latency)}: {e}",
                    )
                    continue

        query_keywords = set(" ".join(queries).lower().split())
        scored = []
        for r in results:
            text = (r.title + " " + r.snippet).lower()
            match_count = sum(1 for kw in query_keywords if kw in text)
            scored.append((match_count, r))
        scored.sort(key=lambda x: -x[0])
        results = [r for _, r in scored[:max_results]]

        self._record_success()
        return results

    @with_retry()
    async def _fetch_subreddit(
        self, client: AsyncClient, sub: str, limit: int
    ) -> list[RawResult]:
        params = {"limit": limit, "raw_json": "1"}
        if self.feed_type == "new":
            params["sort"] = "new"
        if self.feed_type == "top":
            params["t"] = "week"

        url = f"https://www.reddit.com/r/{sub}/{self.feed_type}.json"
        resp = await client.get(url, params=params, headers=self.HEADERS)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_posts(data, sub)

    def _parse_posts(self, data: dict, sub: str) -> list[RawResult]:
        results: list[RawResult] = []
        children = data.get("data", {}).get("children", [])

        for child in children:
            kind = child.get("kind", "")
            if kind != "t3":
                continue
            post = child.get("data", {})
            title = post.get("title", "")
            permalink = post.get("permalink", "")
            url = f"https://www.reddit.com{permalink}"

            selftext = post.get("selftext", "")
            snippet = selftext[:800] if selftext else post.get("url", "")

            created = post.get("created_utc")
            published_at = (
                datetime.fromtimestamp(created, tz=UTC) if created else None
            )

            score = post.get("score", 0)
            num_comments = post.get("num_comments", 0)
            author = post.get("author", "")
            subreddit = post.get("subreddit", sub)

            meta = f"Score: {score} | Comments: {num_comments} | r/{subreddit}"
            snippet = f"{snippet} [{meta}]" if snippet else meta

            results.append(
                RawResult(
                    id=RawResult.make_id(url),
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                    published_at=published_at,
                    authors=[author] if author else [],
                    categories=[f"r/{subreddit}", self.feed_type],
                )
            )

        return results
