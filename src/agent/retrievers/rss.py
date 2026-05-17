import asyncio
import email.utils
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

import feedparser

from agent.config import AppConfig
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry
from agent.logger import fmt_ms, log_info


class RSSRetriever(BaseRetriever):
    name = "rss"
    supports_modes = ["general", "hybrid"]

    DEFAULT_FEEDS: ClassVar[list[str]] = [
        "https://rss.arxiv.org/rss/cs.AI",
        "https://rss.arxiv.org/rss/cs.LG",
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://venturebeat.com/ai/feed/",
        "https://huggingface.co/blog/feed.xml",
        "https://deepmind.google/blog/rss.xml",
    ]

    def __init__(self, config: AppConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feeds = config.retrievers.rss_feeds.urls or self.DEFAULT_FEEDS

    async def fetch(self, queries: list[str], max_results: int = 15) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("rss", "Circuit breaker open, skipping")
            return []

        seen: set[str] = set()
        results: list[RawResult] = []

        for feed_url in self.feeds:
            batch = await self._parse_feed(feed_url)
            for r in batch:
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)

        query_keywords = set(" ".join(queries).lower().split())
        scored = []
        for r in results:
            text = (r.title + " " + r.snippet).lower()
            score = sum(1 for kw in query_keywords if kw in text)
            scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        results = [r for _, r in scored[:max_results]]

        self._record_success()
        return results

    @with_retry()
    async def _parse_feed(self, feed_url: str) -> list[RawResult]:
        t0 = time.perf_counter()

        def _parse() -> list[RawResult]:
            parsed = feedparser.parse(feed_url)
            batch: list[RawResult] = []
            for entry in parsed.entries[:20]:
                url = entry.get("link", "")
                if not url:
                    continue
                published = None
                if entry.get("published"):
                    try:
                        parsed_date = email.utils.parsedate_to_datetime(entry.published)
                        published = parsed_date.replace(tzinfo=UTC)
                    except Exception:
                        published = datetime.now(UTC)
                batch.append(
                    RawResult(
                        id=RawResult.make_id(url),
                        title=entry.get("title", ""),
                        url=url,
                        snippet=entry.get("summary", "")[:500],
                        source="rss",
                        published_at=published,
                        authors=[a.get("name", "") for a in entry.get("authors", [])],
                        categories=entry.get("tags", []),
                    )
                )
            return batch

        loop = asyncio.get_event_loop()
        batch = await loop.run_in_executor(None, _parse)
        latency = (time.perf_counter() - t0) * 1000
        feed_name = feed_url.split("/")[2] if "//" in feed_url else feed_url
        await log_info("rss", f"Feed: {feed_name} | {len(batch)} entries in {fmt_ms(latency)}")
        for i, r in enumerate(batch[:3]):
            await log_info("rss", f"  #{i+1}: \"{r.title[:60]}\" | date={r.published_at}")
        return batch
