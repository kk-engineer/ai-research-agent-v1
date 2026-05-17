import asyncio
import email.utils
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

import feedparser

from agent.config import AppConfig
from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry

_MIN_RELEVANCE = 0.15
_TIME_SECONDS = {
    "day": 86400,
    "week": 604800,
    "month": 2_592_000,
    "year": 31_536_000,
}


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

    async def fetch(
        self, queries: list[str], max_results: int = 15, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("rss", "Circuit breaker open, skipping")
            return []

        feed_tasks = [self._parse_feed(feed_url, time_window) for feed_url in self.feeds]
        nested = await asyncio.gather(*feed_tasks, return_exceptions=True)
        all_entries = [r for batch in nested if isinstance(batch, list) for r in batch]

        query_terms = set(queries[0].lower().split()) if queries else set()
        scored = []
        for entry in all_entries:
            text = (entry.title + " " + entry.snippet).lower()
            relevance = self._score_entry(text, query_terms) if query_terms else 1.0
            if relevance >= _MIN_RELEVANCE:
                scored.append((relevance, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:max_results]]

        self._record_success()
        return results

    @staticmethod
    def _score_entry(text: str, query_terms: set[str]) -> float:
        hits = sum(1 for t in query_terms if t in text)
        return hits / max(len(query_terms), 1)

    @staticmethod
    def _cutoff_timestamp(time_window: str) -> float | None:
        if time_window == "all":
            return None
        seconds = _TIME_SECONDS.get(time_window)
        if seconds is None:
            return None
        return time.time() - seconds

    @with_retry()
    async def _parse_feed(self, feed_url: str, time_window: str = "all") -> list[RawResult]:
        t0 = time.perf_counter()
        cutoff = self._cutoff_timestamp(time_window)

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
                if cutoff is not None and published is not None:
                    if published.timestamp() < cutoff:
                        continue
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
