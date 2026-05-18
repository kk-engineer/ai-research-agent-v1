import asyncio
import time

from ddgs import DDGS

from agent.logger import fmt_ms, log_info
from agent.models.result import RawResult
from agent.retrievers.base import BaseRetriever, with_retry

_TIME_MAP = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
    "all": None,
}


class DuckDuckGoRetriever(BaseRetriever):
    name = "duckduckgo"
    supports_modes = ["general", "hybrid"]

    @staticmethod
    def _time_window_to_timelimit(time_window: str) -> str | None:
        return _TIME_MAP.get(time_window, "y")

    async def fetch(
        self, queries: list[str], max_results: int = 10, time_window: str = "all"
    ) -> list[RawResult]:
        if self._is_circuit_open():
            await log_info("duckduckgo", "Circuit breaker open, skipping")
            return []

        timelimit = self._time_window_to_timelimit(time_window)
        per_query = max(2, max_results // max(len(queries), 1))

        await log_info("duckduckgo", f"timelimit={timelimit} | sub-queries: {len(queries)}")

        tasks = [self._search(q, per_query, timelimit) for q in queries]
        nested = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for batch in nested if isinstance(batch, list) for r in batch]
        deduped = self._deduplicate(results)

        self._record_success()
        return deduped[:max_results]

    @with_retry()
    async def _search(self, query: str, max_results: int, timelimit: str | None) -> list[RawResult]:
        t0 = time.perf_counter()

        def _run() -> list[RawResult]:
            batch: list[RawResult] = []
            try:
                kw: dict[str, str | int] = {"max_results": max_results}
                if timelimit:
                    kw["timelimit"] = timelimit
                with DDGS() as ddgs:
                    for i, result in enumerate(ddgs.text(query, **kw)):
                        if i >= max_results:
                            break
                        url = result.get("href", "")
                        snippet = result.get("body", "")[:500]
                        title = result.get("title", "")
                        batch.append(
                            RawResult(
                                id=RawResult.make_id(url),
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="duckduckgo",
                            )
                        )
                        log_info(
                            "duckduckgo",
                            f"Result #{i+1}: title=\"{title[:60]}\" url=\"{url[:60]}\"",
                        )
            except Exception as e:
                log_info("duckduckgo", f"Search error: {e}")
            return batch

        loop = asyncio.get_event_loop()
        batch = await loop.run_in_executor(None, _run)

        latency = (time.perf_counter() - t0) * 1000
        await log_info(
            "duckduckgo",
            f"\"{query[:80]}\" → {len(batch)} results timelimit={timelimit} in {fmt_ms(latency)}",
        )
        return batch


    async def _fetch_one(
            self, query: str, max_results: int, time_window: str
    ) -> list[RawResult]:
        # Choose search method based on query intent
        use_news = time_window in ("day", "week", "month") or self._is_news_query(query)

        def _run() -> list[dict]:
            with DDGS() as ddgs:
                if use_news:
                    # news() returns individual article URLs with published dates
                    return list(ddgs.news(
                        query,
                        max_results=max_results,
                        # timelimit: 'd'=day, 'w'=week, 'm'=month
                        timelimit={"day": "d", "week": "w", "month": "m"}.get(time_window, "m"),
                    ))
                else:
                    return list(ddgs.text(
                        query,
                        max_results=max_results,
                        timelimit={"year": "y", "month": "m"}.get(time_window),
                    ))

        results = await asyncio.to_thread(_run)

        raw = []
        for r in results:
            # ddgs.news() uses 'url' key; ddgs.text() uses 'href'
            url = r.get("url") or r.get("href", "")
            if not url:
                continue
            # ddgs.news() returns 'date' as ISO string
            published_at = None
            if date_str := r.get("date"):
                with contextlib.suppress(Exception):
                    published_at = datetime.fromisoformat(date_str)

            raw.append(RawResult(
                id=RawResult.make_id(url),
                title=r.get("title", ""),
                url=url,
                snippet=r.get("body", ""),
                source="duckduckgo",
                published_at=published_at,  # ← news() provides this; text() does not
            ))
        return raw

    def _is_news_query(self, query: str) -> bool:
        news_signals = {"news", "happening", "latest", "today", "now", "current", "update"}
        return bool(news_signals & set(query.lower().split()))