import asyncio
from abc import ABC, abstractmethod
from uuid import uuid4

from agent.logger import log_debug, log_warning
from agent.models.result import ExtractedChunk, RawResult

_MIN_WORDS = 60
_SNIPPET_SOURCES = {"arxiv", "semantic_scholar"}


class BaseExtractor(ABC):
    name: str = ""

    @abstractmethod
    async def extract(self, url: str, raw_html: str | None = None) -> ExtractedChunk | None:
        ...


JS_HEAVY_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com",
    "wsj.com", "nytimes.com", "washingtonpost.com",
    "hindustantimes.com", "indianexpress.com", "ndtv.com",
    "thehindu.com", "timesofindia.com",
}


class ExtractorChain:
    def __init__(self, extractors: list[BaseExtractor]) -> None:
        self.extractors = extractors

    @staticmethod
    def _needs_jina(url: str) -> bool:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lstrip("www.")
        return any(domain.endswith(d) for d in JS_HEAVY_DOMAINS)

    async def extract(self, url: str, raw_html: str | None = None) -> ExtractedChunk | None:
        chain = self.extractors
        if self._needs_jina(url):
            chain = [e for e in self.extractors if e.name != "trafilatura"]
            await log_debug("extractor", f"JS-heavy domain, skipping trafilatura: {url[:60]}")
        for extractor in chain:
            try:
                result = await extractor.extract(url, raw_html)
                if result is not None and result.word_count >= _MIN_WORDS:
                    result.extractor_used = extractor.name
                    return result
                elif result is not None:
                    await log_debug(
                        "extractor",
                        f"Rejected {extractor.name} {url[:50]} {result.word_count}w",
                    )
            except Exception:
                continue
        await log_warning("extractor", f"No usable content for {url[:70]}")
        return None

    async def extract_all(
        self,
        results: list[RawResult],
        semaphore: asyncio.Semaphore,
    ) -> list[ExtractedChunk]:
        chunks: list[ExtractedChunk] = []

        for r in results:
            chunk: ExtractedChunk | None = None

            if r.source in _SNIPPET_SOURCES and len(r.snippet) >= 80:
                chunk = self._make_snippet_chunk(r)
                await log_debug("extractor", f"snippet fast-path: {r.url[:60]}")
            else:
                async with semaphore:
                    chunk = await self._extract_with_chain(r)

            if chunk is None and r.snippet:
                chunk = self._make_snippet_chunk(r)
                await log_debug("extractor", f"snippet fallback: {r.url[:60]}")

            if chunk:
                chunks.append(chunk)

        return chunks

    async def _extract_with_chain(self, r: RawResult) -> ExtractedChunk | None:
        chain = self.extractors
        if self._needs_jina(r.url):
            chain = [e for e in self.extractors if e.name != "trafilatura"]
            await log_debug("extractor", f"JS-heavy domain, skipping trafilatura: {r.url[:60]}")
        for extractor in chain:
            try:
                result = await extractor.extract(r.url, r.raw_html)
                if result is not None and result.word_count >= _MIN_WORDS:
                    result.extractor_used = extractor.name
                    return result
                elif result is not None:
                    await log_debug(
                        "extractor",
                        f"Rejected {extractor.name} {r.url[:50]} {result.word_count}w",
                    )
            except Exception:
                continue
        await log_warning("extractor", f"No usable content for {r.url[:70]}")
        return None

    @staticmethod
    def _make_snippet_chunk(r: RawResult) -> ExtractedChunk:
        return ExtractedChunk(
            chunk_id=uuid4().hex,
            source_id=r.id,
            url=r.url,
            title=r.title,
            content_markdown=r.snippet,
            word_count=len(r.snippet.split()),
            extractor_used="snippet",
            extraction_latency_ms=0.0,
            metadata={
                "authors": r.authors,
                "categories": r.categories,
                "published_at": r.published_at.isoformat() if r.published_at else None,
                "source": r.source,
            },
        )
