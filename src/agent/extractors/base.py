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


class ExtractorChain:
    def __init__(self, extractors: list[BaseExtractor]) -> None:
        self.extractors = extractors

    async def extract(self, url: str, raw_html: str | None = None) -> ExtractedChunk | None:
        for extractor in self.extractors:
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
                chunk = ExtractedChunk(
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
                chunks.append(chunk)
                await log_debug("extractor", f"snippet fast-path: {r.url[:60]}")
                continue

            async with semaphore:
                extracted = await self._extract_with_chain(r)
                if extracted:
                    chunks.append(extracted)

        return chunks

    async def _extract_with_chain(self, r: RawResult) -> ExtractedChunk | None:
        for extractor in self.extractors:
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
