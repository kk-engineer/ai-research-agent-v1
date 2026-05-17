import asyncio
from abc import ABC, abstractmethod

from agent.models.result import ExtractedChunk, RawResult


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
                if result is not None:
                    result.extractor_used = extractor.name
                    return result
            except Exception:
                continue
        return None

    async def extract_all(
        self,
        results: list[RawResult],
        semaphore: asyncio.Semaphore,
    ) -> list[ExtractedChunk]:
        async def _extract_one(r: RawResult) -> ExtractedChunk | None:
            async with semaphore:
                return await self.extract(r.url, r.raw_html)

        tasks = [_extract_one(r) for r in results]
        extracted = await asyncio.gather(*tasks, return_exceptions=True)

        chunks: list[ExtractedChunk] = []
        for result in extracted:
            if isinstance(result, ExtractedChunk):
                chunks.append(result)
        return chunks
