from abc import ABC, abstractmethod

from agent.models.result import ExtractedChunk, ScoredChunk


class BaseReranker(ABC):
    @abstractmethod
    async def rank(
        self,
        chunks: list[ExtractedChunk],
        query: str,
        top_k: int,
    ) -> list[ScoredChunk]:
        ...
