from abc import ABC, abstractmethod

from agent.config import AppConfig
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


def create_reranker(config: AppConfig) -> BaseReranker:
    if config.reranker.mode == "cloud":
        from agent.reranker.cloud import CloudReranker

        return CloudReranker(config)

    if config.reranker.mode == "huggingface":
        from agent.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(config)

    from agent.reranker.server import ServerReranker

    return ServerReranker(config)
