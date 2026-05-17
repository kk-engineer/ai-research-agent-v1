from abc import ABC, abstractmethod

from agent.config import AppConfig


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


def create_embeddings(config: AppConfig) -> BaseEmbedder:
    if config.embeddings.mode == "cloud":
        from agent.embeddings.cloud import CloudEmbedder

        return CloudEmbedder(config)

    from agent.embeddings.server import ServerEmbedder

    return ServerEmbedder(config.embeddings.base_url, config.embeddings.model)
