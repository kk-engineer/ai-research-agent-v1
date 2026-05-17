import asyncio
from typing import Any

from agent.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        def _load() -> Any:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self.model_name, device="mps")

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, _load)
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = await self._ensure_model()

        def _run() -> list[float]:
            emb = model.encode(text)
            return emb.tolist()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = await self._ensure_model()

        def _run() -> list[list[float]]:
            embs = model.encode(texts)
            return [e.tolist() for e in embs]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)
