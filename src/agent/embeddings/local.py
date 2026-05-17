import asyncio
from typing import Any

from agent.embeddings.base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self._model = None

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        def _load() -> Any:
            from llama_cpp import Llama

            return Llama(
                model_path=self.model_path,
                embedding=True,
                verbose=False,
            )

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, _load)
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = await self._ensure_model()

        def _run() -> list[float]:
            result = model.create_embedding(text)
            return result["data"][0]["embedding"]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results
