import httpx

from agent.embeddings.base import BaseEmbedder


class ServerEmbedder(BaseEmbedder):
    def __init__(self, base_url: str, model_name: str) -> None:
        self._base_url = base_url
        self._model_name = model_name
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model_name, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model_name, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
