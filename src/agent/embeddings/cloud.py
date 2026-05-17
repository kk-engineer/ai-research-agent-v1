from __future__ import annotations

import os
from typing import Any, cast

import httpx

from agent.config import AppConfig, EmbeddingProviderConfig
from agent.embeddings.base import BaseEmbedder
from agent.logger import log_error, log_info, log_warning

PROVIDER_API_KEY_MAP: dict[str, str] = {
    "openai": "openai_api_key",
    "nvidia": "nvidia_api_key",
    "huggingface": "hf_api_key",
    "gemini": "gemini_api_key",
    "openrouter": "openrouter_api_key",
}


class CloudEmbedder(BaseEmbedder):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._active_provider: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._provider_config: EmbeddingProviderConfig | None = None

    def _get_provider_config(self, name: str) -> EmbeddingProviderConfig:
        return cast(EmbeddingProviderConfig, getattr(self.config.embeddings.cloud, name))

    def _build_client(self, provider_cfg: EmbeddingProviderConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=provider_cfg.base_url,
            timeout=self.config.embeddings.cloud.timeout,
        )

    async def _init_provider(self) -> None:
        if self._client is not None:
            return

        global_api_keys = self.config.api_keys.model_dump()
        last_error: Exception | None = None

        for name in self.config.embeddings.cloud.provider_order:
            provider_cfg = self._get_provider_config(name)
            if not provider_cfg.base_url:
                continue

            if not provider_cfg.api_key:
                key_field = PROVIDER_API_KEY_MAP.get(name, f"{name}_api_key")
                provider_cfg.api_key = (
                    global_api_keys.get(key_field)
                    or os.environ.get(f"{name.lower()}_api_key", "")
                )

            if not provider_cfg.api_key:
                continue

            try:
                client = self._build_client(provider_cfg)
                response = await client.post(
                    "",
                    json={"model": provider_cfg.model, "input": "ping"},
                    headers={"Authorization": f"Bearer {provider_cfg.api_key}"},
                )
                response.raise_for_status()
                self._client = client
                self._provider_config = provider_cfg
                self._active_provider = name
                await log_info(
                    "embeddings",
                    f"Cloud embedder active: {name} ({provider_cfg.model})",
                )
                return
            except Exception as e:
                await log_warning("embeddings", f"Cloud embedder {name} unavailable: {e}")
                last_error = e
                continue

        msg = "No cloud embedding provider available"
        if last_error:
            msg += f" — last error: {last_error}"
        await log_error("embeddings", msg)
        raise RuntimeError(msg)

    async def embed(self, text: str) -> list[float]:
        await self._init_provider()
        assert self._client is not None
        assert self._provider_config is not None
        response = await self._client.post(
            "",
            json={"model": self._provider_config.model, "input": text},
            headers={"Authorization": f"Bearer {self._provider_config.api_key}"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return cast(list[float], data["data"][0]["embedding"])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        await self._init_provider()
        assert self._client is not None
        assert self._provider_config is not None
        response = await self._client.post(
            "",
            json={"model": self._provider_config.model, "input": texts},
            headers={"Authorization": f"Bearer {self._provider_config.api_key}"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [cast(list[float], item["embedding"]) for item in data["data"]]
