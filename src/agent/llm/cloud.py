from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from agent.config import AppConfig, LLMProviderConfig
from agent.llm.base import BaseLLM
from agent.logger import log_error, log_info, log_warning

PROVIDER_API_KEY_MAP: dict[str, str] = {
    "nvidia": "nvidia_api_key",
    "gemini": "gemini_api_key",
    "openrouter": "openrouter_api_key",
    "huggingface": "hf_api_key",
    "deepseek": "deepseek_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


class CloudLLM(BaseLLM):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._active_provider: str | None = None
        self._client: AsyncOpenAI | None = None
        self._provider_config: LLMProviderConfig | None = None

    def _get_provider_config(self, name: str) -> LLMProviderConfig:
        return getattr(self.config.llm.cloud, name)

    def _build_client(self, provider_cfg: LLMProviderConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=provider_cfg.base_url.rstrip("/") + "/",
            api_key=provider_cfg.api_key or None,
            timeout=self.config.llm.cloud.timeout,
        )

    async def _init_provider(self) -> None:
        if self._client is not None:
            return

        global_api_keys = self.config.api_keys.model_dump()
        last_error: Exception | None = None

        for name in self.config.llm.cloud.provider_order:
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
                await client.chat.completions.create(
                    model=provider_cfg.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                self._client = client
                self._provider_config = provider_cfg
                self._active_provider = name
                await log_info("llm", f"Cloud LLM active: {name} ({provider_cfg.model})")
                return
            except Exception as e:
                await log_warning("llm", f"Cloud LLM {name} unavailable: {e}")
                last_error = e
                continue

        msg = "No cloud LLM provider available"
        if last_error:
            msg += f" — last error: {last_error}"
        await log_error("llm", msg)
        raise RuntimeError(msg)

    async def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        await self._init_provider()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = self._provider_config.model
        await log_info(
            "llm",
            f"Cloud request: chat.completions | provider={self._active_provider} | "
            f"model={model} | system={len(system)} chars | user={len(prompt)} chars",
            data={"system_prompt": system, "user_prompt": prompt},
        )

        response = await self._client.chat.completions.create(
            model=self._provider_config.model,
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        if not response.choices:
            raise RuntimeError(f"Empty response from {self._active_provider}")
        content = response.choices[0].message.content or ""

        c_len = len(content)
        await log_info(
            "llm", f"Response ({c_len} chars):\n{content}", data={"response": content}
        )
        return content.strip()

    async def stream(self, prompt: str, system: str = "", **kwargs: Any) -> AsyncIterator[str]:
        await self._init_provider()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = self._provider_config.model
        await log_info(
            "llm",
            f"Cloud stream request: chat.completions | provider={self._active_provider} | "
            f"model={model} | system={len(system)} chars | user={len(prompt)} chars",
            data={"system_prompt": system, "user_prompt": prompt},
        )

        stream = await self._client.chat.completions.create(
            model=self._provider_config.model,
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            stream=True,
        )

        full_content: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue
            token = delta.content
            full_content.append(token)
            yield token

        total = "".join(full_content)
        t_len = len(total)
        await log_info(
            "llm", f"Stream complete ({t_len} chars):\n{total}", data={"response": total}
        )
