from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from agent.config import AppConfig
from agent.llm.base import BaseLLM
from agent.logger import log_info


class RemoteLLM(BaseLLM):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.llm.remote.base_url,
            api_key=config.api_keys.openai_api_key or None,
        )

    async def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        await log_info("llm", f"Request: chat.completions | model={self.config.llm.remote.model} | system={len(system)} chars | user={len(prompt)} chars",
                       data={"system_prompt": system, "user_prompt": prompt})

        response = await self._client.chat.completions.create(
            model=self.config.llm.remote.model,
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        content = response.choices[0].message.content or ""

        await log_info("llm", f"Response ({len(content)} chars):\n{content}",
                       data={"response": content})

        return content.strip()

    async def stream(self, prompt: str, system: str = "", **kwargs: Any) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        await log_info("llm", f"Stream request: chat.completions | model={self.config.llm.remote.model} | system={len(system)} chars | user={len(prompt)} chars",
                       data={"system_prompt": system, "user_prompt": prompt})

        stream = await self._client.chat.completions.create(
            model=self.config.llm.remote.model,
            messages=messages,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            stream=True,
        )

        full_content: list[str] = []
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_content.append(token)
                yield token

        total = "".join(full_content)
        await log_info("llm", f"Stream complete ({len(total)} chars):\n{total}",
                       data={"response": total})
