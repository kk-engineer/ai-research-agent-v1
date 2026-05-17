from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import httpx

from agent.config import LLMConfig
from agent.llm.base import BaseLLM
from agent.logger import log_info


class LocalLLM(BaseLLM):
    name: str = "llama_cpp_server"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._base_url = config.base_url.rstrip("/")
        self._model_name = config.model
        self._http_client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(120.0, connect=15.0),
                headers={"Content-Type": "application/json"},
            )
        return self._http_client

    def _build_messages(self, prompt: str, system: str = "") -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def complete(self, prompt: str, **kwargs: str) -> str:
        system = kwargs.get("system", "")
        body = {
            "model": self._model_name,
            "messages": self._build_messages(prompt, system),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if "stop" in kwargs:
            body["stop"] = kwargs["stop"]

        await log_info("llm", f"Request: POST /chat/completions | model={self._model_name} | system={len(system)} chars | user={len(prompt)} chars",
                       data={"system_prompt": system, "user_prompt": prompt})

        response = await self.client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        content = cast(str, data["choices"][0]["message"]["content"])
        result = content.strip()

        await log_info("llm", f"Response ({len(result)} chars):\n{result}",
                       data={"response": result})

        return result

    async def stream(self, prompt: str, **kwargs: str) -> AsyncIterator[str]:
        system = kwargs.get("system", "")
        body = {
            "model": self._model_name,
            "messages": self._build_messages(prompt, system),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if "stop" in kwargs:
            body["stop"] = kwargs["stop"]

        await log_info("llm", f"Stream request: POST /chat/completions | model={self._model_name} | system={len(system)} chars | user={len(prompt)} chars",
                       data={"system_prompt": system, "user_prompt": prompt})

        full_content: list[str] = []
        async with self.client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    if payload:
                        import json

                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content.append(content)
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        total = "".join(full_content)
        await log_info("llm", f"Stream complete ({len(total)} chars):\n{total}",
                       data={"response": total})
