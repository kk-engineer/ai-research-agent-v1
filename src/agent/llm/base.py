from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import tiktoken

from agent.config import AppConfig
from agent.models.result import ScoredChunk
from agent.logger import log_info


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        ...

    @abstractmethod
    async def stream(self, prompt: str, system: str = "", **kwargs: Any) -> AsyncIterator[str]:
        ...

    def count_tokens(self, text: str) -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text.split())

    async def truncate_to_context(
        self,
        chunks: list[ScoredChunk],
        system_tokens: int,
        prompt_overhead_tokens: int,
        max_context: int,
        response_buffer: int,
    ) -> list[ScoredChunk]:
        available = max_context - system_tokens - prompt_overhead_tokens - response_buffer
        if available <= 0:
            return []

        result: list[ScoredChunk] = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk.content_markdown)
            if total_tokens + chunk_tokens > available:
                remaining = len(chunks) - len(result)
                if remaining > 0:
                    await log_info(
                        "llm",
                        f"Truncated {remaining} chunks to fit context window ({available} tokens)",
                    )
                break
            result.append(chunk)
            total_tokens += chunk_tokens

        return result


def create_llm(config: AppConfig) -> BaseLLM:
    from agent.llm.local import LocalLLM
    from agent.llm.remote import RemoteLLM

    if config.llm.backend == "remote":
        return RemoteLLM(config)
    return LocalLLM(config.llm)
