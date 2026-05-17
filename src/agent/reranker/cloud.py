from __future__ import annotations

import math
import os
import time
from typing import Any, cast

import httpx

from agent.config import AppConfig, RerankerProviderConfig
from agent.logger import fmt_ms, log_error, log_info, log_warning
from agent.models.result import ExtractedChunk, ScoredChunk
from agent.reranker.base import BaseReranker
from agent.reranker.scorer import authority_score, freshness_score, length_score

PROVIDER_API_KEY_MAP: dict[str, str] = {
    "nvidia": "nvidia_api_key",
    "huggingface": "hf_api_key",
}


class CloudReranker(BaseReranker):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._active_provider: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._provider_config: RerankerProviderConfig | None = None

    def _get_provider_config(self, name: str) -> RerankerProviderConfig:
        return cast(RerankerProviderConfig, getattr(self.config.reranker.cloud, name))

    def _build_client(self, provider_cfg: RerankerProviderConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=provider_cfg.base_url,
            timeout=self.config.reranker.cloud.timeout,
        )

    async def _init_provider(self) -> None:
        if self._client is not None:
            return

        global_api_keys = self.config.api_keys.model_dump()
        last_error: Exception | None = None

        for name in self.config.reranker.cloud.provider_order:
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
                    json={
                        "query": "ping",
                        "documents": ["connectivity verification"],
                        "model": provider_cfg.model,
                    },
                    headers={"Authorization": f"Bearer {provider_cfg.api_key}"},
                )
                response.raise_for_status()
                self._client = client
                self._provider_config = provider_cfg
                self._active_provider = name
                await log_info(
                    "reranker",
                    f"Cloud reranker active: {name} ({provider_cfg.model})",
                )
                return
            except Exception as e:
                await log_warning("reranker", f"Cloud reranker {name} unavailable: {e}")
                last_error = e
                continue

        msg = "No cloud reranker provider available"
        if last_error:
            msg += f" — last error: {last_error}"
        await log_error("reranker", msg)
        raise RuntimeError(msg)

    async def rank(
        self,
        chunks: list[ExtractedChunk],
        query: str,
        top_k: int,
    ) -> list[ScoredChunk]:
        await self._init_provider()
        assert self._client is not None
        assert self._provider_config is not None
        t0 = time.perf_counter()
        if not chunks:
            return []

        documents = [c.content_markdown[:512] for c in chunks]
        weights = self.config.reranker.weights

        doc_titles = " | ".join(
            f"[{i}] {c.title[:60]}" for i, c in enumerate(chunks[:10])
        )
        if len(chunks) > 10:
            doc_titles += f" | ... and {len(chunks) - 10} more"
        await log_info("reranker", f"Input: query=\"{query}\" | {len(chunks)} docs | top_k={top_k}")
        await log_info("reranker", f"Input documents: {doc_titles}")

        payload: dict[str, Any] = {
            "query": query, "documents": documents, "model": self._provider_config.model,
        }
        response = await self._client.post(
            "",
            json=payload,
            headers={"Authorization": f"Bearer {self._provider_config.api_key}"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        raw_scores: dict[int, float] = {r["index"]: r["relevance_score"] for r in data["results"]}

        score_summary = ", ".join(
            f"[{i}] {raw_scores.get(i, 0.0):.4f}" for i in range(min(5, len(chunks)))
        )
        await log_info("reranker", f"Server scores (top 5): {score_summary}")

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(chunks):
            semantic = _sigmoid(raw_scores.get(i, 0.0))
            freshness = freshness_score(chunk.metadata.get("published_at"))
            authority = authority_score(chunk.metadata.get("source", ""), chunk.url)
            length = length_score(chunk.word_count)

            final = (
                semantic * weights.semantic
                + freshness * weights.freshness
                + authority * weights.authority
                + length * weights.length
            )

            scored.append(ScoredChunk(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                url=chunk.url,
                title=chunk.title,
                content_markdown=chunk.content_markdown,
                word_count=chunk.word_count,
                extractor_used=chunk.extractor_used,
                extraction_latency_ms=chunk.extraction_latency_ms,
                metadata=chunk.metadata,
                semantic_score=round(semantic, 4),
                freshness_score=round(freshness, 4),
                authority_score=round(authority, 4),
                length_score=round(length, 4),
                final_score=round(final, 4),
                rank=0,
            ))

        scored.sort(key=lambda c: c.final_score, reverse=True)
        for i, c in enumerate(scored[:top_k]):
            c.rank = i + 1

        top = scored[:top_k]
        latency = (time.perf_counter() - t0) * 1000

        top_details = "\n".join(
            f"  #{c.rank} | {c.final_score:.4f} | sem:{c.semantic_score:.4f} "
            f"fresh:{c.freshness_score:.4f} auth:{c.authority_score:.4f} "
            f"len:{c.length_score:.4f} | {c.title[:60]}"
            for c in top[:10]
        )
        await log_info("reranker", f"Output: {len(top)} chunks ranked | {fmt_ms(latency)}")
        await log_info(
            "reranker",
            f"Top {min(len(top), 10)}:\n{top_details}",
        )

        return top
