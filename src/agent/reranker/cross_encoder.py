import asyncio
import math
import time
from typing import Any

from agent.config import AppConfig
from agent.models.result import ExtractedChunk, ScoredChunk
from agent.reranker.base import BaseReranker
from agent.reranker.scorer import authority_score, freshness_score, length_score
from agent.logger import fmt_ms, log_info


class CrossEncoderReranker(BaseReranker):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._device = "mps"

    async def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        def _load() -> Any:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self._device,
            )
            return model

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, _load)
        return self._model

    async def rank(
        self,
        chunks: list[ExtractedChunk],
        query: str,
        top_k: int,
    ) -> list[ScoredChunk]:
        t0 = time.perf_counter()
        if not chunks:
            return []

        await log_info("reranker", f"Input: query=\"{query}\" | {len(chunks)} documents | top_k={top_k}")

        model = await self._load_model()
        weights = self.config.reranker.weights

        pairs = [(query, c.content_markdown[:512]) for c in chunks]

        def _predict() -> list[float]:
            scores = model.predict(pairs)
            return [float(s) for s in scores]

        loop = asyncio.get_event_loop()
        raw_scores = await loop.run_in_executor(None, _predict)

        score_summary = ", ".join(f"[{i}] {raw_scores[i]:.4f}" for i in range(min(5, len(chunks))))
        await log_info("reranker", f"Model scores (top 5): {score_summary}")

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(chunks):
            semantic = _sigmoid(raw_scores[i])
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

        top_summary = ", ".join(
            f"#{c.rank} {c.title[:40]} ({c.final_score:.4f})" for c in top[:5]
        )
        await log_info("reranker", f"Output: {len(top)} chunks ranked | top: {top_summary} | {fmt_ms(latency)}")

        return top
