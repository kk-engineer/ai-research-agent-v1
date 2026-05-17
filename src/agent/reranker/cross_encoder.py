import asyncio
import math
import re
import time
from typing import Any

from agent.config import AppConfig
from agent.logger import fmt_ms, log_info
from agent.models.result import ExtractedChunk, ScoredChunk
from agent.reranker.base import BaseReranker
from agent.reranker.scorer import authority_score, freshness_score, length_score

GGUF_MODEL_MAP: dict[str, str] = {
    "jina-reranker-v2-base-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
}

QUANT_SUFFIX_RE = re.compile(
    r"-(Q[2-8](?:_[KLM0-9]+(?:_[A-Z])?)?|fp(?:16|32)|bf16|GGUF)$", re.IGNORECASE
)


def _resolve_model_name(name: str) -> str:
    if not name.endswith(".gguf"):
        return name
    base = name.replace(".gguf", "")
    base = QUANT_SUFFIX_RE.sub("", base)
    mapped = GGUF_MODEL_MAP.get(base)
    if mapped:
        return mapped
    if not name.startswith("cross-encoder/"):
        return f"jinaai/{base}"
    return base


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

            raw_name = self.config.reranker.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            model_name = _resolve_model_name(raw_name)
            model = CrossEncoder(
                model_name,
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

        doc_titles = " | ".join(
            f"[{i}] {c.title[:60]}" for i, c in enumerate(chunks[:10])
        )
        if len(chunks) > 10:
            doc_titles += f" | ... and {len(chunks) - 10} more"
        await log_info("reranker", f"Input: query=\"{query}\" | {len(chunks)} docs | top_k={top_k}")
        await log_info("reranker", f"Input documents: {doc_titles}")

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

        top_details = "\n".join(
            f"  #{c.rank} | {c.final_score:.4f} | sem:{c.semantic_score:.4f} fresh:{c.freshness_score:.4f} "
            f"auth:{c.authority_score:.4f} len:{c.length_score:.4f} | {c.title[:60]}"
            for c in top[:10]
        )
        await log_info("reranker", f"Output: {len(top)} chunks ranked | {fmt_ms(latency)}")
        await log_info(
            "reranker",
            f"Top {min(len(top), 10)}:\n{top_details}",
        )

        return top
