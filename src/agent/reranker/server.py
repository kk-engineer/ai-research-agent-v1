import math
import time

import httpx

from agent.config import AppConfig
from agent.logger import fmt_ms, log_info
from agent.models.result import ExtractedChunk, ScoredChunk
from agent.reranker.base import BaseReranker
from agent.reranker.scorer import authority_score, freshness_score, length_score


class ServerReranker(BaseReranker):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._base_url = config.reranker.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def rank(
        self,
        chunks: list[ExtractedChunk],
        query: str,
        top_k: int,
    ) -> list[ScoredChunk]:
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

        payload = {"query": query, "documents": documents}
        if self.config.reranker.model:
            payload["model"] = self.config.reranker.model

        response = await self._client.post(f"{self._base_url}/rerank", json=payload)
        response.raise_for_status()
        data = response.json()
        raw_scores = {r["index"]: r["relevance_score"] for r in data["results"]}

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
