import re
import time
from datetime import datetime
from typing import Any

import tiktoken

from agent.config import AppConfig
from agent.llm.base import BaseLLM
from agent.logger import fmt_ms, log_debug, log_info, log_success
from agent.models.query import RouterDecision, UserQuery
from agent.models.report import Citation, PipelineStats, ResearchReport
from agent.models.result import ScoredChunk
from agent.synthesis.prompts import build_synthesis_prompt, build_system_prompt


class Synthesizer:
    def __init__(self, llm: BaseLLM, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    async def run(
        self,
        query: UserQuery,
        decision: RouterDecision,
        chunks: list[ScoredChunk],
    ) -> ResearchReport:
        t0 = time.perf_counter()
        system_prompt = build_system_prompt()
        system_tokens = self.llm.count_tokens(system_prompt)

        overhead_text = (
            f"Research Query: {query.raw}\n"
            f"Mode: {decision.mode}\n"
            f"Sub-queries: {', '.join(decision.sub_queries)}\n\n"
        )
        overhead_tokens = self.llm.count_tokens(overhead_text)

        truncated = await self.llm.truncate_to_context(
            chunks=chunks,
            system_tokens=system_tokens,
            prompt_overhead_tokens=overhead_tokens,
            max_context=self.config.llm.n_ctx,
            response_buffer=self.config.llm.response_buffer,
        )

        user_prompt = build_synthesis_prompt(
            query=query.raw,
            mode=decision.mode,
            sub_queries=decision.sub_queries,
            chunks=truncated,
            max_chunks=len(truncated),
            max_chunk_chars=self.config.synthesis.max_chunk_chars,
        )

        await log_info(
            "llm",
            f"Synthesis starting — {len(truncated)} chunks, {self.config.llm.mode} backend",
        )

        await log_info("llm", "=== SYSTEM PROMPT ===")
        await log_info("llm", system_prompt)
        await log_info("llm", "=== END SYSTEM PROMPT ===")

        await log_info("llm", f"=== USER PROMPT OVERVIEW === query=\"{query.raw}\" mode={decision.mode} sub_queries={decision.sub_queries}")
        await log_info("llm", f"Sources provided ({len(truncated)} chunks):")
        for i, chunk in enumerate(truncated, 1):
            pub_date = chunk.metadata.get("published_at", "unknown")
            source = chunk.metadata.get("source", "unknown")
            preview = chunk.content_markdown[:200].replace("\n", " ")
            await log_info("llm", f"  SOURCE [{i}]: \"{chunk.title[:80]}\" | {source} | date={pub_date}")
            await log_info("llm", f"    URL: {chunk.url}")
            await log_info("llm", f"    Content preview: {preview}...")
        await log_info("llm", "=== END SOURCE BLOCKS ===")

        full_text = ""
        stream_count = 0
        batch: list[str] = []

        async for token in self.llm.stream(user_prompt, system=system_prompt):
            full_text += token
            stream_count += 1
            batch.append(token)
            if len(batch) >= 100:
                preview = full_text.replace("\n", " ")
                await log_debug(
                    "llm",
                    f"LLM response ({stream_count} chunks): {preview}",
                    data={"chunks": stream_count, "partial_response": full_text},
                )
                batch.clear()

        if batch:
            preview = full_text.replace("\n", " ")
            await log_debug(
                "llm",
                f"LLM response final ({stream_count} chunks): {preview}",
                data={"chunks": stream_count, "partial_response": full_text},
            )

        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(full_text))

        citations = self._parse_citations(full_text, truncated)
        synthesis_latency = (time.perf_counter() - t0)

        stats = PipelineStats(
            total_results_fetched=0,
            total_chunks_extracted=len(chunks),
            cache_hits=0,
            chunks_after_rerank=len(chunks),
            extraction_latency=0.0,
            rerank_latency=0.0,
            synthesis_latency=round(synthesis_latency, 2),
            total_latency=round(synthesis_latency, 2),
            llm_tokens_used=token_count,
            llm_backend=self.config.llm.mode,
        )

        report = ResearchReport(
            query=query.raw,
            mode=decision.mode,
            generated_at=datetime.now(),
            markdown=f"**Query:** {query.raw}\n\n{full_text}",
            citations=citations,
            stats=stats,
        )

        await log_success(
            "synthesis",
            f"Report generated — {token_count} tokens, "
            f"{len(citations)} citations in {fmt_ms(synthesis_latency)}",
        )

        return report

    def _parse_citations(self, markdown: str, chunks: list[ScoredChunk]) -> list[Citation]:
        citations: list[Citation] = []
        pattern = r'\[(\d+)\]\s+(.+?)\s+(?:—\s+)?(.+?)\s+(?:—\s+)?(https?://\S+)'

        for match in re.finditer(pattern, markdown):
            idx = int(match.group(1))
            title = match.group(2).strip()
            source = match.group(3).strip()
            url = match.group(4).strip().rstrip(".)")

            citations.append(Citation(
                index=idx,
                title=title,
                source=source,
                url=url,
                authors=[],
                published_at=None,
            ))

        return citations

    def _build_stats(self, **kwargs: Any) -> PipelineStats:
        return PipelineStats(**kwargs)
