import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.cache import AsyncCache
from agent.config import AppConfig
from agent.embeddings import create_embeddings
from agent.extractors import ExtractorChain
from agent.extractors.jina import JinaExtractor
from agent.extractors.readability_ext import ReadabilityExtractor
from agent.extractors.trafilatura_ext import TrafilaturaExtractor
from agent.llm import create_llm
from agent.logger import fmt_ms, log_error, log_info, log_success
from agent.models.query import RouterDecision, UserQuery
from agent.models.report import PipelineStats, ResearchReport
from agent.models.result import RawResult
from agent.reranker import create_reranker
from agent.retrievers import (
    ArxivRetriever,
    BaseRetriever,
    DuckDuckGoRetriever,
    GitHubSearchRetriever,
    GitHubTrendingRetriever,
    HackerNewsRetriever,
    RedditRetriever,
    RSSRetriever,
    SemanticScholarRetriever,
    WikipediaRetriever,
)
from agent.router import QueryRouter
from agent.synthesis import Synthesizer


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.router = QueryRouter(config)
        self.cache = AsyncCache(config)
        self.extractors = self._build_extractors()
        self.embeddings = create_embeddings(config)
        self.reranker = create_reranker(config)
        self.llm = create_llm(config)
        self.synthesizer = Synthesizer(self.llm, config)
        self.retrievers: list[BaseRetriever] = self._build_retrievers()

    def _build_extractors(self) -> ExtractorChain:
        extractors = [
            TrafilaturaExtractor(),
            JinaExtractor(api_key=self.config.api_keys.jina_api_key),
            ReadabilityExtractor(),
        ]
        return ExtractorChain(extractors)

    def _build_retrievers(self) -> list[BaseRetriever]:
        registry: dict[str, type] = {
            "arxiv": ArxivRetriever,
            "semantic_scholar": SemanticScholarRetriever,
            "wikipedia": WikipediaRetriever,
            "hackernews": HackerNewsRetriever,
            "rss": RSSRetriever,
            "duckduckgo": DuckDuckGoRetriever,
            "github_search": GitHubSearchRetriever,
            "github_trending": GitHubTrendingRetriever,
            "reddit": RedditRetriever,
        }
        retrievers: list[BaseRetriever] = []
        for name in self.config.retrievers.enabled:
            cls = registry.get(name)
            if cls is not None:
                if name in ("rss", "reddit"):
                    retrievers.append(cls(self.config))
                else:
                    retrievers.append(cls())
        return retrievers

    SOFTWARE_KEYWORDS = {
        "software", "code", "library", "framework", "api", "sdk", "tool",
        "cli", "implementation", "implement", "build", "docker", "kubernetes",
        "package", "repository", "repo", "npm", "pip", "cargo", "pypi",
        "programming", "language", "compiler", "debugger", "plugin",
        "extension", "middleware", "backend", "frontend", "database", "orm",
        "rest", "graphql", "websocket", "deploy", "devops", "ci/cd",
    }

    SOFTWARE_PRIORITY_SOURCES = {"github_search", "github_trending", "hackernews", "reddit"}

    @staticmethod
    def _is_software_query(query: str) -> bool:
        q_lower = query.lower()
        return any(kw in q_lower for kw in Pipeline.SOFTWARE_KEYWORDS)

    def _active_retrievers(self, decision: RouterDecision) -> list[BaseRetriever]:
        active = [r for r in self.retrievers if decision.mode in r.supports_modes]
        if decision.mode == "general" and decision.time_window in ("day", "week", "month"):
            active = [r for r in active if r.name != "wikipedia"]
        if self._is_software_query(decision.query):
            for r in self.retrievers:
                if r.name in self.SOFTWARE_PRIORITY_SOURCES and r not in active:
                    active.insert(0, r)
        return active

    async def run(self, query_str: str) -> ResearchReport:
        t0 = time.perf_counter()
        await log_info("pipeline", f'Starting pipeline for: "{query_str}"')
        query = UserQuery(raw=query_str.strip())

        await log_info("pipeline", "Router stage starting")
        decision = await self.router.classify(query.raw)
        await log_info(
            "pipeline",
            f"Mode: {decision.mode} | Sub-queries ({len(decision.sub_queries)}): "
            f"{' | '.join(decision.sub_queries)}",
        )

        cached = await self.cache.get_report(query.raw)
        if cached:
            await log_success("pipeline", "Report served from cache")
            return cached

        active = self._active_retrievers(decision)
        await log_info(
            "pipeline",
            f"Retrieval starting — {len(active)} active, tw={decision.time_window}",
        )

        retriever_latencies: dict[str, float] = {}
        retriever_tasks = []
        for r in active:
            retriever_tasks.append(
                self._fetch_with_timing(
                    r, decision.sub_queries, retriever_latencies, decision.time_window
                )
            )

        gather_timeout = self.config.timeouts.retriever_s + 5
        try:
            raw_nested = await asyncio.wait_for(
                asyncio.gather(*retriever_tasks, return_exceptions=True),
                timeout=gather_timeout,
            )
        except TimeoutError:
            await log_error("pipeline", f"Retrieval phase timed out after {gather_timeout}s")
            raw_nested = []
        raw_results = self._flatten_deduplicate(raw_nested)
        raw_results = await self._filter_by_date(raw_results, decision.mode)
        await log_info("pipeline", f"Retrieval done: {len(raw_results)} unique results")
        for r in raw_results[:5]:
            await log_info(
                "pipeline",
                f"  Result: \"{r.title[:80]}\" | {r.source} | date={r.published_at}",
            )

        cache_hits, misses = await self.cache.partition(raw_results)
        await log_info(
            "pipeline",
            f"Extraction starting — {len(misses)} URLs to fetch, {len(cache_hits)} cached",
        )

        sem = asyncio.Semaphore(self.config.extractors.extraction_concurrency)
        t_extract = time.perf_counter()
        fresh_chunks = await self.extractors.extract_all(misses, sem)
        extraction_latency = (time.perf_counter() - t_extract)

        for chunk in fresh_chunks:
            await self.cache.set_chunk(chunk.url, chunk)

        all_chunks = cache_hits + fresh_chunks
        await log_info(
            "pipeline",
            f"Extraction done: {len(all_chunks)} chunks ({len(cache_hits)} cached)",
        )

        t_rerank = time.perf_counter()
        scored = await self.reranker.rank(all_chunks, query.raw, self.config.reranker.top_k)
        rerank_latency = (time.perf_counter() - t_rerank)

        report = await self.synthesizer.run(query, decision, scored)

        report.stats = PipelineStats(
            total_results_fetched=len(raw_results),
            total_chunks_extracted=len(all_chunks),
            cache_hits=len(cache_hits),
            chunks_after_rerank=len(scored),
            retriever_latencies=retriever_latencies,
            extraction_latency=round(extraction_latency, 2),
            rerank_latency=round(rerank_latency, 2),
            synthesis_latency=(
                report.stats.synthesis_latency
                if hasattr(report.stats, "synthesis_latency") else 0.0
            ),
            total_latency=round((time.perf_counter() - t0), 2),
            llm_tokens_used=(
                report.stats.llm_tokens_used
                if hasattr(report.stats, "llm_tokens_used") else 0
            ),
            llm_backend=self.config.llm.mode,
        )

        await self.cache.set_report(query.raw, report)

        path = report.save(Path(self.config.output.reports_dir))
        await log_success("pipeline", f"Done — report saved → {path}")

        return report

    async def _fetch_with_timing(
        self,
        retriever: BaseRetriever,
        sub_queries: list[str],
        latencies: dict[str, float],
        time_window: str = "all",
    ) -> list[RawResult]:
        t0 = time.perf_counter()
        timeout = self.config.timeouts.retriever_s
        try:
            results = await asyncio.wait_for(
                retriever.fetch(
                    sub_queries,
                    self.config.retrievers.max_results_per_source,
                    time_window=time_window,
                ),
                timeout=timeout,
            )
            latency = (time.perf_counter() - t0) * 1000
            latencies[retriever.name] = latency
            await log_info(retriever.name, f"Fetched {len(results)} results in {fmt_ms(latency)}")
            return results
        except TimeoutError:
            latency = (time.perf_counter() - t0) * 1000
            latencies[retriever.name] = latency
            await log_error(retriever.name, f"Timeout after {timeout}s ({fmt_ms(latency)})")
            return []
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            latencies[retriever.name] = latency
            await log_error(retriever.name, f"Fetch failed: {e}")
            return []

    async def _filter_by_date(self, results: list[RawResult], mode: str) -> list[RawResult]:
        current_year = datetime.now().year
        filtered = []
        dropped = 0
        for r in results:
            if r.published_at is not None and r.published_at.year < current_year:
                dropped += 1
                continue
            filtered.append(r)
        if dropped:
            await log_info(
                "pipeline",
                f"Filtered {dropped} results from before {current_year} (mode: {mode})",
            )
        return filtered

    def _flatten_deduplicate(self, nested: list[Any]) -> list[RawResult]:
        seen: set[str] = set()
        results: list[RawResult] = []

        for item in nested:
            if isinstance(item, Exception):
                continue
            if isinstance(item, list):
                for r in item:
                    if isinstance(r, RawResult) and r.id not in seen:
                        seen.add(r.id)
                        results.append(r)

        return results
