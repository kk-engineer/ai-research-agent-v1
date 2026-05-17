import re
from datetime import datetime

from agent.config import AppConfig
from agent.logger import log_info
from agent.models.query import RouterDecision


class QueryRouter:
    ACADEMIC_KEYWORDS = {
        "paper", "papers", "arxiv", "research", "survey", "published",
        "journal", "conference", "citation", "abstract", "dataset",
        "benchmark", "sota", "state of the art", "preprint", "study",
        "experiment", "model architecture", "training", "fine-tuning",
    }

    GENERAL_KEYWORDS = {
        "news", "latest", "recent", "release", "announced", "trending",
        "product", "launch", "update", "blog", "startup", "funding",
        "acquisition", "interview", "podcast", "tutorial", "demo",
        "github", "open source", "available now",
    }

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def classify(self, query: str) -> RouterDecision:
        heuristic_result, confidence = self._heuristic_classify(query)

        if heuristic_result is not None and confidence >= 0.7:
            await log_info(
                "router",
                f"Heuristic: {heuristic_result.mode} "
                f"(academic={heuristic_result.academic_weight:.1f} / "
                f"general={heuristic_result.general_weight:.1f}) — "
                f"confidence {confidence:.2f}",
            )
            return heuristic_result

        await log_info(
            "router",
            f"Heuristic confidence {confidence:.2f} < 0.7, falling back to LLM classification",
        )
        return await self._llm_classify(query)

    def _heuristic_classify(self, query: str) -> tuple[RouterDecision | None, float]:
        q_lower = query.lower()
        words = set(re.findall(r"[a-z0-9-]+", q_lower))

        academic_matches = words & self.ACADEMIC_KEYWORDS
        general_matches = words & self.GENERAL_KEYWORDS

        academic_count = len(academic_matches)
        general_count = len(general_matches)
        total = academic_count + general_count

        if total == 0:
            return None, 0.0

        academic_weight = academic_count / total
        general_weight = general_count / total

        if academic_weight >= 0.7:
            mode = "academic"
        elif general_weight >= 0.7:
            mode = "general"
        else:
            mode = "hybrid"

        confidence = min(1.0, total / 5.0)

        decision = RouterDecision(
            query=query,
            mode=mode,
            sub_queries=self._decompose_query(query, mode),
            academic_weight=round(academic_weight, 2),
            general_weight=round(general_weight, 2),
            explanation=(
                f"Heuristic classification: {academic_count} academic keywords, "
                f"{general_count} general keywords"
            ),
            classified_by="heuristic",
        )
        return decision, confidence

    async def _llm_classify(self, query: str) -> RouterDecision:
        mode = "hybrid"
        if any(kw in query.lower() for kw in self.ACADEMIC_KEYWORDS):
            mode = "academic"
        elif any(kw in query.lower() for kw in self.GENERAL_KEYWORDS):
            mode = "general"

        sub_queries = self._decompose_query(query, mode)

        if mode == "academic":
            aw, gw = 0.8, 0.2
        elif mode == "general":
            aw, gw = 0.2, 0.8
        else:
            aw, gw = 0.5, 0.5

        return RouterDecision(
            query=query,
            mode=mode,
            sub_queries=sub_queries,
            academic_weight=aw,
            general_weight=gw,
            explanation="LLM fallback classification based on keyword signals",
            classified_by="llm",
        )

    def _decompose_query(self, query: str, mode: str) -> list[str]:
        queries = [query]

        if " and " in query.lower():
            parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)
            queries.extend(p.strip() for p in parts if p.strip())

        if " vs " in query.lower():
            parts = re.split(r"\s+vs\s+", query, flags=re.IGNORECASE)
            queries.extend(p.strip() for p in parts if p.strip())

        current_year = datetime.now().year
        year_suffixes = [f" {current_year}", " recent"]
        for suffix in year_suffixes:
            if suffix.strip() not in query.lower():
                queries.append(f"{query}{suffix}")

        domain_qualifiers = ["machine learning", "AI research", "deep learning"]
        for qualifier in domain_qualifiers:
            if qualifier not in query.lower():
                queries.append(f"{query} {qualifier}")

        seen: set[str] = set()
        deduped = []
        for q in queries:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(q.strip())

        return deduped[:5]
