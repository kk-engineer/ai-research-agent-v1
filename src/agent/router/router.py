import json
import re
from typing import Literal, cast

from agent.config import AppConfig
from agent.llm import create_llm
from agent.logger import log_info, log_warning
from agent.models.query import RouterDecision
from agent.synthesis.prompts import build_classification_prompt


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

    TEMPORAL_MAP = {
        "day": ["today", "24 hours", "past day", "last day"],
        "week": ["this week", "past week", "last week", "7 days"],
        "month": ["this month", "past month", "last month", "30 days", "4 weeks"],
        "year": ["this year", "past year", "last year", "12 months"],
    }

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = create_llm(config)

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

    def _detect_time_window(self, query: str) -> str:
        q = query.lower()
        for window, phrases in self.TEMPORAL_MAP.items():
            if any(p in q for p in phrases):
                return window
        if any(k in q for k in ["latest", "recent", "new", "now"]):
            return "month"
        return "all"

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

        mode: Literal["academic", "general", "hybrid"]
        if academic_weight >= 0.7:
            mode = "academic"
        elif general_weight >= 0.7:
            mode = "general"
        else:
            mode = "hybrid"

        time_window_raw = self._detect_time_window(query)
        valid_tw = ("day", "week", "month", "year", "all")
        tw: Literal["day", "week", "month", "year", "all"] = cast(
            Literal["day", "week", "month", "year", "all"],
            time_window_raw if time_window_raw in valid_tw else "all",
        )
        confidence = min(1.0, total / 5.0)

        decision = RouterDecision(
            query=query,
            mode=mode,
            sub_queries=self._decompose_query(query, mode, tw),
            time_window=tw,
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
        try:
            prompt = build_classification_prompt(query)
            raw = await self.llm.complete(prompt, system="Respond only with valid JSON.")
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            time_window = self._detect_time_window(query)
            sub_queries = data.get("sub_queries", [query])
            if not sub_queries or not isinstance(sub_queries, list):
                sub_queries = [query]
            raw_mode = data.get("mode", "hybrid")
            mode: Literal["academic", "general", "hybrid"] = cast(
                Literal["academic", "general", "hybrid"],
                raw_mode if raw_mode in ("academic", "general", "hybrid") else "hybrid",
            )
            valid_tw = ("day", "week", "month", "year", "all")
            tw: Literal["day", "week", "month", "year", "all"] = cast(
                Literal["day", "week", "month", "year", "all"],
                time_window if time_window in valid_tw else "all",
            )
            return RouterDecision(
                query=query,
                mode=mode,
                sub_queries=sub_queries,
                time_window=tw,
                academic_weight=data.get("academic_weight", 0.5),
                general_weight=data.get("general_weight", 0.5),
                explanation=data.get("explanation", "llm classified"),
                classified_by="llm",
            )
        except Exception as e:
            await log_warning("router", f"LLM classify failed: {e} — using keyword fallback")
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> RouterDecision:
        raw_mode: str = "hybrid"
        if any(kw in query.lower() for kw in self.ACADEMIC_KEYWORDS):
            raw_mode = "academic"
        elif any(kw in query.lower() for kw in self.GENERAL_KEYWORDS):
            raw_mode = "general"
        mode: Literal["academic", "general", "hybrid"] = cast(
            Literal["academic", "general", "hybrid"],
            raw_mode,
        )

        time_window = self._detect_time_window(query)
        valid_tw = ("day", "week", "month", "year", "all")
        tw: Literal["day", "week", "month", "year", "all"] = cast(
            Literal["day", "week", "month", "year", "all"],
            time_window if time_window in valid_tw else "all",
        )
        sub_queries = self._decompose_query(query, mode, tw)

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
            time_window=tw,
            academic_weight=aw,
            general_weight=gw,
            explanation="Keyword fallback classification",
            classified_by="llm",
        )

    def _decompose_query(self, query: str, mode: str, time_window: str) -> list[str]:
        sub = [query]

        q = query.lower()

        if " and " in q:
            parts = [p.strip() for p in query.split(" and ", 1) if len(p.strip()) > 10]
            sub.extend(parts)
        if " vs " in q:
            parts = [p.strip() for p in query.split(" vs ", 1) if len(p.strip()) > 10]
            sub.extend(parts)

        if mode in ("academic", "hybrid"):
            sub.append(f"{query} paper arxiv")
            sub.append(f"{query} research 2025 2026")

        if mode in ("general", "hybrid") and time_window != "all":
            sub.append(f"{query} news announcement")

        seen = set()
        result = []
        for s in sub:
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result[:4]
