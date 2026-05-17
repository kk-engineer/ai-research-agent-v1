# Scope: router/router.py, retrievers/, extractors/
# Style: surgical edits — no rewrites of working code
---
## CONTEXT

Read these files before writing any code:
- `src/agent/router/router.py`
- `src/agent/retrievers/base.py` + all retrievers
- `src/agent/extractors/base.py` + all extractors
- `src/agent/models/query.py` + `result.py`
- `src/agent/config.py`
- `src/agent/pipeline.py`

Do NOT touch: `synthesizer.py`, `reranker/`, `llm/`, `cache/`, `ui/`, `main.py`.

---

## BUGS TO FIX FIRST (before improvements)

### BUG 1 — `_llm_classify()` is a fake LLM call
**File:** `router/router.py`
**Problem:** `_llm_classify()` never calls the LLM. It just repeats keyword matching, making the "fallback" identical to the heuristic. Low-confidence queries (e.g. "top 5 AI news in last 4 weeks" scores confidence=0.20) always land here and get the same shallow classification.

**Fix:** Replace `_llm_classify()` body with a real LLM call using `build_classification_prompt()` from `prompts.py`. Parse the JSON response into `RouterDecision`. Guard with a try/except — on any failure, fall back to the keyword result already computed by `_heuristic_classify()`. Use the existing `self.llm` if available, else keep old keyword fallback.

```python
async def _llm_classify(self, query: str) -> RouterDecision:
    try:
        prompt = build_classification_prompt(query)
        raw = await self.llm.complete(prompt, system="Respond only with valid JSON.")
        # strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
        return RouterDecision(
            query=query,
            mode=data["mode"],
            sub_queries=data.get("sub_queries", [query]),
            academic_weight=data.get("academic_weight", 0.5),
            general_weight=data.get("general_weight", 0.5),
            explanation=data.get("explanation", "llm classified"),
            classified_by="llm",
        )
    except Exception as e:
        await log_warning("router", f"LLM classify failed: {e} — using keyword fallback")
        # return the already-computed heuristic result with low confidence
        return self._keyword_fallback(query)
```

Also update `build_classification_prompt()` in `prompts.py` to request this exact JSON schema:
```json
{
  "mode": "academic|general|hybrid",
  "academic_weight": 0.0-1.0,
  "general_weight": 0.0-1.0,
  "sub_queries": ["...", "..."],
  "explanation": "one sentence"
}
```

---

### BUG 2 — Temporal intent is buried inside DuckDuckGo only
**File:** `retrievers/duckduckgo.py`
**Problem:** `timelimit` detection (day/week/month/year) is hardcoded inside `DuckDuckGoRetriever.fetch()`. Other retrievers (HackerNews, RSS, Reddit) ignore time constraints entirely. A query like "AI news last week" passes `timelimit='m'` to DDG but HackerNews still fetches all-time results.

**Fix — Step A:** Add `time_window` field to `RouterDecision` in `models/query.py`:
```python
time_window: Literal["day", "week", "month", "year", "all"] = "all"
```

**Fix — Step B:** Compute `time_window` inside `_heuristic_classify()` before returning:
```python
TEMPORAL_MAP = {
    "day": ["today", "24 hours", "past day", "last day"],
    "week": ["this week", "past week", "last week", "7 days"],
    "month": ["this month", "past month", "last month", "30 days", "4 weeks"],
    "year": ["this year", "past year", "last year", "12 months"],
}

def _detect_time_window(self, query: str) -> Literal["day","week","month","year","all"]:
    q = query.lower()
    for window, phrases in TEMPORAL_MAP.items():
        if any(p in q for p in phrases):
            return window
    if any(k in q for k in ["latest", "recent", "new", "now"]):
        return "month"
    return "all"
```

**Fix — Step C:** Pass `RouterDecision` (already passed as part of sub_queries context in pipeline) into each retriever's `fetch()` signature. Change base class:
```python
# retrievers/base.py
@abstractmethod
async def fetch(
    self,
    queries: list[str],
    max_results: int,
    time_window: str = "all",   # ADD THIS
) -> list[RawResult]: ...
```

Update all 9 retriever `fetch()` signatures accordingly. Use `time_window` in:
- **DuckDuckGo**: map `"day"→'d'`, `"week"→'w'`, `"month"→'m'`, `"year"→'y'`, `"all"→None`
- **HackerNews**: add `numericFilters=f"created_at_i>{cutoff_unix}"` based on `time_window`
- **Reddit**: add `t=week|month|year` param to `.json` URL based on `time_window`
- **RSS**: filter entries by `published_at` cutoff computed from `time_window`

Update `pipeline.py` call sites to pass `decision.time_window` into each `retriever.fetch()`.

---

### BUG 3 — Sub-query decomposition generates noise, not signal
**File:** `router/router.py`, `_decompose_query()`
**Problem:** The function always appends `"{year}"`, `"recent"`, `"machine learning"`, `"AI research"`, `"deep learning"` regardless of query content. For `"top 5 AI news in last 4 weeks"` this produces `"top 5 AI news in last 4 weeks machine learning"` — a worse query than the original. All retrievers then use these padded queries.

**Fix:** Rewrite `_decompose_query()` with mode-aware, minimal expansion:

```python
def _decompose_query(self, query: str, mode: str, time_window: str) -> list[str]:
    sub = [query]   # always include original verbatim

    q = query.lower()

    # Split compound queries
    if " and " in q:
        parts = [p.strip() for p in query.split(" and ", 1) if len(p.strip()) > 10]
        sub.extend(parts)
    if " vs " in q:
        parts = [p.strip() for p in query.split(" vs ", 1) if len(p.strip()) > 10]
        sub.extend(parts)

    # Mode-specific expansion only
    if mode in ("academic", "hybrid"):
        # add paper-discovery variant
        sub.append(f"{query} paper arxiv")
        sub.append(f"{query} research 2025 2026")

    if mode in ("general", "hybrid") and time_window != "all":
        # add news-discovery variant
        sub.append(f"{query} news announcement")

    # Deduplicate preserving order; limit to 4
    seen = set()
    result = []
    for s in sub:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result[:4]
```

---

## IMPROVEMENTS (after bugs are fixed)

### IMP 1 — Short-circuit extraction for arXiv and Semantic Scholar
**Files:** `extractors/base.py` (ExtractorChain), `pipeline.py`
**Problem:** Every arXiv URL goes through Trafilatura → Jina → Readability, fetching the HTML abstract page. The abstract is already in `RawResult.snippet` from the arXiv retriever. This wastes 1–3 seconds per paper with zero quality gain.

**Fix:** In `ExtractorChain.extract_all()`, before attempting any HTTP extraction, check the source:

```python
async def extract_all(
    self,
    results: list[RawResult],
    semaphore: asyncio.Semaphore,
) -> list[ExtractedChunk]:
    chunks = []
    for r in results:
        # Fast path: use the snippet directly for abstract sources
        if r.source in ("arxiv", "semantic_scholar") and len(r.snippet) >= 80:
            chunk = ExtractedChunk(
                chunk_id=str(uuid4()),
                source_id=r.id,
                url=r.url,
                title=r.title,
                content_markdown=r.snippet,  # abstract is already clean text
                word_count=len(r.snippet.split()),
                extractor_used="snippet",
                extraction_latency_ms=0.0,
                metadata={
                    "authors": r.authors,
                    "categories": r.categories,
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                    "source": r.source,
                },
            )
            chunks.append(chunk)
            await log_debug("extractor", f"snippet fast-path: {r.url[:60]}")
            continue

        # Normal extraction path under semaphore
        async with semaphore:
            chunk = await self._extract_with_chain(r)
            if chunk:
                chunks.append(chunk)

    return chunks
```

This alone saves ~2–5s per arXiv-heavy query (15 papers × 200ms avg = 3s saved).

---

### IMP 2 — Run sub-queries concurrently inside each retriever
**Files:** `retrievers/arxiv.py`, `retrievers/semantic_scholar.py`, `retrievers/hackernews.py`, `retrievers/duckduckgo.py`
**Problem:** All retrievers loop over sub-queries sequentially with `for q in queries: results += await fetch_one(q)`. With 4 sub-queries and 800ms per query, this is 3.2s serial where it could be ~0.8s parallel.

**Fix:** In each retriever's `fetch()`, replace the sequential loop with `asyncio.gather()`:

```python
# Pattern to apply in each retriever
async def fetch(self, queries: list[str], max_results: int, time_window: str = "all") -> list[RawResult]:
    tasks = [self._fetch_one(q, max_results, time_window) for q in queries]
    nested = await asyncio.gather(*tasks, return_exceptions=True)
    return self._deduplicate([r for batch in nested if isinstance(batch, list) for r in batch])
```

For sync SDK-based retrievers (arXiv, SemanticScholar), each `_fetch_one` already wraps in `asyncio.to_thread()` so `gather()` runs them in the thread pool concurrently.

Add `_deduplicate()` helper on `BaseRetriever`:
```python
def _deduplicate(self, results: list[RawResult]) -> list[RawResult]:
    seen: set[str] = set()
    out = []
    for r in results:
        if r.id not in seen:
            seen.add(r.id)
            out.append(r)
    return out
```

---

### IMP 3 — RSS: score-then-filter before returning, not after
**File:** `retrievers/rss.py`
**Problem:** All 11 feeds are always fetched and parsed. For a query like "vision transformer papers", feeds like TechCrunch AI and VentureBeat return noise (startup funding, product launches) that lowers reranker quality and wastes context window.

**Fix:** Add a minimum relevance threshold before returning RSS results:

```python
def _score_entry(self, entry_text: str, query_terms: set[str]) -> float:
    text = entry_text.lower()
    hits = sum(1 for t in query_terms if t in text)
    return hits / max(len(query_terms), 1)

async def fetch(self, queries: list[str], max_results: int, time_window: str = "all") -> list[RawResult]:
    query_terms = set(queries[0].lower().split())  # use original query terms
    MIN_RELEVANCE = 0.15   # at least 15% of query terms must appear

    all_entries = await self._parse_all_feeds()
    scored = [
        (entry, self._score_entry(entry.title + " " + entry.get("summary",""), query_terms))
        for entry in all_entries
    ]
    filtered = [(e, s) for e, s in scored if s >= MIN_RELEVANCE]
    filtered.sort(key=lambda x: x[1], reverse=True)

    return [self._to_raw_result(e) for e, _ in filtered[:max_results]]
```

---

### IMP 4 — Wikipedia: skip for pure news/general queries
**File:** `retrievers/wikipedia.py` and `pipeline.py`
**Problem:** Wikipedia is fetched for every query including `"top 5 AI news last week"`. It returns definitional content (what is a neural network) that has zero freshness score, lowering reranker throughput.

**Fix:** In `pipeline.py`, pass a `skip_wikipedia` hint when `mode == "general"` and `time_window in ("day", "week", "month")`:

```python
# In pipeline.py _active_retrievers()
def _active_retrievers(self, decision: RouterDecision) -> list[BaseRetriever]:
    active = [r for r in self.retrievers if decision.mode in r.supports_modes]

    # Skip Wikipedia for pure-news temporal queries — adds no freshness value
    if decision.mode == "general" and decision.time_window in ("day", "week", "month"):
        active = [r for r in active if r.name != "wikipedia"]

    return active
```

---

### IMP 5 — Increase synthesis content window from 800 to 2000 chars
**File:** `synthesis/prompts.py`, `build_synthesis_prompt()`
**Problem:** Each source block sends only `content_markdown[:800]`. An 800-char snippet is ~120 words — often cuts off mid-sentence and misses key details. The LLM context is not the bottleneck (8192 tokens with 20 chunks × 2000 chars = ~12k chars ≈ 3k tokens for sources, well within budget).

**Fix:** Change the slice in `build_synthesis_prompt()`:
```python
# Change this line:
content_preview = chunk.content_markdown[:800]
# To:
content_preview = chunk.content_markdown[:2000]
```

Also add a separator line so the LLM doesn't conflate adjacent sources:
```python
source_block = (
    f"=== SOURCE [{i+1}] ===\n"
    f"Title: {chunk.title}\n"
    f"URL: {chunk.url}\n"
    f"Source: {chunk.source if hasattr(chunk, 'source') else chunk.metadata.get('source','unknown')}\n"
    f"Date: {date_str}\n"
    f"---\n"
    f"{content_preview}\n"
    f"{'='*40}\n\n"   # ADD: clear visual separator
)
```

---

### IMP 6 — Add minimum content quality gate in ExtractorChain
**File:** `extractors/base.py`
**Problem:** Extractors can return very short chunks (e.g. paywalled sites return only a headline, 10–20 words). These get reranked and waste context window space in synthesis.

**Fix:** After each extraction attempt, apply a minimum word count check before accepting the result:

```python
MIN_WORDS = 60  # below this, content is likely a headline/paywall stub

# In ExtractorChain._extract_with_chain():
for extractor in self.chain:
    chunk = await extractor.extract(url, raw_html)
    if chunk and chunk.word_count >= MIN_WORDS:
        return chunk
    elif chunk:
        await log_debug(
            "extractor",
            f"Rejected {extractor.name} result for {url[:50]} — only {chunk.word_count} words"
        )

# All extractors failed or returned stubs
await log_warning("extractor", f"No usable content for {url[:70]}")
return None
```

---

## IMPLEMENTATION ORDER

Apply changes in this exact order to avoid breaking imports:

1. `models/query.py` — add `time_window` field to `RouterDecision`
2. `router/router.py` — fix `_llm_classify()`, add `_detect_time_window()`, rewrite `_decompose_query()`
3. `retrievers/base.py` — add `time_window` param to `BaseRetriever.fetch()` ABC
4. All `retrievers/*.py` — add `time_window` param, implement concurrent sub-query gather, use time_window
5. `pipeline.py` — pass `decision.time_window` to retrievers, update `_active_retrievers()` for Wikipedia skip
6. `extractors/base.py` — add snippet fast-path in `extract_all()`, add MIN_WORDS gate
7. `synthesis/prompts.py` — increase content slice to 2000 chars, add separator

Run after each file group:
```bash
uv run ruff check src/agent/ --fix
uv run mypy src/agent/ --ignore-missing-imports
```

---

## RULES FOR THIS TASK

- Edit only the files listed per improvement. Do not touch other files.
- Keep all existing `@with_retry` and circuit breaker decorators — do not remove them.
- All new async functions use `await asyncio.sleep(0)` not `time.sleep()`.
- All log calls use the existing `log_info / log_warning / log_debug` helpers from `logger.py`.
- Type-annotate all new function parameters and return types.
- After each file, show the complete modified file — not a diff.