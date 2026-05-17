# AI Research Agent — Code Workflow

## 1. Architecture Overview

```
User Query (CLI or Shell)
        │
        ▼
  ┌──────────────────────────────────────────────────────┐
  │                    main.py / cli.py                  │
  │  - Typer CLI (run, shell, config-show, cache-clear)  │
  │  - Interactive shell with /commands                  │
  └──────────────────────┬───────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │                   startup.py                         │
  │  - validate_connectivity()                           │
  │    ├── LLM check (complete with "Respond with 'OK'") │
  │    ├── Embeddings check (server or sentence-trans.)  │
  │    └── Reranker check (local cross-encoder or API)   │
  └──────────────────────┬───────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │                   pipeline.py                        │
  │  Pipeline.run(query_str)                             │
  │                                                      │
  │   1. ROUTE    ───►  QueryRouter.classify()           │
  │   2. CACHE    ───►  AsyncCache.get_report()          │
  │   3. RETRIEVE ───►  6 retrievers in parallel         │
  │   4. EXTRACT  ───►  ExtractorChain.extract_all()     │
  │   5. RERANK   ───►  CrossEncoderReranker / Server    │
  │   6. SYNTHESIZE ─►  Synthesizer.run()                │
  │   7. CACHE    ───►  AsyncCache.set_report()          │
  │   8. SAVE     ───►  ResearchReport.save()            │
  └──────────────────────────────────────────────────────┘
```

---

## 2. Entry Points

### 2.1 `agent run <query>` (main.py)

`typer` CLI with commands:

| Command | File | Description |
|---------|------|-------------|
| `run <query>` | `main.py:43-131` | Single-shot research pipeline |
| `shell` | `main.py:234-250` | Interactive REPL |
| `config-show` | `main.py:133-146` | Print resolved config (keys masked) |
| `cache-clear` | `main.py:149-159` | Purge SQLite cache |
| `cache-stats` | `main.py:162-174` | Show L1/L2 cache entry counts |
| `health` | `main.py:177-231` | Test all retrievers + LLM |

Flow:
1. `load_config()` reads `config.toml` + env var overrides
2. `FileLogWriter.start()` begins draining log queue to `logs/agent_YYYYMMDD.log`
3. `validate_connectivity()` pings LLM, embeddings, reranker
4. `Pipeline(config).run(query)` executes the full workflow
5. Report is displayed as Markdown, statistics table printed
6. `report.save()` writes to `reports/{timestamp}_{slug}.md`
7. `FileLogWriter.stop()` flushes remaining log entries

Signal handling: SIGINT triggers graceful shutdown; second SIGINT forces exit.

### 2.2 Interactive Shell (cli.py)

`InteractiveShell` class runs a REPL loop:

- **`/help`** — Show commands
- **`/clear`** — Clear cache
- **`/stats`** — Cache statistics
- **`/config`** — Show resolved config
- **`/exit`** / `Ctrl+C` — Quit
- **`--no-cache <query>`** — Bypass cache
- **`--export-json <query>`** — Also save as JSON
- **Plain text** — Run research pipeline

---

## 3. Configuration (config.py)

Pydantic `BaseModel` hierarchy loaded from `config.toml`:

```
AppConfig
├── llm: LLMConfig
│   ├── backend: "local" | "remote"
│   ├── base_url (e.g. http://localhost:8000/v1)
│   ├── model (e.g. mistral-nemo-12b)
│   ├── n_ctx (default 8192)
│   ├── temperature (default 0.3)
│   ├── max_tokens (default 2048)
│   ├── response_buffer (default 1024)
│   └── remote: LLMRemoteConfig (base_url + model)
├── embeddings: EmbeddingsConfig
├── retrievers: RetrieverConfig
│   ├── max_results_per_source (default 15)
│   ├── enabled: list[str] (default 6 retrievers)
│   └── rss_feeds: RSSFeedsConfig (7 default feed URLs)
├── extractors: ExtractorConfig (chunk_size_words, concurrency)
├── reranker: RerankerConfig
│   ├── top_k (default 20)
│   ├── backend: "local" | "remote"
│   ├── base_url + model
│   └── weights: semantic=0.55, freshness=0.20, authority=0.15, length=0.10
├── cache: CacheConfig (SQLite DB path + TTLs)
├── timeouts: TimeoutConfig (retriever, extractor, LLM)
├── output: OutputConfig (reports_dir)
├── ui: UIConfig (log_level)
└── api_keys: APIKeyConfig (openai, jina, semantic_scholar)
```

**Env var overrides** (30+ mapped, e.g. `OPENAI_API_KEY`, `LLM_BACKEND`, `LLM_N_CTX`). Environment variables take precedence over TOML file values.

---

## 4. Startup Validation (startup.py)

`validate_connectivity()` runs before any query:

1. **LLM**: Sends `"Respond with exactly 'OK'."` via `llm.complete()`. Exits with code 1 if unreachable.
2. **Embeddings**: Pings server embedder or sentence-transformer. Warning only on failure.
3. **Reranker**: Loads cross-encoder model or pings remote API. Warning only on failure.

---

## 5. The Pipeline (pipeline.py)

`Pipeline.run(query_str)` is the core orchestrator. Detailed stage-by-stage:

### Stage 1: Query Routing

```python
decision = await self.router.classify(query.raw)
```

**File:** `router/router.py`

**`QueryRouter.classify(query)` → `RouterDecision`**

#### 5.1.1 Heuristic Classification

1. Extract lowercase alphanumeric tokens from query
2. Match against `ACADEMIC_KEYWORDS` (14 terms: paper, arxiv, survey, benchmark, sota, etc.) and `GENERAL_KEYWORDS` (16 terms: news, latest, release, trending, etc.)
3. Compute weights: `academic_weight = academic_matches / total`, `general_weight = general_matches / total`
4. Mode assignment: `academic_weight >= 0.7 → "academic"`, `general_weight >= 0.7 → "general"`, else `"hybrid"`
5. Confidence = `min(1.0, total_matches / 5.0)`
6. If confidence >= 0.7, return immediately. Otherwise fall back to...

#### 5.1.2 LLM Classification Fallback

Not actually used — the `_llm_classify()` method does simple keyword matching (same keywords). The LLM classification prompt (`build_classification_prompt` in `prompts.py`) exists but is never invoked because the heuristic handles everything.

#### 5.1.3 Query Decomposition

`_decompose_query(query, mode)` produces up to 5 sub-queries:

1. **Original query**
2. **"And" splits**: If query contains " and ", split into parts
3. **"Vs" splits**: If query contains " vs ", split into parts
4. **Year suffix**: Appends `" {current_year}"` (computed from `datetime.now().year`) — e.g. " 2026"
5. **"Recent" suffix**: Appends `" recent"` if not already in query
6. **Domain qualifiers**: Appends "machine learning", "AI research", "deep learning"
7. Deduplicates, limits to 5

### Stage 2: Cache Check

```python
cached = await self.cache.get_report(query.raw)
```

**File:** `cache/cache.py` — `AsyncCache`

- SQLite-backed (aiosqlite), tables: `chunks` and `reports`
- TTL: chunks 24h, reports 6h
- Lookup by SHA256 hash of lowercase query
- Stale entries auto-deleted on read
- Skipped if `config.cache.enabled = False`

### Stage 3: Retrieval

```python
raw_nested = await asyncio.gather(*retriever_tasks, return_exceptions=True)
```

All active retrievers run in parallel. Active = those whose `supports_modes` includes the decision mode:

| Retriever | Supports Modes | Source |
|-----------|---------------|--------|
| ArxivRetriever | academic, hybrid | `arxiv` Python client |
| SemanticScholarRetriever | academic, hybrid | `semanticscholar` Python client |
| WikipediaRetriever | all | `en.wikipedia.org` REST API |
| HackerNewsRetriever | general, hybrid | `hn.algolia.com/api/v1/search` |
| RSSRetriever | general, hybrid | feedparser on 7 RSS feeds |
| DuckDuckGoRetriever | general, hybrid | `ddgs` library |

#### 3a. DuckDuckGoRetriever

- **`timelimit` auto-detection**: If sub-queries contain temporal keywords (last, recent, latest, this week, this month, past week, past month, new, upcoming), uses `timelimit='m'` (month). Otherwise `timelimit='y'`.
- **Circuit breaker**: After 5 failures, skips for 60s
- **`@with_retry`**: Retries up to 3 times with exponential backoff + jitter on 429/500/502/503
- Sub-queries AND year suffixes sent to DDGS
- No `published_at` on results (DuckDuckGo doesn't return dates)

#### 3b. ArxivRetriever

- Uses `arxiv.Client` with `page_size=50`, 3s delay, 3 retries
- Sort by `SubmittedDate` (newest first)
- Each paper returns: title, entry_id (URL), summary (snippet), published date, authors, categories

#### 3c. SemanticScholarRetriever

- Uses `semanticscholar.SemanticScholar` client
- Searches with `search_paper(query, limit=max_results)`
- Returns: title, paperId/URL, TLDR/abstract, publicationDate, authors

#### 3d. HackerNewsRetriever

- Calls `hn.algolia.com/api/v1/search` with `tags=story`
- Returns: objectID, title, URL, author (`created_at` now captured as `published_at`)

#### 3e. RSSRetriever

- Fetches 7 default RSS feeds (arXiv cs.AI, arXiv cs.LG, TechCrunch AI, VentureBeat AI, HuggingFace blog, DeepMind blog, OpenAI blog) — configurable
- Parses with `feedparser`, extracts title, link, summary, published date
- Post-filter: scores results by keyword overlap with sub-queries, keeps top `max_results`

#### 3f. WikipediaRetriever

- MediaWiki API search: `action=query&list=search`
- Fetches page summary via REST API `page/summary/{title}`
- Returns: title, URL, snippet

**Post-retrieval filtering:**
- `_flatten_deduplicate()` — dedup by URL hash across all retrievers
- `_filter_by_date()` — drops results with `published_at.year < current_year` (e.g. before 2026)

### Stage 4: Cache Partition

```python
cache_hits, misses = await self.cache.partition(raw_results)
```

For each URL, checks if an `ExtractedChunk` exists in the cache. Hits bypass extraction entirely.

### Stage 5: Content Extraction

```python
fresh_chunks = await self.extractors.extract_all(misses, sem)
```

**File:** `extractors/base.py` — `ExtractorChain`

Runs concurrently with a semaphore (default concurrency: 10). Three extractors tried in order:

| Extractor (name) | Method | Strengths |
|-----------------|--------|-----------|
| **TrafilaturaExtractor** (trafilatura) | Downloads page + `trafilatura.extract(format="markdown")` | Best quality, handles most sites |
| **JinaExtractor** (jina) | `r.jina.ai/{url}` proxy API | Falls back when direct fetch fails |
| **ReadabilityExtractor** (readability) | `readability.Document` + `markdownify` | Last resort |

Each chunk stores: `chunk_id`, `source_id`, `url`, `title`, `content_markdown`, `word_count`, `extractor_used`, `extraction_latency_ms`, `metadata`.

Fresh chunks are written to cache: `await self.cache.set_chunk(chunk.url, chunk)`.

### Stage 6: Reranking

```python
scored = await self.reranker.rank(all_chunks, query.raw, top_k)
```

**File:** `reranker/cross_encoder.py` or `reranker/server.py`

Two backends:

#### Local (CrossEncoderReranker)
- Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
- Creates query-document pairs `(query, chunk.content[:512])`
- Predicts relevance scores
- Computes composite score:

```
final = semantic × 0.55 + freshness × 0.20 + authority × 0.15 + length × 0.10
```

Where:
- **semantic**: sigmoid of cross-encoder score
- **freshness**: `1 / (1 + days_old / 30)` — decays over time, 0.5 for unknown
- **authority**: from lookup table (arxiv=0.90, openai=0.78, techcrunch=0.60, duckduckgo=0.40, etc.)
- **length**: `min(1.0, word_count / 300)` — rewards substantive content

#### Remote (ServerReranker)
- POSTs to `{base_url}/rerank` with `{query, documents}`
- Expects `{results: [{index, relevance_score}]}`

Output: `top_k` `ScoredChunk` objects with `semantic_score`, `freshness_score`, `authority_score`, `length_score`, `final_score`, `rank`.

### Stage 7: Synthesis

```python
report = await self.synthesizer.run(query, decision, scored)
```

**File:** `synthesis/synthesizer.py`

#### 7.1 Build System Prompt

```python
system_prompt = build_system_prompt()
```

Contains:
- Role: "expert AI/ML research analyst"
- **CRITICAL DATE RULE**: "Current date is {today}". Explicitly computes relative time periods (e.g. "last 4 weeks" = from {date-28} to {today}). Instructs LLM to NOT use training data dates.
- 6 rules: Markdown only, inline citations [N], no fabrication, note missing coverage, clear headings.
- Report structure template (6 sections).

#### 7.2 Context Truncation

```python
truncated = await self.llm.truncate_to_context(chunks, system_tokens, prompt_overhead_tokens, max_context, response_buffer)
```

Fits chunks into LLM context window. Drops least-important chunks (already sorted by rank from reranker) to fit within available tokens.

#### 7.3 Build Synthesis Prompt

```python
user_prompt = build_synthesis_prompt(query, mode, sub_queries, truncated)
```

Contains:
- Current date
- Research query and mode
- Sub-queries list
- Source blocks: `=== SOURCE [N] ===` with Title, URL, Source, Date, and content (first 800 chars)
- "Generate the report now:"

#### 7.4 LLM Generation

```python
async for token in self.llm.stream(user_prompt, system=system_prompt):
```

**File:** `llm/local.py` or `llm/remote.py`

Local backend (`LocalLLM`):
- POST to `{base_url}/chat/completions` (OpenAI-compatible API on llama.cpp server)
- Messages: `[{"role": "system", "content": system}, {"role": "user", "content": prompt}]`
- Parameters: temperature, max_tokens, stream=True
- Parses SSE stream, yields tokens as they arrive
- Full prompt + response logged to console and log file

Remote backend (`RemoteLLM`):
- Same interface via `openai` Python client
- Supports any OpenAI-compatible API (OpenAI, Anthropic via proxy, etc.)

#### 7.5 Citation Parsing

`_parse_citations(markdown, chunks)` — regex extracts `[N] Title — Source — URL` patterns from generated Markdown.

#### 7.6 Report Assembly

```python
ResearchReport(query, mode, generated_at=datetime.now(), markdown, citations, stats)
```

### Stage 8: Post-Pipeline

```python
report.stats = PipelineStats(...)
await self.cache.set_report(query.raw, report)
report.save(Path(config.output.reports_dir))
```

- Report saved to `reports/{YYYYMMDD_HHMMSS}_{slug}.md`
- Can also export JSON with `--export-json`
- Stats table printed to console

---

## 6. Logging System (logger.py)

### Architecture

- **Async queue**: `log_queue: asyncio.Queue[LogEvent]`
- **LogEvent**: `{level, module, message, timestamp, data, duration_ms}`
- **Console output**: Rich-formatted to stderr: `[{time}] {LEVEL} {module} {message} ({duration})`
- **File output**: `FileLogWriter` drains queue to `logs/agent_YYYYMMDD.log` as JSON Lines

### Log Levels

`DEBUG < INFO < WARNING < ERROR < SUCCESS`

Configurable via `UI_LOG_LEVEL` env var or `config.ui.log_level`. Events below the threshold are still queued and written to file but not printed to console.

### What Gets Logged (comprehensive)

| Module | Event | Data |
|--------|-------|------|
| `startup` | Component validation results | — |
| `pipeline` | Stage transitions, sub-queries, result counts | — |
| `router` | Heuristic classification: mode, weights, confidence | — |
| `duckckgo` | Per-query: sub-query text, timelimit, each result title+URL | — |
| `arxiv` | Per-query text, each paper title + published_at | — |
| `semantic_scholar` | Per-query text, each paper title + published_at | — |
| `hackernews` | Per-query text, each story title + author + created_at | — |
| `rss` | Feed name, entry count, top 3 titles + dates | — |
| `wikipedia` | Per-query text, each article title | — |
| `trafilatura` | URL, word count, latency | — |
| `jina` | URL, word count, latency | — |
| `readability` | URL, word count, latency | — |
| `reranker` | Input count, top 5 scores, output rankings | — |
| `llm` | **Full system prompt** + **full user prompt** + **full response** | `{system_prompt, user_prompt, response}` in JSON file |
| `cache` | Hit/miss counts | — |
| `synthesis` | Token count, citations, latency | — |

---

## 7. Data Models

### Query Flow

```
                    ┌──────────────┐
                    │  UserQuery   │
                    │  raw: str    │
                    │  created_at  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │RouterDecision│
                    │  query       │
                    │  mode        │
                    │  sub_queries │
                    │  weights     │
                    │  explanation │
                    │  classified_by│
                    └──────┬───────┘
```

### Result Flow

```
                    ┌──────────────┐
     Retrieval ────►│  RawResult   │
                    │  id, title   │
                    │  url, snippet│
                    │  source      │
                    │  published_at│
                    │  authors, cat│
                    └──────┬───────┘
                           │ extraction
                           ▼
                    ┌──────────────┐
     Extraction ───►│ExtractedChunk│
                    │  chunk_id    │
                    │  source_id   │
                    │  url, title  │
                    │  content_md  │
                    │  word_count  │
                    │  extractor   │
                    │  metadata    │
                    └──────┬───────┘
                           │ reranking
                           ▼
                    ┌──────────────┐
     Reranking ────►│ ScoredChunk  │
                    │ (all above + │
                    │  semantic    │
                    │  freshness   │
                    │  authority   │
                    │  length      │
                    │  final_score │
                    │  rank)       │
                    └──────┬───────┘
                           │ synthesis
                           ▼
                    ┌──────────────┐
     Output  ──────►│ResearchReport│
                    │  query       │
                    │  mode        │
                    │  generated_at│
                    │  markdown    │
                    │  citations[] │
                    │  stats       │
                    └──────────────┘
```

---

## 8. Cache Architecture (cache.py)

Two-level SQLite cache:

| Level | Table | Key | Value | TTL |
|-------|-------|-----|-------|-----|
| L1 (chunks) | `chunks` | `url` (PK) | `ExtractedChunk.model_dump_json()` | 24h |
| L2 (reports) | `reports` | `sha256(query)` (PK) | `ResearchReport.model_dump_json()` | 6h |

- Concurrency-safe via aiosqlite
- Stale entries deleted on read
- `partition()` splits raw results into cache hits (skip extraction) and misses (need extraction)

---

## 9. Reranker Scoring Details (reranker/scorer.py)

### Freshness
```python
freshness_score(published_at):
    None → 0.5
    else → 1.0 / (1.0 + days_old / 30.0)
```
- 0 days old → 1.0
- 30 days old → 0.5
- 365 days old → 0.076

### Authority
Lookup table:

| Source | Score |
|--------|-------|
| arxiv.org | 0.90 |
| semantic_scholar | 0.88 |
| wikipedia.org | 0.80 |
| deepmind.google | 0.78 |
| openai.com | 0.78 |
| huggingface.co | 0.75 |
| techcrunch.com | 0.60 |
| venturebeat.com | 0.58 |
| hackernews | 0.50 |
| rss | 0.55 |
| duckduckgo | 0.40 |

### Length
```python
length_score(word_count):
    < 50 → 0.2
    else → min(1.0, word_count / 300)
```

### Composite Weight
```
final = semantic × 0.55 + freshness × 0.20 + authority × 0.15 + length × 0.10
```

---

## 10. Error Handling

### Circuit Breaker (`BaseRetriever._circuit`)
- After 5 consecutive failures, retriever enters "open" state for 60s
- After 60s, transitions to "half_open" — next request determines if it closes or re-opens
- Per-retriever state, independent for each

### Retry Decorator (`@with_retry`)
- 3 retries with exponential backoff: `base_delay × 2^attempt`
- Jitter: `delay × random(0.5, 1.0)`
- Non-retryable: 400, 401, 403, 404 (fail fast)
- Retryable: 429, 500, 502, 503

### Exceptions
- `ExtractionError`: content extraction failure
- `LLMError`: LLM backend hard failure
- `LLMContextError`: context window exceeded and truncation fails

### Graceful Shutdown
- SIGINT: begins graceful shutdown (completes current operation)
- Second SIGINT: `os._exit(1)` force quit
- `FileLogWriter.stop()`: cancels drain task

---

## 11. Complete Request/Response Flow (End-to-End Example)

Query: `"top 5 AI news in last 4 weeks"`

```
Step 1: Router (router.py)
───────────────────────────
  QueryRouter._heuristic_classify("top 5 ai news in last 4 weeks")
    General keywords matched: news → general_count=1
    Mode: general (gw=1.0, aw=0.0), confidence=0.20 (1/5)
    Confidence < 0.7 → falls to _llm_classify()
  
  QueryRouter._llm_classify(): simple keyword → mode="general"
  
  QueryRouter._decompose_query():
    Original: "top 5 ai news in last 4 weeks"
    Year suffix: "top 5 ai news in last 4 weeks 2026"
    Recent: "top 5 ai news in last 4 weeks recent"
    Total 3 sub-queries (deduplicated)


Step 2: Pipeline (pipeline.py)
───────────────────────────────
  Active retrievers for "general" mode:
    DuckDuckGo, HackerNews, RSS, Wikipedia
    (arXiv & SemanticScholar excluded — academic only)

  Temporal keywords detected → DuckDuckGo timelimit='m'
  (month, not year)


Step 3: DuckDuckGo (duckduckgo.py)
───────────────────────────────────
  For each sub-query, search via DDGS with timelimit='m'
  Returns: title, href, body snippet (no published_at)
  
  Results logged individually:
    [duckduckgo] timelimit=m | sub-queries: 3
    [duckduckgo] Result #1: title="..." url="..."
    [duckduckgo] "top 5 ai news..." → 10 results


Step 4: HackerNews (hackernews.py)
───────────────────────────────────
  GET hn.algolia.com/api/v1/search?query=...&tags=story
  Returns: title, url, author, created_at


Step 5: RSS (rss.py)
─────────────────────
  Parses 7 RSS feeds, scores entries by keyword relevance
  Returns: title, link, summary, published date


Step 6: Wikipedia (wikipedia.py)
─────────────────────────────────
  MediaWiki search, then page summary REST API


Step 7: Pipeline post-processing
─────────────────────────────────
  _flatten_deduplicate(): removes duplicate URLs
  _filter_by_date(): drops results with published_at.year < 2026


Step 8: Cache partition (cache.py)
───────────────────────────────────
  Checks each URL in SQLite cache
  Hits → skip extraction; Misses → extract content


Step 9: Extraction (extractors/)
─────────────────────────────────
  For each miss URL:
    try TrafilaturaExtractor → try JinaExtractor → try ReadabilityExtractor
  
  Extracted content stored as ExtractedChunk with metadata
  Fresh chunks written to cache


Step 10: Reranking (reranker/)
────────────────────────────────
  CrossEncoderReranker.rank(chunks, query, top_k=20)
  
  For each chunk:
    semantic = sigmoid(cross_encoder.predict(query, chunk.content))
    freshness = freshness_score(chunk.metadata.get("published_at"))
    authority = authority_score(source, url)
    length = length_score(chunk.word_count)
    final = semantic×0.55 + freshness×0.20 + authority×0.15 + length×0.10
  
  Sort by final_score, keep top 20, assign rank 1-20


Step 11: Synthesis (synthesizer.py)
────────────────────────────────────
  System prompt (build_system_prompt):
    "Current date: May 17, 2026"
    CRITICAL DATE RULE with explicit relative date computation
  
  User prompt (build_synthesis_prompt):
    "Current date: May 17, 2026
     Research Query: top 5 ai news in last 4 weeks
     Mode: general
     Sub-queries: ...
     
     === SOURCE [1] ===
     Title: ...
     URL: ...
     Source: duckduckgo
     Date: unknown
     ---
     {content preview}
     
     Generate the report now:"
  
  LLM.stream(prompt, system=system_prompt)
    Messages: [system + user]
    Streams tokens, logs progress every 50 tokens


Step 12: Output
────────────────
  ResearchReport saved to reports/20260517_164307_top-5-ai-news-in-last-4-weeks..md
  Cache updated
  Statistics displayed
```

---

## 12. Date Sensitivity System-wide Summary

Every point in the pipeline that respects the system date:

| Component | What it does | File |
|-----------|-------------|------|
| `_decompose_query()` | Uses `datetime.now().year` for year suffix | `router/router.py:123` |
| `DuckDuckGoRetriever.fetch()` | Detects temporal queries → `timelimit='m'` | `retrievers/duckduckgo.py` |
| `_filter_by_date()` | Drops results with `published_at.year < current_year` | `pipeline.py:183` |
| `freshness_score()` | `datetime.now(UTC)` for recency scoring | `reranker/scorer.py:22` |
| `build_system_prompt()` | `datetime.now().strftime("%B %d, %Y")` + explicit date rules | `synthesis/prompts.py:7-25` |
| `build_synthesis_prompt()` | `Current date: {today}` in user prompt | `synthesis/prompts.py` |
| `build_classification_prompt()` | `Current date: {today}` | `synthesis/prompts.py:79` |
| `LocalLLM._build_messages()` | Sends `system` message with date instructions to model | `llm/local.py:32-37` |
| `ResearchReport.generated_at` | `datetime.now()` | `synthesis/synthesizer.py:96` |
| `LogEvent.timestamp` | `datetime.now()` | `logger.py:31` |
| `FileLogWriter._current_path()` | `datetime.now().strftime("%Y%m%d")` | `logger.py:125` |
