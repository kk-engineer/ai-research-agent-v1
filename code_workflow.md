# AI Research Agent — Code Workflow

## 1. Architecture Overview

```
User Query (CLI or Shell)
        │
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    main.py / cli.py                      │
  │  - Typer CLI (run, shell, config-show, cache-clear,      │
  │           cache-stats, health)                           │
  │  - Interactive shell with /commands                      │
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    startup.py                            │
  │  - validate_connectivity()                               │
  │    ├── LLM check (complete "Respond with 'OK'")          │
  │    ├── Embeddings check (server or OpenAI-compat. API)   │
  │    └── Reranker check (local server or HF CrossEncoder)  │
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    pipeline.py                           │
  │  Pipeline.run(query_str)                                 │
  │                                                          │
  │   1. ROUTE    ───►  QueryRouter.classify()              │
  │   2. CACHE    ───►  AsyncCache.get_report()             │
  │   3. RETRIEVE ───►  9 retrievers in parallel            │
  │        ├── Software-query detection injects              │
  │        │   github_search, github_trending, reddit        │
  │   4. CACHE    ───►  AsyncCache.partition()              │
  │   5. EXTRACT  ───►  ExtractorChain.extract_all()         │
  │   6. RERANK   ───►  ServerReranker / CrossEncoder       │
  │   7. SYNTHESIZE ─►  Synthesizer.run()                    │
  │   8. CACHE    ───►  AsyncCache.set_report()              │
  │   9. SAVE     ───►  ResearchReport.save()                │
  └──────────────────────────────────────────────────────────┘
```

---

## 2. Entry Points

### 2.1 `agent run <query>` (main.py)

`typer` CLI with commands:

| Command | File | Description |
|---------|------|-------------|
| `run <query>` | `main.py` | Single-shot research pipeline |
| `shell` | `main.py` / `cli.py` | Interactive REPL |
| `config-show` | `main.py` | Print resolved config (keys masked) |
| `cache-clear` | `main.py` | Purge SQLite cache |
| `cache-stats` | `main.py` | Show L1/L2 cache entry counts |
| `health` | `main.py` | Test all retrievers + LLM |

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
│   ├── mode: "local" | "cloud"
│   ├── base_url (e.g. http://localhost:8000/v1)
│   ├── model (e.g. mistral-nemo-12b)
│   ├── n_ctx (default 8192)
│   ├── temperature (default 0.3)
│   ├── max_tokens (default 2048)
│   ├── response_buffer (default 1024)
│   ├── remote: LLMRemoteConfig (base_url + model for legacy remote)
│   └── cloud: LLMCloudConfig
│       ├── timeout (default 60s)
│       ├── provider_order: list[str]
│       │   (nvidia → gemini → openrouter → huggingface →
│       │    deepseek → openai → anthropic)
│       └── Per-provider: LLMProviderConfig
│           ├── nvidia (base_url + model)
│           ├── gemini
│           ├── openrouter
│           ├── huggingface
│           ├── deepseek
│           ├── openai
│           └── anthropic
├── embeddings: EmbeddingsConfig
│   ├── mode: "local" | "cloud"
│   ├── base_url (server embedder endpoint)
│   ├── model (e.g. nomic-embed-text)
│   └── cloud: EmbeddingCloudConfig (OpenAI-compatible)
├── retrievers: RetrieverConfig
│   ├── max_results_per_source (default 15)
│   ├── enabled: list[str] (default 9 retrievers)
│   ├── rss_feeds: RSSFeedsConfig (11 default feed URLs)
│   └── reddit: RedditConfig
│       ├── subreddits (LocalLLaMA, MachineLearning, artificial, OpenAI)
│       └── feed_type ("hot")
├── extractors: ExtractorConfig
│   └── extraction_concurrency (default 10)
├── reranker: RerankerConfig
│   ├── top_k (default 20)
│   ├── mode: "local" | "cloud"
│   │   ├── "local" → ServerReranker
│   │   └── "cloud" → CrossEncoderReranker (HF model)
│   ├── base_url (localhost for local)
│   ├── model (GGUF name or HF model)
│   └── weights: semantic=0.55, freshness=0.20,
│                authority=0.15, length=0.10
├── cache: CacheConfig (SQLite DB path + TTLs + enabled flag)
├── timeouts: TimeoutConfig (retriever_s, default 30)
├── output: OutputConfig (reports_dir)
├── log: UIConfig (log_level)  — note: field name is "log"
└── api_keys: APIKeyConfig
    ├── semantic_scholar_api_key
    ├── openai_api_key
    ├── jina_api_key
    ├── nvidia_api_key
    ├── gemini_api_key
    ├── openrouter_api_key
    ├── huggingface_api_key
    ├── deepseek_api_key
    └── anthropic_api_key
```

**Env var overrides** (30+ mapped, e.g. `OPENAI_API_KEY`, `LLM_MODE`, `LLM_BASE_URL`, `LLM_N_CTX`, `RERANKER_MODE`, `CLOUD_LLM_TIMEOUT`). Environment variables take precedence over TOML file values. API keys are also propagated to cloud provider configs automatically.

---

## 4. Startup Validation (startup.py)

`validate_connectivity()` runs before any query:

1. **LLM**: Creates LLM via `create_llm(config)`, sends `"Respond with exactly 'OK'."` via `llm.complete()`. **Hard failure** — exits with code 1 if unreachable (ConnectError, HTTP error, or any exception).
2. **Embeddings**: Pings server embedder at `embeddings.base_url` or `embeddings.cloud.base_url`. **Warning only** on failure.
3. **Reranker**: If `mode == "local"` pings `ServerReranker`; if `mode == "cloud"` loads `CrossEncoder` model from HuggingFace. **Warning only** on failure.

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
2. Match against `ACADEMIC_KEYWORDS` (20 terms: paper, papers, arxiv, research, survey, published, journal, conference, citation, abstract, dataset, benchmark, sota, state of the art, preprint, study, experiment, model architecture, training, fine-tuning) and `GENERAL_KEYWORDS` (22 terms: news, latest, recent, release, announced, trending, product, launch, update, blog, startup, funding, acquisition, interview, podcast, tutorial, demo, github, open source, available now)
3. Compute weights: `academic_weight = academic_matches / total`, `general_weight = general_matches / total`
4. Mode assignment: `academic_weight >= 0.7 → "academic"`, `general_weight >= 0.7 → "general"`, else `"hybrid"`
5. Confidence = `min(1.0, total_matches / 5.0)`
6. If confidence >= 0.7, return immediately with `classified_by="heuristic"`. Otherwise fall back to...

#### 5.1.2 LLM Classification Fallback

`_llm_classify()` — does **not** actually call the LLM. Instead performs simple keyword matching against the same keyword sets:

- If any academic keyword found → `mode="academic"` (aw=0.8, gw=0.2)
- Else if any general keyword found → `mode="general"` (aw=0.2, gw=0.8)
- Else → `mode="hybrid"` (aw=0.5, gw=0.5)

Returns with `classified_by="llm"`. The LLM classification prompt (`build_classification_prompt` in `prompts.py`) exists but is never invoked.

#### 5.1.3 Query Decomposition

`_decompose_query(query, mode)` produces up to 5 sub-queries:

1. **Original query**
2. **"And" splits**: If query contains " and ", split into parts
3. **"Vs" splits**: If query contains " vs ", split into parts
4. **Year suffix**: Appends `" {current_year}"` (e.g. " 2026")
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
- Lookup by SHA256 hash of lowercase trimmed query
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
| RSSRetriever | general, hybrid | feedparser on 11 RSS feeds |
| DuckDuckGoRetriever | general, hybrid | `ddgs` library |
| GitHubSearchRetriever | general, hybrid | `github.com/search` HTML scraping |
| GitHubTrendingRetriever | general, hybrid | `github.com/trending` HTML scraping |
| RedditRetriever | general, hybrid | `reddit.com/r/{sub}/hot.json` |

#### Software Query Detection

Before selecting active retrievers, `Pipeline._is_software_query()` checks for software-related keywords:

```python
SOFTWARE_KEYWORDS = {
    "software", "code", "library", "framework", "api", "sdk", "tool",
    "cli", "implementation", "implement", "build", "docker", "kubernetes",
    "package", "repository", "repo", "npm", "pip", "cargo", "pypi",
    "programming", "language", "compiler", "debugger", "plugin",
    "extension", "middleware", "backend", "frontend", "database", "orm",
    "rest", "graphql", "websocket", "deploy", "devops", "ci/cd",
}
```

`SOFTWARE_PRIORITY_SOURCES = {"github_search", "github_trending", "hackernews", "reddit"}`

If any keyword matches, software-priority sources are injected at the front of the active list (even if they wouldn't normally activate for that mode).

#### 3a. DuckDuckGoRetriever

- **`timelimit` auto-detection**: If sub-queries contain temporal keywords (last, recent, latest, this week, this month, this year, past week, past month, past year, new, upcoming), uses `timelimit='m'` (month). Otherwise `timelimit='y'`.
- **Circuit breaker**: After 5 failures, skips for 60s
- **`@with_retry`**: Retries up to 3 times with exponential backoff + jitter on 429/500/502/503
- Runs DDGS in a thread executor (ddgs is synchronous)
- No `published_at` on results (DuckDuckGo doesn't return dates)

#### 3b. ArxivRetriever

- Uses `arxiv.Client` with `page_size=50`, 3s delay, 3 retries
- Runs in thread executor (arxiv client is synchronous)
- Sort by `SubmittedDate` (newest first)
- Each paper returns: title, entry_id (URL), summary (snippet), published date, authors, categories

#### 3c. SemanticScholarRetriever

- Uses `semanticscholar.SemanticScholar` client
- Runs in thread executor (client is synchronous)
- Searches with `search_paper(query, limit=max_results)`
- Returns: title, paperId/URL, TLDR/abstract, publicationDate, authors

#### 3d. HackerNewsRetriever

- Calls `hn.algolia.com/api/v1/search` with `tags=story` via httpx
- Returns: objectID, title, URL, author, created_at

#### 3e. RSSRetriever

- Fetches 11 default RSS feeds (arXiv cs.AI, arXiv cs.LG, TechCrunch AI, VentureBeat AI, HuggingFace blog, DeepMind blog, OpenAI blog, 4 Reddit RSS feeds) — configurable
- Parses with `feedparser` in thread executor
- Extracts title, link, summary, published date
- Post-filter: scores results by keyword overlap with sub-queries, keeps top `max_results`

#### 3f. WikipediaRetriever

- MediaWiki API search: `action=query&list=search`
- Fetches page summary via REST API `page/summary/{title}`
- Returns: title, URL, snippet

#### 3g. GitHubSearchRetriever

- Scrapes `https://github.com/search?q={query}&type=repositories`
- Parses HTML with BeautifulSoup, selects `[data-testid="results-list"] > div`
- Extracts: repo name, URL, description, language, topics
- No published_at available

#### 3h. GitHubTrendingRetriever

- Scrapes `https://github.com/trending` (optionally with language suffix if detected in query)
- Parses HTML with BeautifulSoup, selects `article.Box-row`
- Extracts: repo name, URL, description, language, stars today
- Language detection from query terms (Python, JS, Rust, Go, etc.)
- No published_at available

#### 3i. RedditRetriever

- Fetches `https://www.reddit.com/r/{sub}/{feed_type}.json` per configured subreddit
- **Circuit breaker**: After 5 failures, skips for 60s
- **`@with_retry`**: 3 retries with exponential backoff
- Post-filter: scores results by keyword overlap against query terms
- Extracts: title, permalink (URL), selftext/URL, created_utc → published_at, score, num_comments, author

**Post-retrieval timing:**
- Each retriever gets `_fetch_with_timing()` wrapper that enforces `config.timeouts.retriever_s` timeout (default 30s)
- Latencies collected into `retriever_latencies` dict
- Retriever failures logged; results silently dropped

**Post-retrieval filtering:**
- `_flatten_deduplicate()` — dedup by URL hash across all retrievers, skips exceptions
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
| **JinaExtractor** (jina) | `r.jina.ai/{url}` proxy API with Bearer token auth | Falls back when direct fetch fails |
| **ReadabilityExtractor** (readability) | `readability.Document` + `markdownify` | Last resort |

Each chunk stores: `chunk_id`, `source_id`, `url`, `title`, `content_markdown`, `word_count`, `extractor_used`, `extraction_latency_ms`, `metadata`.

Fresh chunks are written to cache: `await self.cache.set_chunk(chunk.url, chunk)`.

### Stage 6: Reranking

```python
scored = await self.reranker.rank(all_chunks, query.raw, top_k)
```

**File:** `reranker/server.py` or `reranker/cross_encoder.py`

Two modes, selected via `reranker.mode` in config:

#### Local Mode — ServerReranker (`reranker.mode = "local"`)
- Connects to a locally running reranker server at `base_url` (e.g. `http://localhost:8002`)
- POSTs to `{base_url}/rerank` with `{query, documents, model?}`
- Expects `{results: [{index, relevance_score}]}`
- Use this when you have a reranker model served behind an API (e.g. llama.cpp server, custom FastAPI endpoint)

#### Cloud Mode — CrossEncoderReranker (`reranker.mode = "cloud"`)
- Downloads the model from HuggingFace via `sentence-transformers.CrossEncoder`
- Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Supports GGUF model name resolution via `_resolve_model_name()` (maps short names like `jina-reranker-v2-base-multilingual` to HuggingFace paths)
- Creates query-document pairs `(query, chunk.content[:512])`
- Predicts relevance scores locally with `model.predict(pairs)` in a thread executor
- Device: "mps" (Apple Silicon)

Both modes compute the same composite score:

```
final = semantic × 0.55 + freshness × 0.20 + authority × 0.15 + length × 0.10
```

Where:
- **semantic**: sigmoid of raw model score
- **freshness**: `1 / (1 + days_old / 30)` — decays over time, 0.5 for unknown
- **authority**: from lookup table (arxiv=0.90, semantic_scholar=0.88, wikipedia=0.80, openai=0.78, huggingface=0.75, techcrunch=0.60, hackernews=0.50, duckduckgo=0.40, etc.)
- **length**: `min(1.0, word_count / 300)`, min 0.2 for < 50 words — rewards substantive content

#### Logging

Both modes log the same structured output:

```
[HH:MM:SS] INFO  reranker      Input: query="<query>" | 42 docs | top_k=20
[HH:MM:SS] INFO  reranker      Input documents: [0] Attention Is All You Need | [1] BERT: Pre-training... | ... and 32 more
[HH:MM:SS] INFO  reranker      Model scores (top 5): [0] 0.9821 | [1] 0.8743 | [2] 0.6512 | [3] 0.4321 | [4] 0.2109
[HH:MM:SS] INFO  reranker      Output: 20 chunks ranked | 1.23s
[HH:MM:SS] INFO  reranker      Top 10:
  #1 | 0.8934 | sem:0.7301 fresh:0.8713 auth:0.9000 len:1.0000 | Attention Is All You Need
  #2 | 0.8211 | sem:0.6512 fresh:0.9210 auth:0.8800 len:0.9500 | BERT: Pre-training of Deep Bidirectional...
```

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

**File:** `synthesis/prompts.py`

Contains:
- Role: "expert AI/ML research analyst"
- **CRITICAL DATE RULE**: "Current date is {today}". Explicitly computes relative time periods (e.g. "last 4 weeks" = from {date-28} to {today}). Instructs LLM to NOT use training data dates. Explicit instruction: "DO NOT reference dates earlier than 2026 for news queries."
- 6 rules: Markdown only, inline citations [N], no fabrication, note missing coverage, clear headings.
- Report structure template (6 sections): Executive Summary, Key Findings, Recent Developments, Academic Highlights, Limitations & Gaps, References.

#### 7.2 Context Truncation

```python
truncated = await self.llm.truncate_to_context(chunks, system_tokens, prompt_overhead_tokens, max_context, response_buffer)
```

**File:** `llm/base.py`

Fits chunks into LLM context window using tiktoken (`cl100k_base` encoding). Falls back to word-splitting if tiktoken unavailable. Drops least-important chunks (already sorted by rank from reranker) to fit within `max_context - system_tokens - overhead_tokens - response_buffer`.

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

**File:** `llm/local.py` or `llm/cloud.py`

LLM selection via `create_llm(config)` in `llm/__init__.py`:

- If `config.llm.mode == "cloud"` → `CloudLLM`
- Else → `LocalLLM`

**LocalLLM** (`llm/local.py`):
- POST to `{base_url}/chat/completions` (OpenAI-compatible API on llama.cpp server)
- Messages: `[{"role": "system", "content": system}, {"role": "user", "content": prompt}]`
- Parameters: temperature, max_tokens, stream=True
- Parses SSE stream, yields tokens as they arrive
- Full prompt + response logged to console and log file

**CloudLLM** (`llm/cloud.py`):
- Multi-provider fallback: tries `provider_order` in sequence
- For each provider: builds `AsyncOpenAI` client with provider's `base_url` and `api_key`, sends a `max_tokens=1` ping
- First provider to respond becomes the active provider for the session
- Supported: nvidia, gemini, openrouter, huggingface, deepseek, openai, anthropic
- Same streaming interface via `AsyncOpenAI` SDK
- Raises `RuntimeError` if no provider available (with last error message)

**RemoteLLM** (`llm/remote.py`): Exists but is **not** used by `create_llm()`. Legacy implementation for direct OpenAI-compatible API access.

#### 7.5 Citation Parsing

`_parse_citations(markdown, chunks)` — regex extracts `[N] Title — Source — URL` patterns from generated Markdown:
```python
pattern = r'\[(\d+)\]\s+(.+?)\s+(?:—\s+)?(.+?)\s+(?:—\s+)?(https?://\S+)'
```

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
- Stats table printed to console, including:
  - `total_results_fetched`, `total_chunks_extracted`, `cache_hits`, `chunks_after_rerank`
  - Per-retriever latencies table
  - Extraction, rerank, synthesis, total latencies
  - LLM tokens used and backend name

---

## 6. Logging System (logger.py)

### Architecture

- **Async queue**: `log_queue: asyncio.Queue[LogEvent]`
- **LogEvent**: `{level, module, message, timestamp, data, duration_ms}`
- **Console output**: Rich-formatted to stderr: `[{time}] {LEVEL} {module} {message}`
- **File output**: `FileLogWriter` drains queue to `logs/agent_YYYYMMDD.log` as JSON Lines

### Log Levels

`DEBUG < INFO < WARNING < ERROR < SUCCESS`

Configurable via `LOG_LEVEL` env var or `config.log.log_level`. Events below the threshold are still queued and written to file but not printed to console.

### What Gets Logged (comprehensive)

| Module | Event | Data |
|--------|-------|------|
| `startup` | Component validation results | — |
| `pipeline` | Stage transitions, sub-queries, result counts, per-result details | — |
| `router` | Heuristic classification: mode, weights, confidence | — |
| `duckduckgo` | Per-query: sub-query text, timelimit, each result title+URL | — |
| `arxiv` | Per-query text, each paper title + published_at | — |
| `semantic_scholar` | Per-query text, each paper title + published_at | — |
| `hackernews` | Per-query text, each story title + author + created_at | — |
| `rss` | Feed name, entry count, top 3 titles + dates | — |
| `wikipedia` | Per-query text, each article title | — |
| `github_search` | Per-query text, result count, latency | — |
| `github_trending` | Fetch results | — |
| `reddit` | Per-subreddit post count, latency | — |
| `trafilatura` | URL, word count, latency | — |
| `jina` | URL, word count, latency | — |
| `readability` | URL, word count, latency | — |
| `reranker` | Input query + doc titles, top 5 model scores, output ranked list with score breakdown (sem/fresh/auth/len), latency | — |
| `llm` | **Full system prompt** + **source overview** + **full response** | `{system_prompt, user_prompt, response}` in JSON file |
| `cache` | Hit/miss counts | — |
| `synthesis` | Token count, citations, latency | — |
| `retry` | Retry attempts with delay | — |

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
                    │  ("heuristic"│
                    │   or "llm")  │
                    └──────┬───────┘
```

### Result Flow

```
                    ┌──────────────┐
     Retrieval ────►│  RawResult   │
                    │  id (sha256) │
                    │  title, url  │
                    │  snippet     │
                    │  source      │
                    │  published_at│
                    │  authors, cat│
                    │  raw_html?   │
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
| L2 (reports) | `reports` | `sha256(lowercase(query))` (PK) | `ResearchReport.model_dump_json()` | 6h |

- Concurrency-safe via aiosqlite
- Stale entries deleted on read
- `partition()` splits raw results into cache hits (skip extraction) and misses (need extraction)
- `stats()` returns `{l1_entries, l2_entries}`
- `clear()` deletes all rows from both tables
- Entire cache disabled when `config.cache.enabled = False`

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

| Source Pattern | Score |
|----------------|-------|
| arxiv.org / semantic_scholar | 0.90 / 0.88 |
| wikipedia.org | 0.80 |
| deepmind.google / openai.com | 0.78 |
| huggingface.co | 0.75 |
| techcrunch.com | 0.60 |
| venturebeat.com | 0.58 |
| hackernews | 0.50 |
| rss | 0.55 |
| duckduckgo | 0.40 |

Fallback for unknown domains: 0.45. Matches by domain substring.

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
- Max delay: 30s
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
    Domain: "top 5 ai news in last 4 weeks machine learning"
    Domain: "top 5 ai news in last 4 weeks AI research"
    Domain: "top 5 ai news in last 4 weeks deep learning"
    Total 3 sub-queries (5 after domain dedup)


Step 2: Pipeline (pipeline.py)
───────────────────────────────
  Active retrievers for "general" mode:
    DuckDuckGo, HackerNews, RSS, Wikipedia
    (arXiv & SemanticScholar excluded — academic only)
    No software keywords → GitHub/Reddit not injected

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
  Parses 11 RSS feeds, scores entries by keyword relevance
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
  # Mode depends on config.reranker.mode:
  #   "local"  → ServerReranker (POST to base_url/rerank)
  #   "cloud"  → CrossEncoderReranker (load HF model locally)
  
  reranker.rank(chunks, query, top_k=20)
  
  Logs:
    Input: query="top 5 ai news in last 4 weeks" | 35 docs | top_k=20
    Input documents: [0] AI Startup Raises $500M ... | [1] New RLHF Technique ...
    
  For each chunk:
    semantic = sigmoid(model_score[chunk])
    freshness = freshness_score(chunk.metadata.get("published_at"))
    authority = authority_score(source, url)
    length = length_score(chunk.word_count)
    final = semantic×0.55 + freshness×0.20 + authority×0.15 + length×0.10
  
  Sort by final_score, keep top 20, assign rank 1-20
  
  Logs:
    Model scores (top 5): [0] 0.9821 | [1] 0.8743 | [2] 0.6512 ...
    Output: 20 chunks ranked | 1.23s
    Top 10:
      #1 | 0.8934 | sem:0.7301 fresh:0.8713 auth:0.9000 len:1.0000 | AI Startup Raises $500M
      #2 | 0.8211 | sem:0.6512 fresh:0.9210 auth:0.8800 len:0.9500 | New RLHF Technique ...


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
    If mode="local": POST to llama.cpp /chat/completions SSE
    If mode="cloud": AsyncOpenAI SDK streaming → first available provider
    Messages: [system + user]
    Streams tokens, logs progress every 50 tokens


Step 12: Output
────────────────
  ResearchReport saved to reports/20260517_164307_top-5-ai-news-in-last-4-weeks..md
  Cache updated
  Statistics displayed (latency breakdown, per-retriever table)
```

---

## 12. LLM Backend Selection (llm/__init__.py)

```python
def create_llm(config: AppConfig) -> BaseLLM:
    if config.llm.mode == "cloud":
        return CloudLLM(config)     # Multi-provider fallback
    return LocalLLM(config.llm)     # Local llama.cpp server
```

### LocalLLM (llm/local.py)
- Connects to `{base_url}/chat/completions` (OpenAI-compatible)
- Uses httpx AsyncClient with 120s timeout
- SSE streaming with `[DONE]` termination
- Token counting via tiktoken

### CloudLLM (llm/cloud.py)
- Iterates `provider_order` list at init
- Pings each provider with `max_tokens=1` to verify availability
- First available provider becomes active for session
- Supported: nvidia, gemini, openrouter, huggingface, deepseek, openai, anthropic
- API key resolution: `PROVIDER_API_KEY_MAP` → `config.api_keys.*` → env vars
- `RuntimeError` if no provider available

### RemoteLLM (llm/remote.py) — Legacy/Unused
- Direct OpenAI-compatible API via `AsyncOpenAI` client
- NOT instantiated by `create_llm()`; kept for backward compatibility

---

## 13. Software Query Detection (pipeline.py)

The pipeline injects software-priority retrievers when a query contains software development keywords:

```python
SOFTWARE_KEYWORDS = {
    "software", "code", "library", "framework", "api", "sdk",
    "tool", "cli", "implementation", "docker", "kubernetes",
    "package", "repository", "repo", "npm", "pip", "cargo",
    "programming", "language", "compiler", "debugger",
    "plugin", "extension", "middleware", "backend", "frontend",
    "database", "orm", "rest", "graphql", "websocket",
    "deploy", "devops", "ci/cd",
}

SOFTWARE_PRIORITY_SOURCES = {"github_search", "github_trending", "hackernews", "reddit"}
```

`Pipeline._is_software_query()` performs a substring check (not exact token match), so "kubernetes" matches "kubernetes" anywhere in the query string.

---

## 14. Date Sensitivity System-wide Summary

Every point in the pipeline that respects the system date:

| Component | What it does | File |
|-----------|-------------|------|
| `_decompose_query()` | Uses `datetime.now().year` for year suffix | `router/router.py:123` |
| `DuckDuckGoRetriever.fetch()` | Detects temporal queries → `timelimit='m'` | `retrievers/duckduckgo.py` |
| `_filter_by_date()` | Drops results with `published_at.year < current_year` | `pipeline.py` |
| `freshness_score()` | `datetime.now(UTC)` for recency scoring | `reranker/scorer.py:22` |
| `build_system_prompt()` | `datetime.now().strftime("%B %d, %Y")` + explicit date rules | `synthesis/prompts.py:7-54` |
| `build_synthesis_prompt()` | `Current date: {today}` in user prompt | `synthesis/prompts.py` |
| `build_classification_prompt()` | `Current date: {today}` | `synthesis/prompts.py:98` |
| `LocalLLM._build_messages()` | Sends `system` message with date instructions to model | `llm/local.py:32-37` |
| `CloudLLM` | Sends system+user messages to cloud provider | `llm/cloud.py` |
| `ResearchReport.generated_at` | `datetime.now()` | `synthesis/synthesizer.py:114` |
| `LogEvent.timestamp` | `datetime.now()` | `logger.py:31` |
| `FileLogWriter._current_path()` | `datetime.now().strftime("%Y%m%d")` | `logger.py:125` |
