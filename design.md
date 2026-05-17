# AI Research Agent — Design Document
> **Version:** 1.0.0 | **Target:** Python 3.12+ | **Package Manager:** uv | **Platform:** macOS M1 (Apple Silicon)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Tech Stack](#4-tech-stack)
5. [Configuration System](#5-configuration-system)
6. [Module Specifications](#6-module-specifications)
   - 6.1 [Query Router](#61-query-router)
   - 6.2 [Retrievers](#62-retrievers)
   - 6.3 [Content Extractor](#63-content-extractor)
   - 6.4 [Reranker](#64-reranker)
   - 6.5 [LLM Interface](#65-llm-interface)
   - 6.6 [Synthesizer](#66-synthesizer)
   - 6.7 [Cache Layer](#67-cache-layer)
   - 6.8 [CLI & UI](#68-cli--ui)
7. [Data Models](#7-data-models)
8. [Async Execution Strategy](#8-async-execution-strategy)
9. [Logging Architecture](#9-logging-architecture)
10. [pyproject.toml](#10-pyprojecttoml)
11. [config.toml](#11-configtoml)
12. [Error Handling & Resilience](#12-error-handling--resilience)
13. [Performance Targets](#13-performance-targets)
14. [Extension Points](#14-extension-points)

---

## 1. Project Overview

The **AI Research Agent** is a CLI-based autonomous research tool that:

- Accepts a natural language query about AI/ML topics
- Routes it to one or more specialised retrieval pipelines (academic vs. general)
- Fetches, extracts, and cleans raw content from multiple sources concurrently
- Reranks and deduplicates results for relevance
- Synthesises a structured, cited Markdown report using a local or remote LLM
- Displays all progress and output through a polished, async-safe Rich/Textual terminal UI

### Design Principles

| Principle | How It Is Applied                                                                                                                                                                                                                                                                      |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Async-first** | Every I/O call uses `asyncio` + `httpx.AsyncClient`; no blocking code on the main loop                                                                                                                                                                                                 |
| **Accuracy over speed** | Reranking + multi-source fusion; configurable result depth                                                                                                                                                                                                                             |
| **Local-first** | `llama-cpp-python` on Apple Silicon MPS by default; remote Nvidia, OpenRouter, HF, Groq etc. APIs to be read from config.toml file api_key to be read from exported environment. Give options to enable all and fallback to other in case the first API call fails or is rate limited. |
| **Modular** | Each stage is an independent, swappable module behind a clear `ABC` interface                                                                                                                                                                                                          |
| **Observable** | Every stage emits structured log events through an async queue; the UI renders them live                                                                                                                                                                                               |
| **Resilient** | Per-retriever circuit breakers, exponential back-off, and graceful partial failures                                                                                                                                                                                                    |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CLI ENTRY POINT                             │
│                   src/agent/main.py  (Typer + Textual)               │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  UserQuery
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        QUERY ROUTER                                  │
│             src/agent/router/router.py                               │
│                                                                      │
│  Input  : raw query string                                           │
│  Output : RouterDecision {academic, general, hybrid}                 │
│  Method : keyword heuristics + optional LLM classification           │
└────────────────┬──────────────────────────┬──────────────────────────┘
                 │ academic                 │ general / hybrid
                 ▼                          ▼
┌───────────────────────┐     ┌────────────────────────────────────────┐
│  ACADEMIC RETRIEVERS  │     │         GENERAL RETRIEVERS             │
│  retrievers/          │     │         retrievers/                    │
│  ├─ arxiv.py          │     │  ├─ wikipedia.py                       │
│  └─ semantic_scholar  │     │  ├─ hackernews.py                      │
│       .py             │     │  ├─ rss.py  (configurable feeds)       │
│                       │     │  └─ duckduckgo.py                      │
└────────────┬──────────┘     └──────────────────┬─────────────────────┘
             │  List[RawResult]                   │  List[RawResult]
             └────────────────┬───────────────────┘
                              │  merged, deduplicated
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      CONTENT EXTRACTOR                               │
│               src/agent/extractors/                                  │
│                                                                      │
│  Priority chain:  Trafilatura → Jina Reader → Readability fallback   │
│  Output: List[ExtractedChunk]  (clean markdown, metadata)            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        CACHE CHECK                                   │
│              src/agent/cache/cache.py  (DiskCache / SQLite)          │
│  Skip re-fetch for URLs seen within TTL window                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       RERANKER                                       │
│              src/agent/reranker/                                     │
│                                                                      │
│  Scores on: semantic similarity, freshness, source authority         │
│  Models:    local cross-encoder (default)  |  jina-reranker-v2           │
│  Output:    List[ScoredChunk]  top-K kept                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LLM SYNTHESIS                                   │
│              src/agent/llm/  +  src/agent/synthesis/                 │
│                                                                      │
│  Local:   llama-cpp-python (Mistral Nemo 12B GGUF on MPS)          │
│  Remote:  OpenAI-compat endpoint (opt-in)                            │
│  Output:  ResearchReport (structured Markdown + citations JSON)      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       FINAL OUTPUT                                   │
│  • Rich-rendered Markdown in terminal                                 │
│  • Saved to  reports/<timestamp>_<slug>.md                           │
│  • Optional JSON export with full source metadata                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Async Pipeline Overview

```
asyncio.run(main())
    └── Pipeline.run(query)
            ├── router.classify()                  # fast, sync or 1 LLM call
            ├── asyncio.gather(                    # concurrent retrieval
            │       arxiv.fetch(),
            │       semantic_scholar.fetch(),
            │       wikipedia.fetch(),
            │       hackernews.fetch(),
            │       rss.fetch(),
            │       duckduckgo.fetch()
            │   )
            ├── asyncio.gather(                    # concurrent extraction
            │       *[extractor.extract(r) for r in results]
            │   )
            ├── reranker.rank(chunks)              # batch, CPU/MPS
            └── synthesizer.generate(top_k)        # streaming LLM
```

---

## 3. Folder Structure

```
ai-research-agent/
│
├── pyproject.toml                   # uv-managed deps, scripts, tool config
├── config.toml                      # user-facing configuration (NOT committed)
├── config.example.toml              # committed template with all keys
├── .env                             # optional override (API keys if not in config.toml)
├── .gitignore
├── README.md
│
├── models/                          # GGUF model files (gitignored)
│   └── .gitkeep
│
├── reports/                         # generated reports (gitignored)
│   └── .gitkeep
│
├── cache/                           # SQLite cache DB (gitignored)
│   └── .gitkeep
│
└── src/
    └── agent/
        │
        ├── __init__.py
        ├── main.py                  # Typer CLI entry point; launches Textual app
        ├── pipeline.py              # Orchestrator: wires all stages together
        │
        ├── config.py                # Loads config.toml → exports to os.environ
        │
        ├── models/                  # Pydantic v2 data models (pure data, no I/O)
        │   ├── __init__.py
        │   ├── query.py             # UserQuery, RouterDecision
        │   ├── result.py            # RawResult, ExtractedChunk, ScoredChunk
        │   └── report.py            # ResearchReport, Citation
        │
        ├── router/
        │   ├── __init__.py
        │   └── router.py            # QueryRouter (heuristic + LLM fallback)
        │
        ├── retrievers/
        │   ├── __init__.py
        │   ├── base.py              # BaseRetriever ABC
        │   ├── arxiv.py             # arXiv API client
        │   ├── semantic_scholar.py  # Semantic Scholar API client
        │   ├── wikipedia.py         # Wikipedia REST API + Wikidata
        │   ├── hackernews.py        # HN Algolia search API
        │   ├── rss.py               # feedparser over configurable feed list
        │   └── duckduckgo.py        # duckduckgo-search (HTML scrape, no key)
        │
        ├── extractors/
        │   ├── __init__.py
        │   ├── base.py              # BaseExtractor ABC
        │   ├── trafilatura.py       # Trafilatura extractor (primary)
        │   ├── jina.py              # Jina Reader r.jina.ai (fallback)
        │   └── readability.py       # readability-lxml (last resort)
        │
        ├── reranker/
        │   ├── __init__.py
        │   ├── base.py              # BaseReranker ABC
        │   ├── cross_encoder.py     # sentence-transformers cross-encoder
        │   └── scorer.py            # Freshness + authority scoring helpers
        │
        ├── llm/
        │   ├── __init__.py
        │   ├── base.py              # BaseLLM ABC (complete + stream methods)
        │   ├── local.py             # llama-cpp-python backend (MPS on M1)
        │   └── remote.py            # OpenAI-compatible HTTP backend
        │
        ├── embeddings/
        │   ├── __init__.py
        │   ├── base.py              # BaseEmbedder ABC
        │   ├── local.py             # llama.cpp embeddings endpoint
        │   └── sentence_transformer.py  # ST fallback
        │
        ├── synthesis/
        │   ├── __init__.py
        │   ├── synthesizer.py       # Report builder; calls LLM with prompt template
        │   └── prompts.py           # All prompt templates (f-strings / Jinja2)
        │
        ├── cache/
        │   ├── __init__.py
        │   └── cache.py             # SQLite-backed async cache (aiosqlite)
        │
        └── ui/
            ├── __init__.py
            ├── app.py               # Textual Application (main TUI)
            ├── logger.py            # AsyncLogQueue + QueueHandler
            ├── panels.py            # Rich renderables: status, progress, report
            └── theme.py             # Color palette, styles
```

---

## 4. Tech Stack

### Core Runtime

| Package | Version | Purpose |
|---|---|---|
| `python` | `≥3.12` | Language runtime |
| `uv` | latest | Package & venv manager |
| `typer` | `≥0.12` | CLI argument parsing |
| `asyncio` | stdlib | Async event loop |
| `httpx` | `≥0.27` | Async HTTP client (replaces requests) |

### Retrieval

| Package           | Purpose |
|-------------------|---|
| `arxiv`           | Official arXiv Python SDK |
| `semanticscholar` | Semantic Scholar API client |
| `wikipedia-api`   | Wikipedia REST API wrapper |
| `ddgs`            | DDG HTML search (no API key) |
| `feedparser`      | RSS/Atom feed parser |

### Content Extraction

| Package | Purpose |
|---|---|
| `trafilatura` | Primary web → clean text extractor |
| `readability-lxml` | Fallback boilerplate remover |
| `markdownify` | HTML → Markdown conversion |
| `httpx` | For Jina Reader REST calls |

### Reranking & Embeddings

| Package | Purpose |
|---|---|
| `sentence-transformers` | Cross-encoder reranking; ST embeddings fallback |
| `torch` | Backend for sentence-transformers (MPS on M1) |
| `llama-cpp-python` | Local embedding via GGUF model |

### LLM

| Package | Purpose |
|---|---|
| `llama-cpp-python` | Local inference on Apple MPS (`n_gpu_layers=-1`) |
| `openai` | Optional remote LLM via OpenAI-compatible API |
| `tiktoken` | Token counting for context window management |

### Data & Config

| Package | Purpose |
|---|---|
| `pydantic` | `≥2.0` — data models, validation, serialisation |
| `tomllib` | stdlib (3.11+) — parse config.toml |
| `aiosqlite` | Async SQLite for cache |
| `diskcache` | Disk-backed key-value cache (secondary) |

### CLI / UI

| Package | Purpose |
|---|---|
| `rich` | Markdown rendering, panels, tables, progress |
| `textual` | Full TUI app framework |

### Dev & Quality

| Package | Purpose |
|---|---|
| `ruff` | Linting + formatting |
| `mypy` | Static type checking |
| `pytest` + `pytest-asyncio` | Testing |
| `pytest-httpx` | Mock async HTTP calls |

---

## 5. Configuration System

### 5.1 Loading Strategy

`src/agent/config.py` is the **single source of truth** for all config:

```python
# src/agent/config.py  (pseudocode — implement fully)

import tomllib, os
from pathlib import Path
from pydantic import BaseModel

CONFIG_PATH = Path("config.toml")

class Config(BaseModel):
    # nested sub-models for each section
    llm: LLMConfig
    retrievers: RetrieverConfig
    reranker: RerankerConfig
    cache: CacheConfig
    ui: UIConfig
    api_keys: APIKeyConfig   # all sensitive values

_config: Config | None = None

def load_config(path: Path = CONFIG_PATH) -> Config:
    global _config
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    _config = Config.model_validate(raw)
    # Export all api_keys to environment so sub-processes & libs pick them up
    for field, value in _config.api_keys.model_dump().items():
        if value:
            os.environ[field.upper()] = value
    return _config

def get_config() -> Config:
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
```

- Config is loaded **once** at startup in `main.py`.
- All modules import `get_config()` — never read TOML files themselves.
- Environment variables (set before launch) override `config.toml` values.

### 5.2 Secrets Priority

```
os.environ  >  config.toml [api_keys]  >  .env file (loaded by main.py)
```

---

## 6. Module Specifications

### 6.1 Query Router

**File:** `src/agent/router/router.py`

**Responsibility:** Classify an incoming query and decide which retrievers to activate.

**Interface:**
```python
class QueryRouter:
    async def classify(self, query: str) -> RouterDecision: ...
```

**RouterDecision fields:**
```python
class RouterDecision(BaseModel):
    query: str
    mode: Literal["academic", "general", "hybrid"]
    sub_queries: list[str]        # expanded/decomposed queries for retrieval
    academic_weight: float        # 0.0–1.0
    general_weight: float
    explanation: str              # for logging/debugging
```

**Classification Logic (two-tier):**

1. **Fast Heuristic** (no LLM call, <1 ms):
   - Keywords like `paper`, `arxiv`, `research`, `survey`, `published` → `academic`
   - Keywords like `news`, `latest`, `release`, `trending`, `announcement` → `general`
   - Both present or ambiguous → `hybrid`

2. **LLM Fallback** (triggered only when heuristic confidence < threshold):
   - Single LLM call with a compact classification prompt
   - Returns structured JSON via constrained generation

**Query Decomposition:**
The router also expands the query into 2–4 sub-queries for broader retrieval (e.g. `"latest LLM developments"` → `["large language model research 2025"`, `"LLM news announcements"`, `"transformer architecture papers"]`).

---

### 6.2 Retrievers

**Base Interface:** `src/agent/retrievers/base.py`

```python
class BaseRetriever(ABC):
    name: str
    supports_modes: list[Literal["academic", "general"]]

    @abstractmethod
    async def fetch(
        self,
        queries: list[str],
        max_results: int,
    ) -> list[RawResult]: ...

    async def health_check(self) -> bool: ...
```

**RawResult fields:**
```python
class RawResult(BaseModel):
    id: str                        # deduplication key
    title: str
    url: str
    snippet: str                   # short preview / abstract
    source: str                    # "arxiv" | "hackernews" | ...
    published_at: datetime | None
    authors: list[str]
    categories: list[str]
    raw_html: str | None           # for extraction if fetched
    fetch_latency_ms: float
```

#### arXiv Retriever

- Uses `arxiv` Python SDK with `asyncio.to_thread()` (SDK is sync)
- Searches `cs.AI`, `cs.LG`, `cs.CL`, `stat.ML` categories
- Retrieves: title, abstract (used as snippet), authors, PDF URL, published date
- Default max results: `20` per query

#### Semantic Scholar Retriever

- Uses `semanticscholar` SDK via `asyncio.to_thread()`
- Enriches arXiv results with citation count and influence score
- Falls back gracefully if API key absent (public rate limit)

#### Wikipedia Retriever

- Uses Wikipedia REST API (`/page/summary/{title}`, `/page/search`)
- No API key required
- Primary use: factual background context for technical terms

#### HackerNews Retriever

- Uses Algolia HN Search API: `http://hn.algolia.com/api/v1/search`
- Filters by `tags=story` and sorts by `search_by_date`
- Captures: title, URL, score, comment count, posted date

#### RSS Retriever

- Parses a configurable list of feed URLs via `feedparser` in a thread pool
- Default feeds (configurable in `config.toml`):
  - `https://rss.arxiv.org/rss/cs.AI`
  - `https://techcrunch.com/category/artificial-intelligence/feed/`
  - `https://venturebeat.com/ai/feed/`
  - `https://feeds.feedburner.com/oreilly/radar`
  - `https://huggingface.co/blog/feed.xml`
  - `https://deepmind.google/blog/rss.xml`

#### DuckDuckGo Retriever

- Uses `duckduckgo-search` library (no API key)
- Performs text search and news search
- Used as a broad fallback / discovery layer

---

### 6.3 Content Extractor

**File:** `src/agent/extractors/`

**Purpose:** Fetch full page content and convert it to clean Markdown chunks.

**Priority Chain (tried in order, first success wins):**

```
1. Trafilatura   — best boilerplate removal, handles paywalls partially
2. Jina Reader   — r.jina.ai/URL  (free tier, external call)
3. Readability   — readability-lxml, last resort
```

**Interface:**
```python
class BaseExtractor(ABC):
    @abstractmethod
    async def extract(self, url: str, raw_html: str | None = None) -> ExtractedChunk | None: ...
```

**ExtractedChunk fields:**
```python
class ExtractedChunk(BaseModel):
    source_id: str              # matches RawResult.id
    url: str
    title: str
    content_markdown: str       # clean extracted text
    word_count: int
    extractor_used: str
    extraction_latency_ms: float
    metadata: dict              # authors, date, etc.
```

**Chunking strategy:**
- After extraction, split long documents into `chunk_size` (default: 512) token windows with `chunk_overlap` (default: 64) using a simple sliding window
- Each sub-chunk retains parent URL and title for citation

---

### 6.4 Reranker

**File:** `src/agent/reranker/`

**Purpose:** Score all chunks against the original query; keep top-K.

**Scoring formula:**
```
final_score = (
    semantic_score  * weights.semantic   +   # cross-encoder cosine sim
    freshness_score * weights.freshness  +   # decay: 1/(1 + days_old/30)
    authority_score * weights.authority  +   # per-source static weight
    length_score    * weights.length         # penalise very short chunks
)
```

**Default weights** (tunable in `config.toml`):
```toml
[reranker.weights]
semantic   = 0.55
freshness  = 0.20
authority  = 0.15
length     = 0.10
```

**Authority table** (per source):
| Source | Score |
|---|---|
| arXiv | 0.90 |
| Semantic Scholar | 0.88 |
| Wikipedia | 0.80 |
| DeepMind / OpenAI blogs | 0.78 |
| TechCrunch / VentureBeat | 0.60 |
| HackerNews | 0.50 |
| DuckDuckGo generic | 0.40 |

**Cross-encoder model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, ~22 MB)

**Alternative:** `BAAI/bge-reranker-base` for higher accuracy (configurable).

**MPS acceleration:** `torch.device("mps")` is set automatically on Apple Silicon.

---

### 6.5 LLM Interface

**File:** `src/agent/llm/`

#### Local Backend (`local.py`)

```python
class LocalLLM(BaseLLM):
    """
    Wraps llama-cpp-python for local GGUF inference.
    Uses MPS (Metal) on Apple Silicon via n_gpu_layers=-1.
    """
    def __init__(self, config: LLMConfig): ...

    async def complete(self, prompt: str, **kwargs) -> str:
        # run in asyncio.to_thread() to avoid blocking event loop
        ...

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # yields token strings; UI subscribes to this
        ...
```

**Recommended models for M1 Mac:**

| Model | Size | Use case |
|---|---|---|
| `Llama-3.2-3B-Instruct.Q8_0.gguf` | ~3 GB | Fast, fits in 8 GB RAM |
| `Mistral-7B-Instruct-v0.3.Q5_K_M.gguf` | ~5 GB | Good quality / speed balance |
| `Llama-3.1-8B-Instruct.Q4_K_M.gguf` | ~4.7 GB | Best quality for the size |

**Default model path:** `./models/` (configured in `config.toml`).

**Context window management:**
- Count tokens with `tiktoken` before sending
- If context > `max_context_tokens`, truncate chunks from the bottom (keep most relevant at top)
- Always reserve `response_buffer_tokens` (default: 1024) for the response

#### Remote Backend (`remote.py`)

```python
class RemoteLLM(BaseLLM):
    """
    OpenAI-compatible endpoint. Works with:
    - OpenAI API
    - Groq
    - Together AI
    - Ollama (local HTTP server, alternative to llama-cpp-python)
    - Any OpenAI-spec server
    """
```

---

### 6.6 Synthesizer

**File:** `src/agent/synthesis/synthesizer.py`

**Responsibility:** Build the final report from ranked chunks.

**Report structure:**
```markdown
# [Title derived from query]
> *Generated: {timestamp} | Sources: {n} | Query: {query}*

## Executive Summary
[2–3 sentence TL;DR]

## Key Findings
### [Topic 1]
[Synthesised content with inline citations [1][2]]

### [Topic 2]
...

## Recent Developments
[News and announcements]

## Academic Highlights
[Paper summaries with authors, date, link]

## Limitations & Gaps
[What the sources don't cover; areas of uncertainty]

## References
[1] Title — Source — URL — Date
[2] ...
```

**Prompt design principles:**
- System prompt defines: role (expert AI research analyst), output format (Markdown), citation style (`[N]`), tone (clear, technical but accessible)
- All source chunks are injected into the user message with their index number for grounding
- Temperature: `0.3` (factual focus, low hallucination)
- LLM is instructed to cite only from provided context; never fabricate

---

### 6.7 Cache Layer

**File:** `src/agent/cache/cache.py`

**Two-level cache:**

| Level | Backend | TTL | Scope |
|---|---|---|---|
| L1 — URL content | `aiosqlite` | 24 hours | Extracted chunks by URL |
| L2 — Query results | `diskcache` | 6 hours | Full pipeline output by query hash |

```python
class AsyncCache:
    async def get_chunk(self, url: str) -> ExtractedChunk | None: ...
    async def set_chunk(self, url: str, chunk: ExtractedChunk) -> None: ...
    async def get_report(self, query_hash: str) -> ResearchReport | None: ...
    async def set_report(self, query_hash: str, report: ResearchReport) -> None: ...
    async def invalidate(self, url: str | None = None) -> None: ...
```

Cache is checked **before extraction**; hit → skip HTTP fetch and extraction entirely.

---

### 6.8 CLI & UI

**File:** `src/agent/ui/`

#### Entry Point (`main.py`)

```bash
# Usage
agent run "latest developments in multimodal LLMs"
agent run "transformer architecture papers 2025" --mode academic --top-k 15
agent run "AI news today" --no-cache --export-json
agent config --show
agent cache --clear
```

Built with `typer`. Launches the Textual TUI application for long runs.

#### Textual Application (`app.py`)

```
┌─ AI Research Agent ──────────────────────────────────────────────┐
│  Query: "latest developments in multimodal LLMs"                 │
├──────────────────────────────────────────────────────────────────┤
│  PIPELINE PROGRESS                               [=====>   ] 65% │
│  ✅ Router      → hybrid (academic 0.6 / general 0.4)            │
│  ✅ Retrievers  → 47 results in 2.1s                             │
│  ✅ Extraction  → 38 chunks extracted (9 cached)                 │
│  🔄 Reranking  → scoring 38 chunks...                            │
│  ⏳ Synthesis  → waiting                                         │
├─────────────────────────────┬────────────────────────────────────┤
│  SOURCES                    │  LOG                               │
│                             │                                    │
│  📄 arXiv         (12)      │  14:02:31 [arxiv] Fetched 12 papers│
│  📄 Semantic Sch. ( 8)      │  14:02:31 [ddg]   Fetched 9 pages  │
│  🌐 Wikipedia     ( 3)      │  14:02:32 [extractor] trafilatura  │
│  📰 RSS Feeds     ( 7)      │    ✓ openai.com/research/gpt4o     │
│  💬 HackerNews    ( 5)      │  14:02:33 [reranker] scored 38/38  │
│  🔍 DuckDuckGo    ( 9)      │  14:02:33 [llm] streaming...       │
│                             │                                    │
├─────────────────────────────┴────────────────────────────────────┤
│  REPORT (streaming)                                              │
│                                                                  │
│  # Latest Developments in Multimodal LLMs                        │
│  > Generated: 2025-07-14 14:02:41 | Sources: 44 | ...            │
│                                                                  │
│  ## Executive Summary                                            │
│  Multimodal large language models have seen rapid advancement... │
│  ▌                                                               │
└──────────────────────────────────────────────────────────────────┘
 [Q] Quit  [S] Save Report  [J] Export JSON  [C] Clear Cache  [?] Help
```

The Textual app uses:
- `Header` widget: query + timestamp
- `ProgressBar`: overall pipeline progress
- `DataTable`: source counts
- `RichLog`: streaming log panel (fed by async queue)
- `Markdown` widget: streaming report output
- `Footer`: keyboard shortcuts

#### Async Log Queue (`logger.py`)

All log messages flow through a single `asyncio.Queue`:

```python
# Any module:
from agent.ui.logger import log_queue

await log_queue.put(LogEvent(
    level="INFO",
    module="arxiv",
    message="Fetched 12 papers in 0.8s",
    timestamp=datetime.now(),
    data={"count": 12, "latency_ms": 800}
))
```

The Textual app's `on_mount` starts a background worker that drains the queue and renders events into the `RichLog` panel. No `logging.Handler` or thread-safety issues — pure async.

---

## 7. Data Models

All models live in `src/agent/models/` and are Pydantic v2 `BaseModel` subclasses.

```python
# models/query.py
class UserQuery(BaseModel):
    raw: str
    created_at: datetime = Field(default_factory=datetime.now)

class RouterDecision(BaseModel):
    query: str
    mode: Literal["academic", "general", "hybrid"]
    sub_queries: list[str]
    academic_weight: float
    general_weight: float
    explanation: str

# models/result.py
class RawResult(BaseModel):
    id: str               # sha256(url)[:16]
    title: str
    url: str
    snippet: str
    source: str
    published_at: datetime | None
    authors: list[str] = []
    categories: list[str] = []
    raw_html: str | None = None
    fetch_latency_ms: float = 0.0

class ExtractedChunk(BaseModel):
    chunk_id: str
    source_id: str
    url: str
    title: str
    content_markdown: str
    word_count: int
    extractor_used: str
    extraction_latency_ms: float
    metadata: dict = {}

class ScoredChunk(ExtractedChunk):
    semantic_score: float
    freshness_score: float
    authority_score: float
    final_score: float
    rank: int

# models/report.py
class Citation(BaseModel):
    index: int
    title: str
    source: str
    url: str
    authors: list[str]
    published_at: datetime | None

class ResearchReport(BaseModel):
    query: str
    mode: str
    generated_at: datetime
    markdown: str
    citations: list[Citation]
    stats: dict               # latencies, source counts, token usage
```

---

## 8. Async Execution Strategy

### Concurrency Model

```python
# pipeline.py
async def run(self, query: UserQuery) -> ResearchReport:

    # 1. Route (fast)
    decision = await self.router.classify(query.raw)

    # 2. Retrieve from all sources concurrently
    retriever_tasks = [
        r.fetch(decision.sub_queries, max_results=cfg.max_results_per_source)
        for r in self._active_retrievers(decision)
    ]
    results_nested = await asyncio.gather(*retriever_tasks, return_exceptions=True)
    # Flatten and discard exceptions (log them)
    raw_results = self._flatten_and_deduplicate(results_nested)

    # 3. Check cache; build extraction tasks only for cache misses
    cache_hits, cache_misses = await self.cache.partition(raw_results)
    extract_tasks = [
        self.extractor.extract(r.url)
        for r in cache_misses
    ]
    # Limit concurrency to avoid overwhelming target servers
    sem = asyncio.Semaphore(cfg.extraction_concurrency)  # default: 10
    async def extract_with_sem(r):
        async with sem:
            return await self.extractor.extract(r.url)

    extracted = await asyncio.gather(
        *[extract_with_sem(r) for r in cache_misses],
        return_exceptions=True
    )

    # 4. Merge cache hits + fresh extractions
    all_chunks = cache_hits + [c for c in extracted if isinstance(c, ExtractedChunk)]

    # 5. Rerank (CPU/MPS, batched)
    scored = await asyncio.to_thread(self.reranker.rank, all_chunks, query.raw)
    top_k = scored[:cfg.synthesis_top_k]   # default: 20

    # 6. Synthesise (streaming)
    report = await self.synthesizer.generate(query, decision, top_k)
    await self.cache.set_report(query, report)
    return report
```

### Rate Limiting & Back-off

Each retriever has a `RetryConfig`:
```python
class RetryConfig(BaseModel):
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True
    retry_on: list[int] = [429, 500, 502, 503]
```

Implemented as a decorator `@with_retry(config)` that wraps `async` functions.

### Timeout Strategy

```toml
[timeouts]
retriever_s   = 15    # per retriever fetch
extractor_s   = 10    # per URL extraction
llm_stream_s  = 120   # total synthesis
```

`asyncio.wait_for()` wraps every external call.

---

## 9. Logging Architecture

### Log Event Schema

```python
@dataclass
class LogEvent:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    module: str         # "arxiv", "extractor", "reranker", "llm", ...
    message: str
    timestamp: datetime
    data: dict = field(default_factory=dict)   # structured extras
    duration_ms: float | None = None
```

### Queue Flow

```
Module coroutine
    └── await log_queue.put(LogEvent(...))
            └── asyncio.Queue (unbounded, non-blocking put)
                    └── LogDrainer worker (Textual background task)
                            ├── RichLog panel  (live TUI display)
                            └── FileHandler    (logs/agent_YYYYMMDD.log)
```

### Why a queue?

- Log writes from concurrent coroutines never block each other
- The Textual render loop is never interrupted by log I/O
- Log persistence happens in a single background coroutine — no thread locks

### Log Level Colours (Rich)

| Level | Colour | Panel prefix |
|---|---|---|
| DEBUG | `dim white` | `·` |
| INFO | `bright_cyan` | `ℹ` |
| WARNING | `yellow` | `⚠` |
| ERROR | `bold red` | `✖` |
| SUCCESS | `bright_green` | `✔` |

---

## 10. pyproject.toml

```toml
[project]
name = "ai-research-agent"
version = "0.1.0"
description = "Async AI/ML research agent with local LLM synthesis"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }

dependencies = [
    # Core
    "typer>=0.12",
    "httpx>=0.27",
    "pydantic>=2.7",

    # Retrieval
    "arxiv>=2.1",
    "semanticscholar>=0.8",
    "wikipedia-api>=0.7",
    "duckduckgo-search>=6.2",
    "feedparser>=6.0",

    # Extraction
    "trafilatura>=1.9",
    "readability-lxml>=0.8",
    "markdownify>=0.12",

    # Reranking & Embeddings
    "sentence-transformers>=3.0",
    "torch>=2.3",           # MPS backend for Apple Silicon

    # LLM
    "llama-cpp-python>=0.2.77",   # install with CMAKE_ARGS="-DLLAMA_METAL=on"
    "openai>=1.30",               # optional remote fallback
    "tiktoken>=0.7",

    # Cache
    "aiosqlite>=0.20",
    "diskcache>=5.6",

    # UI
    "rich>=13.7",
    "textual>=0.61",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.4",
    "mypy>=1.10",
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "pytest-httpx>=0.30",
]

[project.scripts]
agent = "agent.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agent"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ASYNC"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### uv Setup Commands

```bash
# Create venv and install
uv venv --python 3.12
source .venv/bin/activate

# Install llama-cpp-python with Metal support FIRST (special build)
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install llama-cpp-python --no-binary llama-cpp-python

# Install all other deps
uv sync

# Run
agent run "latest vision language model papers"
```

---

## 11. config.toml

```toml
# config.example.toml — Copy to config.toml and fill in values.
# All keys listed here; sensitive values can be set as environment variables instead.

[llm]
backend          = "local"                  # "local" | "remote"
base_url         = "http://localhost:8000/v1"
model            = "mistral-nemo-12b"
temperature      = 0.3
max_tokens       = 2048
response_buffer  = 1024                     # tokens reserved for response

[llm.remote]
base_url         = "https://api.openai.com/v1"
model            = "gpt-4o-mini"

[embeddings]
backend          = "local"
base_url         = "http://localhost:8001/v1"
model            = "nomic-embed-text-v1.5"

[retrievers]
max_results_per_source = 15
enabled = [
    "arxiv",
    "semantic_scholar",
    "wikipedia",
    "hackernews",
    "rss",
    "duckduckgo",
]

[retrievers.rss_feeds]
urls = [
    "https://rss.arxiv.org/rss/cs.AI",
    "https://rss.arxiv.org/rss/cs.LG",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://venturebeat.com/ai/feed/",
    "https://huggingface.co/blog/feed.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://openai.com/blog/rss.xml",
]

[extractors]
primary          = "trafilatura"
jina_base_url    = "https://r.jina.ai"
chunk_size_words = 400
chunk_overlap    = 50
extraction_concurrency = 10

[reranker]
backend          = "remote"                 # "local" (HuggingFace) | "remote" (API)
base_url         = "http://localhost:8002"
model            = "jina-reranker-v2-base-multilingual-Q4_K_M.gguf"
top_k            = 20

[reranker.weights]
semantic   = 0.55
freshness  = 0.20
authority  = 0.15
length     = 0.10

[cache]
enabled          = true
db_path          = "./cache/agent.db"
chunk_ttl_hours  = 24
report_ttl_hours = 6

[timeouts]
retriever_s      = 15
extractor_s      = 10
llm_stream_s     = 120

[output]
reports_dir      = "./reports"
default_format   = "markdown"

[ui]
theme            = "dark"
log_level        = "INFO"
stream_report    = true

# ─── API Keys ──────────────────────────────────────────────────────────────────
# Prefer setting these as environment variables rather than storing in this file.
# Variable names match the keys below, uppercased (e.g. SEMANTIC_SCHOLAR_API_KEY).

[api_keys]
semantic_scholar_api_key = ""
openai_api_key           = ""
jina_api_key             = ""
```

---

## 12. Error Handling & Resilience

### Principle: **Partial success is success**

- If 2 of 6 retrievers fail, the agent continues with the 4 that succeeded
- If extraction fails for a URL, the snippet from the retriever is used instead
- If the primary extractor fails, the chain tries the next one automatically
- LLM synthesis failure: surface the error with collected source URLs so the user can investigate manually

### Circuit Breaker (per retriever)

```python
class CircuitBreaker:
    state: Literal["closed", "open", "half_open"]
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0

    # closed  → normal operation
    # open    → requests rejected immediately (too many failures)
    # half_open → one trial request to test recovery
```

### Exception Taxonomy

| Exception | Action |
|---|---|
| `httpx.TimeoutException` | Retry with back-off |
| `httpx.HTTPStatusError` (429) | Retry after `Retry-After` header |
| `httpx.HTTPStatusError` (5xx) | Retry up to `max_retries` |
| `httpx.HTTPStatusError` (4xx) | Log error, skip this result |
| `ExtractionError` | Try next extractor in chain |
| `LLMContextOverflowError` | Trim context and retry once |
| `LLMError` (hard failure) | Raise to pipeline; surface in UI |

---

## 13. Performance Targets

| Stage | Target Latency | Notes |
|---|---|---|
| Query routing | < 50 ms | Heuristic path; LLM path < 3 s |
| Retrieval (all sources) | < 8 s | All concurrent; slowest wins |
| Extraction (20 URLs) | < 12 s | Semaphore(10); cache helps |
| Reranking (20 chunks) | < 2 s | CPU cross-encoder |
| LLM synthesis | 30–90 s | Depends on model; streamed |
| **Total (cold)** | **< 2 min** | Typical query, no cache |
| **Total (warm)** | **< 45 s** | With cache hits on extraction |

---

## 14. Extension Points

The architecture is designed to make the following extensions trivial:

| Extension | Where to add |
|---|---|
| New data source | Add class in `retrievers/`, register in `config.toml` enabled list |
| New extractor | Add class in `extractors/`, update priority chain in `config.toml` |
| New reranker model | Subclass `BaseReranker` in `reranker/` |
| New LLM backend | Subclass `BaseLLM` in `llm/` |
| Vector store / RAG | Add `vectorstore/` module; feed scored chunks in; replace or augment cross-encoder reranking |
| Web UI (FastAPI) | Add `api/` module; pipeline is already async-compatible |
| Scheduled runs | Wrap `pipeline.run()` in `asyncio` cron; email/Slack report delivery |
| Multi-query synthesis | Run pipeline N times; add a `merger.py` synthesis step |

---

*End of Design Document*
