# AI Research Agent

A fully async CLI-based AI/ML Research Agent that accepts natural language queries, fetches results from multiple sources, extracts content, reranks by relevance, and synthesises a structured Markdown report using a local or remote LLM.

## Quick Start

```bash
# Setup (see SETUP.md for detailed instructions)
uv venv --python 3.12
source .venv/bin/activate
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install llama-cpp-python --no-binary llama-cpp-python
uv sync
cp config.example.toml config.toml
# Edit config.toml: set model_path to your GGUF file

# Run
agent run "latest developments in vision language models"
```

## Usage

```bash
# Academic research
agent run "transformer architecture papers 2025" --mode academic

# General news
agent run "AI startup news this week" --mode general --no-cache

# Hybrid (both academic and general)
agent run "latest developments in multimodal LLMs" --top-k 25

# Export JSON alongside Markdown
agent run "RLHF techniques comparison" --export-json

# Interactive shell
agent shell
```

## Commands

| Command | Description |
|---------|-------------|
| `agent run <query>` | Run the full research pipeline |
| `agent shell` | Interactive REPL with `/help` |
| `agent health` | Run health checks on all components |
| `agent config-show` | Print resolved configuration (keys masked) |
| `agent cache-clear` | Clear all cached data |
| `agent cache-stats` | Show cache statistics |

## Documentation

- **[SETUP.md](./SETUP.md)** — Prerequisites, environment setup, model download, configuration, troubleshooting
- **[code_workflow.md](./code_workflow.md)** — Complete code architecture: entry points, routing, retrieval, extraction, reranking, synthesis, logging, data models, error handling

## Configuration

All configuration lives in `config.toml` (copy from `config.example.toml`). See `agent config-show` for the resolved config. API keys can also be set as environment variables (e.g. `OPENAI_API_KEY`).

Key sections:
- `[llm]` — Backend (`local`/`remote`), model path, context window, temperature
- `[retrievers]` — Enabled sources, max results per source, RSS feed URLs
- `[reranker]` — Top-K, weights for semantic/freshness/authority/length scoring
- `[cache]` — SQLite cache TTLs for chunks and reports
- `[timeouts]` — Per-retriever timeout (`retriever_s`), extraction, LLM streaming

## Reports

Generated reports are saved as Markdown (and optionally JSON) to the `./reports/` directory. Logs are written to `./logs/agent_YYYYMMDD.log` as JSON Lines.

## Logging

All pipeline stages log detailed information to stderr and the log file. Full LLM prompts and responses are captured. See [code_workflow.md](./code_workflow.md#6-logging-system) for the complete logging specification.
