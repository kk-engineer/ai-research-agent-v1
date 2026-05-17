# Setup Guide

## 1. Prerequisites

```bash
# Install uv (Python package manager)
brew install uv

# Ensure Python 3.12+ is available
python3.12 --version
```

## 2. Clone & Create Virtual Environment

```bash
cd ai-research-agent
uv venv --python 3.12
source .venv/bin/activate
```

## 3. Install llama-cpp-python with Metal Support

**This MUST be done before `uv sync`.**

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" \
  FORCE_CMAKE=1 \
  uv pip install llama-cpp-python --no-binary llama-cpp-python
```

If the build fails, ensure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

## 4. Install All Other Dependencies

```bash
uv sync
```

For development extras:
```bash
uv sync --group dev
```

## 5. Download a GGUF Model

```bash
mkdir -p models
```

Download a model from HuggingFace. Recommended models for Apple Silicon:

| Model | Size | RAM Required | Quality |
|---|---|---|---|
| `Llama-3.2-3B-Instruct.Q8_0.gguf` | ~3 GB | 8 GB | Fast, good |
| `Mistral-7B-Instruct-v0.3.Q5_K_M.gguf` | ~5 GB | 16 GB | Better |
| `Llama-3.1-8B-Instruct.Q4_K_M.gguf` | ~4.7 GB | 16 GB | Best |

Example download (using `curl` or `wget`):
```bash
# Replace URL with actual HuggingFace download link
curl -L -o models/Llama-3.2-3B-Instruct.Q8_0.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct.Q8_0.gguf"
```

Or use `huggingface-cli`:
```bash
pip install huggingface-hub
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct.Q8_0.gguf" \
  --local-dir ./models
```

## 6. Configure

```bash
cp config.example.toml config.toml
```

Edit `config.toml`:
- Set `model_path` to point to your downloaded GGUF file
- Optionally set API keys for enhanced functionality
  - `[api_keys] openai_api_key` — for remote LLM fallback
  - `[api_keys] semantic_scholar_api_key` — for higher rate limits
  - `[api_keys] jina_api_key` — for Jina Reader extraction

## 7. Run

```bash
# Basic query
agent run "latest developments in vision language models"

# Academic mode (focus on papers)
agent run "transformer architecture papers 2025" --mode academic

# General mode (news, blogs)
agent run "AI startup news this week" --mode general --no-cache

# Hybrid mode (default when ambiguous)
agent run "RLHF advances and news" --top-k 25

# Export JSON alongside the Markdown report
agent run "diffusion models comparison" --export-json

# Use a remote LLM (set backend=remote in config.toml and provide API key)
agent run "latest GPT-4 research" --config config.toml
```

## 8. Verify Installation

```bash
# Run health checks
agent health

# View cache statistics
agent cache-stats

# View configuration
agent config-show
```

## Troubleshooting

### llama-cpp-python build fails
Ensure Xcode CLT is installed and try with verbose flags:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 \
  uv pip install llama-cpp-python --no-binary llama-cpp-python -v
```

### Model not found
Double-check `model_path` in `config.toml` is an absolute or relative (to project root) path to the GGUF file.

### No results from Semantic Scholar
The public API has rate limits. Set `SEMANTIC_SCHOLAR_API_KEY` environment variable or add it to `config.toml`.

### Textual TUI not rendering properly
Ensure your terminal supports 24-bit colour and is at least 80x24 characters.

### Apple Silicon MPS errors
Set `n_gpu_layers = 0` in `config.toml` to fall back to CPU (slower but stable).
