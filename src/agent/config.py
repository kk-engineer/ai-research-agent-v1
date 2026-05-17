import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console

_console = Console()
_config: "AppConfig | None" = None


class LLMRemoteConfig(BaseModel):
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    model: str = Field(default="gpt-4o-mini", description="Remote model name")


class LLMConfig(BaseModel):
    backend: str = Field(default="local", description='"local" | "remote"')
    base_url: str = Field(
        default="http://localhost:8000/v1", description="LLM API base URL"
    )
    model: str = Field(default="mistral-nemo-12b", description="LLM model name")
    n_ctx: int = Field(default=8192, description="Context window size in tokens")
    temperature: float = Field(default=0.3, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Max tokens in response")
    response_buffer: int = Field(default=1024, description="Tokens reserved for response")
    remote: LLMRemoteConfig = Field(default_factory=LLMRemoteConfig)


class EmbeddingsConfig(BaseModel):
    backend: str = Field(default="local", description='"local" (llama.cpp server) | "sentence_transformer"')
    base_url: str = Field(default="http://localhost:8001/v1", description="Embedding server base URL")
    model: str = Field(default="nomic-embed-text-v1.5", description="Embedding model name")


class RSSFeedsConfig(BaseModel):
    urls: list[str] = Field(
        default=[
            "https://rss.arxiv.org/rss/cs.AI",
            "https://rss.arxiv.org/rss/cs.LG",
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://venturebeat.com/ai/feed/",
            "https://huggingface.co/blog/feed.xml",
            "https://deepmind.google/blog/rss.xml",
            "https://openai.com/blog/rss.xml",
        ]
    )


class RetrieverConfig(BaseModel):
    max_results_per_source: int = Field(default=15, description="Max results per retriever")
    enabled: list[str] = Field(
        default=["arxiv", "semantic_scholar", "wikipedia", "hackernews", "rss", "duckduckgo"]
    )
    rss_feeds: RSSFeedsConfig = Field(default_factory=RSSFeedsConfig)


class ExtractorConfig(BaseModel):
    extraction_concurrency: int = Field(default=10, description="Max concurrent extractions")


class RerankerWeightsConfig(BaseModel):
    semantic: float = Field(default=0.55, ge=0.0, le=1.0)
    freshness: float = Field(default=0.20, ge=0.0, le=1.0)
    authority: float = Field(default=0.15, ge=0.0, le=1.0)
    length: float = Field(default=0.10, ge=0.0, le=1.0)


class RerankerConfig(BaseModel):
    top_k: int = Field(default=20, description="Number of chunks to keep after reranking")
    backend: str = Field(default="local", description='"local" (sentence-transformers) | "remote" (API)')
    base_url: str = Field(default="", description="Reranker API base URL (for remote backend)")
    model: str = Field(default="", description="Reranker model name or path")
    weights: RerankerWeightsConfig = Field(default_factory=RerankerWeightsConfig)


class CacheConfig(BaseModel):
    enabled: bool = Field(default=True)
    db_path: str = Field(default="./cache/agent.db", description="Path to SQLite cache DB")
    chunk_ttl_hours: int = Field(default=24, description="TTL for cached chunks")
    report_ttl_hours: int = Field(default=6, description="TTL for cached reports")


class TimeoutConfig(BaseModel):
    retriever_s: int = Field(default=15, description="Per-retriever timeout in seconds")


class OutputConfig(BaseModel):
    reports_dir: str = Field(default="./reports")


class UIConfig(BaseModel):
    log_level: str = Field(default="INFO")


class APIKeyConfig(BaseModel):
    semantic_scholar_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    jina_api_key: str = Field(default="")


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrievers: RetrieverConfig = Field(default_factory=RetrieverConfig)
    extractors: ExtractorConfig = Field(default_factory=ExtractorConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    api_keys: APIKeyConfig = Field(default_factory=APIKeyConfig)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Override config values from environment variables."""
    env_map = {
        "OPENAI_API_KEY": ("api_keys", "openai_api_key"),
        "SEMANTIC_SCHOLAR_API_KEY": ("api_keys", "semantic_scholar_api_key"),
        "JINA_API_KEY": ("api_keys", "jina_api_key"),
        "LLM_BASE_URL": ("llm", "base_url"),
        "LLM_MODEL": ("llm", "model"),
        "LLM_BACKEND": ("llm", "backend"),
        "LLM_N_CTX": ("llm", "n_ctx"),
        "LLM_TEMPERATURE": ("llm", "temperature"),
        "LLM_MAX_TOKENS": ("llm", "max_tokens"),
        "EMBEDDINGS_BACKEND": ("embeddings", "backend"),
        "RETRIEVERS_MAX_RESULTS": ("retrievers", "max_results_per_source"),
        "CACHE_DB_PATH": ("cache", "db_path"),
        "CACHE_CHUNK_TTL": ("cache", "chunk_ttl_hours"),
        "CACHE_REPORT_TTL": ("cache", "report_ttl_hours"),
        "EXTRACTOR_CONCURRENCY": ("extractors", "extraction_concurrency"),
        "RERANKER_TOP_K": ("reranker", "top_k"),
        "LOG_LEVEL": ("log", "log_level"),
        "OUTPUT_REPORTS_DIR": ("output", "reports_dir"),
    }

    for env_key, (section, field) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            section_obj = getattr(config, section)
            current = getattr(section_obj, field)
            if isinstance(current, bool):
                setattr(section_obj, field, value.lower() in ("true", "1", "yes"))
            elif isinstance(current, int):
                setattr(section_obj, field, int(value))
            elif isinstance(current, float):
                setattr(section_obj, field, float(value))
            else:
                setattr(section_obj, field, value)

    return config


def load_config(path: Path) -> AppConfig:
    global _config

    if not path.exists():
        _console.print(
            f"[yellow]⚠ config.toml not found at {path}, "
            f"using config.example.toml with defaults[/yellow]"
        )
        fallback = Path("config.example.toml")
        if fallback.exists():
            path = fallback
        else:
            _config = AppConfig()
            return _config

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    _config = AppConfig.model_validate(raw)
    _config = _apply_env_overrides(_config)

    for field, value in _config.api_keys.model_dump().items():
        if value:
            os.environ[field.upper()] = value

    return _config


def get_config() -> AppConfig:
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
