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


class LLMProviderConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class LLMCloudConfig(BaseModel):
    timeout: int = Field(default=60, description="Per-provider timeout in seconds")
    provider_order: list[str] = Field(
        default=[
            "nvidia", "gemini", "openrouter", "huggingface",
            "deepseek", "openai", "anthropic",
        ]
    )
    nvidia: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        model="meta/llama-3.1-8b-instruct",
    ))
    gemini: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.0-flash",
    ))
    openrouter: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.0-flash-lite-preview-02-05:free",
    ))
    huggingface: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://api-inference.huggingface.co/v1",
        model="mistralai/Mistral-7B-Instruct-v0.3",
    ))
    deepseek: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    ))
    openai: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    ))
    anthropic: LLMProviderConfig = Field(default_factory=lambda: LLMProviderConfig(
        base_url="https://api.anthropic.com/v1",
        model="claude-3-haiku-20240307",
    ))


class LLMConfig(BaseModel):
    mode: str = Field(default="local", description='"local" | "cloud"')
    base_url: str = Field(default="http://localhost:8000/v1", description="LLM API base URL")
    model: str = Field(default="mistral-nemo-12b", description="LLM model name")
    n_ctx: int = Field(default=8192, description="Context window size in tokens")
    temperature: float = Field(default=0.3, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Max tokens in response")
    response_buffer: int = Field(default=1024, description="Tokens reserved for response")
    remote: LLMRemoteConfig = Field(default_factory=LLMRemoteConfig)
    cloud: LLMCloudConfig = Field(default_factory=LLMCloudConfig)


class EmbeddingCloudConfig(BaseModel):
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = ""
    model: str = Field(default="text-embedding-3-small")


class EmbeddingsConfig(BaseModel):
    mode: str = Field(default="local", description='"local" | "cloud"')
    base_url: str = Field(
        default="http://localhost:8001/v1", description="Embedding server base URL"
    )
    model: str = Field(default="nomic-embed-text-v1.5", description="Embedding model name")
    cloud: EmbeddingCloudConfig = Field(default_factory=EmbeddingCloudConfig)


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
            "https://www.reddit.com/r/LocalLLaMA/hot/.rss",
            "https://www.reddit.com/r/MachineLearning/top/.rss?t=week",
            "https://www.reddit.com/r/OpenAI/hot/.rss",
            "https://www.reddit.com/r/LanguageTechnology/hot/.rss",
        ]
    )


class RedditConfig(BaseModel):
    subreddits: list[str] = Field(
        default=["LocalLLaMA", "MachineLearning", "artificial", "OpenAI"]
    )
    feed_type: str = Field(default="hot")


class RetrieverConfig(BaseModel):
    max_results_per_source: int = Field(default=15, description="Max results per retriever")
    enabled: list[str] = Field(
        default=["arxiv", "semantic_scholar", "wikipedia", "hackernews", "rss", "duckduckgo"]
    )
    rss_feeds: RSSFeedsConfig = Field(default_factory=RSSFeedsConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)


class ExtractorConfig(BaseModel):
    extraction_concurrency: int = Field(default=10, description="Max concurrent extractions")


class RerankerWeightsConfig(BaseModel):
    semantic: float = Field(default=0.55, ge=0.0, le=1.0)
    freshness: float = Field(default=0.20, ge=0.0, le=1.0)
    authority: float = Field(default=0.15, ge=0.0, le=1.0)
    length: float = Field(default=0.10, ge=0.0, le=1.0)


class RerankerCloudConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class RerankerConfig(BaseModel):
    mode: str = Field(default="local", description='"local" | "cloud"')
    top_k: int = Field(default=20, description="Number of chunks to keep after reranking")
    base_url: str = Field(default="", description="Reranker API base URL")
    model: str = Field(default="", description="Reranker model name or path")
    weights: RerankerWeightsConfig = Field(default_factory=RerankerWeightsConfig)
    cloud: RerankerCloudConfig = Field(default_factory=RerankerCloudConfig)


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
    nvidia_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    openrouter_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")
    deepseek_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrievers: RetrieverConfig = Field(default_factory=RetrieverConfig)
    extractors: ExtractorConfig = Field(default_factory=ExtractorConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    log: UIConfig = Field(default_factory=UIConfig)
    api_keys: APIKeyConfig = Field(default_factory=APIKeyConfig)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    env_map = {
        "OPENAI_API_KEY": ("api_keys", "openai_api_key"),
        "SEMANTIC_SCHOLAR_API_KEY": ("api_keys", "semantic_scholar_api_key"),
        "JINA_API_KEY": ("api_keys", "jina_api_key"),
        "NVIDIA_API_KEY": ("api_keys", "nvidia_api_key"),
        "GEMINI_API_KEY": ("api_keys", "gemini_api_key"),
        "OPENROUTER_API_KEY": ("api_keys", "openrouter_api_key"),
        "HUGGINGFACE_API_KEY": ("api_keys", "huggingface_api_key"),
        "DEEPSEEK_API_KEY": ("api_keys", "deepseek_api_key"),
        "ANTHROPIC_API_KEY": ("api_keys", "anthropic_api_key"),
        "LLM_MODE": ("llm", "mode"),
        "LLM_BASE_URL": ("llm", "base_url"),
        "LLM_MODEL": ("llm", "model"),
        "LLM_N_CTX": ("llm", "n_ctx"),
        "LLM_TEMPERATURE": ("llm", "temperature"),
        "LLM_MAX_TOKENS": ("llm", "max_tokens"),
        "EMBEDDINGS_MODE": ("embeddings", "mode"),
        "RERANKER_MODE": ("reranker", "mode"),
        "RETRIEVERS_MAX_RESULTS": ("retrievers", "max_results_per_source"),
        "CACHE_DB_PATH": ("cache", "db_path"),
        "CACHE_CHUNK_TTL": ("cache", "chunk_ttl_hours"),
        "CACHE_REPORT_TTL": ("cache", "report_ttl_hours"),
        "EXTRACTOR_CONCURRENCY": ("extractors", "extraction_concurrency"),
        "RERANKER_TOP_K": ("reranker", "top_k"),
        "LOG_LEVEL": ("log", "log_level"),
        "OUTPUT_REPORTS_DIR": ("output", "reports_dir"),
        "CLOUD_LLM_TIMEOUT": ("llm", "cloud", "timeout"),
    }

    for env_key, path_parts in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            section = config
            for part in path_parts[:-1]:
                section = getattr(section, part)
            field = path_parts[-1]
            current = getattr(section, field)
            if isinstance(current, bool):
                setattr(section, field, value.lower() in ("true", "1", "yes"))
            elif isinstance(current, int):
                setattr(section, field, int(value))
            elif isinstance(current, float):
                setattr(section, field, float(value))
            else:
                setattr(section, field, value)

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

    _propagate_api_keys(_config)

    for field, value in _config.api_keys.model_dump().items():
        if value:
            os.environ[field.upper()] = value

    _apply_env_overrides(_config)

    return _config


def _propagate_api_keys(config: AppConfig) -> None:
    """Populate cloud provider api_key fields from environment variables."""

    provider_env_map = {
        "nvidia": "NVIDIA_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    for provider_name, env_key in provider_env_map.items():
        provider_cfg = getattr(config.llm.cloud, provider_name, None)
        if provider_cfg and not provider_cfg.api_key:
            env_value = os.getenv(env_key)
            if env_value:
                provider_cfg.api_key = env_value

    if config.embeddings.cloud and not config.embeddings.cloud.api_key:
        config.embeddings.cloud.api_key = os.getenv("OPENAI_API_KEY", "")

    if config.reranker.cloud and not config.reranker.cloud.api_key:
        config.reranker.cloud.api_key = os.getenv("HF_API_KEY", "")


def get_config() -> AppConfig:
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
