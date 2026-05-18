from datetime import UTC, datetime
from urllib.parse import urlparse

AUTHORITY_SCORES: dict[str, float] = {
    "arxiv": 0.90,
    "semantic_scholar": 0.88,
    "wikipedia": 0.80,
    "deepmind.google": 0.78,
    "openai.com": 0.78,
    "huggingface.co": 0.75,
    "techcrunch.com": 0.60,
    "venturebeat.com": 0.58,
    "hackernews": 0.50,
    "rss": 0.55,
    "duckduckgo": 0.40,
}


def freshness_score(published_at: datetime | str | None) -> float:
    if published_at is None:
        return 0.5
    if isinstance(published_at, str):
        try:
            dt = datetime.fromisoformat(published_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            published_at = dt
        except Exception:
            return 0.5
    now = datetime.now(UTC)
    days_old = (now - published_at).days
    return 1.0 / (1.0 + days_old / 30.0)


def authority_score(source: str, url: str) -> float:
    if source in AUTHORITY_SCORES:
        return AUTHORITY_SCORES[source]

    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        for key, score in AUTHORITY_SCORES.items():
            if key in domain or domain.endswith("." + key):
                return score
    except Exception:
        pass

    return 0.45


def length_score(word_count: int) -> float:
    if word_count < 50:
        return 0.2
    return min(1.0, word_count / 300.0)
