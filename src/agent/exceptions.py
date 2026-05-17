class ExtractionError(Exception):
    """Raised when content extraction fails for a URL."""


class LLMError(Exception):
    """Raised when the LLM backend encounters a hard failure."""


class LLMContextError(LLMError):
    """Raised when the context window is exceeded and truncation fails."""
