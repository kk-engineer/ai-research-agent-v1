import os
from pathlib import Path

_CACHE_DIR = Path("./local_models")
_HUB_DIR = _CACHE_DIR / "hub"

_configured = False


def ensure_hf_cache(model_names: list[str] | None = None) -> None:
    global _configured
    if _configured:
        return

    os.environ.setdefault("HF_HOME", str(_CACHE_DIR))

    if model_names is not None:
        all_cached = all(
            (_HUB_DIR / _to_hf_folder(name)).exists()
            for name in model_names
        )
        if all_cached:
            os.environ["HF_HUB_OFFLINE"] = "1"
            return

    os.environ.pop("HF_HUB_OFFLINE", None)


def _to_hf_folder(model_name: str) -> str:
    return f"models--{model_name.replace('/', '--')}"
