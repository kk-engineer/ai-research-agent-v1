import sys

import httpx

from agent.config import AppConfig
from agent.embeddings import create_embeddings
from agent.llm import create_llm
from agent.logger import log_error, log_info, log_success, log_warning
from agent.reranker import create_reranker


async def validate_connectivity(config: AppConfig) -> None:
    """Validate LLM, embeddings, and reranker at startup.

    Exits with code 1 if any component is unreachable.
    """
    await log_info("startup", "Validating component connectivity...")

    await log_info("startup", "Checking LLM...")
    try:
        llm = create_llm(config)
        response = await llm.complete("Respond with exactly 'OK'.")
        if "ok" in response.lower():
            await log_success("startup", f"LLM ({config.llm.mode}) verified")
        else:
            await log_warning("startup", f"LLM responded unexpectedly: {response.strip()}")
    except SystemExit:
        await log_error("startup", "LLM verification failed")
        raise
    except httpx.ConnectError:
        await log_error("startup", f"LLM unreachable: cannot connect to {config.llm.base_url}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        await log_error(
            "startup",
            f"LLM unreachable: {config.llm.base_url} returned HTTP {e.response.status_code}",
        )
        sys.exit(1)
    except Exception as e:
        await log_error("startup", f"LLM unreachable: {e}")
        sys.exit(1)

    await log_info("startup", "Checking embeddings...")
    try:
        embedder = create_embeddings(config)
        await embedder.embed("connectivity verification")
        await log_success(
            "startup",
            f"Embeddings ({config.embeddings.mode}) verified",
        )
    except SystemExit:
        await log_error("startup", "Embeddings verification failed")
        raise
    except Exception as e:
        await log_error("startup", f"Embeddings check failed: {e}")
        sys.exit(1)

    await log_info("startup", "Checking reranker...")
    try:
        reranker = create_reranker(config)
        if config.reranker.mode == "cloud":
            await reranker.rank([], "connectivity verification", 0)
        await log_success("startup", f"Reranker ({config.reranker.mode}) verified")
    except SystemExit:
        await log_error("startup", "Reranker verification failed")
        raise
    except Exception as e:
        await log_error("startup", f"Reranker check failed: {e}")
        sys.exit(1)

    await log_success("startup", "All components verified")
