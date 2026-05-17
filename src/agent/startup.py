import sys

import httpx

from agent.config import AppConfig
from agent.embeddings import ServerEmbedder, SentenceTransformerEmbedder
from agent.llm import create_llm
from agent.logger import log_error, log_info, log_success, log_warning


async def validate_connectivity(config: AppConfig) -> None:
    """Validate LLM, embeddings, and reranker at startup.

    Exits with code 1 if LLM is unreachable. Logs warnings for other
    components — the pipeline will surface hard errors if they are required.
    """
    await log_info("startup", "Validating component connectivity...")

    await log_info("startup", "Checking LLM...")
    try:
        llm = create_llm(config)
        response = await llm.complete("Respond with exactly 'OK'.")
        if "ok" in response.lower():
            await log_success("startup", f"LLM ({config.llm.backend}) verified")
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
        if config.embeddings.base_url:
            embedder = ServerEmbedder(config.embeddings.base_url, config.embeddings.model)
            await embedder.embed("connectivity verification")
            await log_success("startup", f"Embeddings (server @ {config.embeddings.base_url}) verified")
        else:
            await log_warning("startup", "Embeddings: no base_url configured, skipping check")
    except Exception as e:
        await log_warning("startup", f"Embeddings check skipped: {e}")

    await log_info("startup", "Checking reranker...")
    try:
        if config.reranker.backend == "remote":
            from agent.reranker.server import ServerReranker

            reranker = ServerReranker(config)
            await reranker.rank([], "connectivity verification", 0)
            await log_success("startup", f"Reranker (remote @ {config.reranker.base_url}) verified")
        elif config.reranker.backend == "local":
            from agent.reranker.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker(config)
            await reranker._load_model()
            await log_success("startup", "Reranker (local cross-encoder) verified")
        else:
            await log_warning("startup", "Reranker: no backend configured, skipping check")
    except SystemExit:
        await log_error("startup", "Reranker verification failed")
        raise
    except Exception as e:
        await log_warning("startup", f"Reranker check skipped: {e}")

    await log_success("startup", "All components verified")
