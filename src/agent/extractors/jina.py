import hashlib
import time

import httpx

from agent.extractors.base import BaseExtractor
from agent.models.result import ExtractedChunk
from agent.logger import fmt_ms, log_info, log_warning


class JinaExtractor(BaseExtractor):
    name = "jina"
    BASE = "https://r.jina.ai/"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    async def extract(self, url: str, raw_html: str | None = None) -> ExtractedChunk | None:
        t0 = time.perf_counter()

        try:
            headers = {
                "Accept": "text/markdown",
                "X-Return-Format": "markdown",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                jina_url = f"{self.BASE}{url}"
                resp = await client.get(jina_url, headers=headers)
                resp.raise_for_status()
                text = resp.text
        except Exception as e:
            await log_warning("jina", f"Jina error for {url[:80]}: {e}")
            return None

        if not text or not text.strip():
            return None

        word_count = len(text.split())
        latency = (time.perf_counter() - t0) * 1000
        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        await log_info("jina", f"Extracted {word_count} words from {url[:80]} in {fmt_ms(latency)}")

        return ExtractedChunk(
            chunk_id=source_id,
            source_id=source_id,
            url=url,
            title=url.split("/")[-1].replace("-", " ").title() if "/" in url else url,
            content_markdown=text,
            word_count=word_count,
            extractor_used=self.name,
            extraction_latency_ms=latency,
        )
