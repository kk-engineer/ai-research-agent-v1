import asyncio
import hashlib
import time

import httpx
import trafilatura

from agent.extractors.base import BaseExtractor
from agent.logger import fmt_ms, log_info, log_warning
from agent.models.result import ExtractedChunk


class TrafilaturaExtractor(BaseExtractor):
    name = "trafilatura"

    async def extract(self, url: str, raw_html: str | None = None) -> ExtractedChunk | None:
        t0 = time.perf_counter()

        html = raw_html
        if html is None:
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    html = resp.text
            except Exception as e:
                await log_warning("trafilatura", f"HTTP error for {url[:80]}: {e}")
                return None

        def _extract() -> str | None:
            return trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
            )

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _extract)

        if not text or not text.strip():
            await log_warning("trafilatura", f"No content extracted from {url[:80]}")
            return None

        word_count = len(text.split())
        latency = (time.perf_counter() - t0) * 1000
        await log_info(
            "trafilatura",
            f"Extracted {word_count} words from {url[:80]} in {fmt_ms(latency)}",
        )

        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        return ExtractedChunk(
            chunk_id=hashlib.sha256(url.encode()).hexdigest()[:16],
            source_id=source_id,
            url=url,
            title=url.split("/")[-1].replace("-", " ").title() if "/" in url else url,
            content_markdown=text,
            word_count=word_count,
            extractor_used=self.name,
            extraction_latency_ms=latency,
        )
