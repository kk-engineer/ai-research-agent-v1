import asyncio
import hashlib
import time

import httpx
from markdownify import markdownify as md
from readability import Document

from agent.extractors.base import BaseExtractor
from agent.models.result import ExtractedChunk
from agent.logger import fmt_ms, log_info, log_warning


class ReadabilityExtractor(BaseExtractor):
    name = "readability"

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
                await log_warning("readability", f"HTTP error for {url[:80]}: {e}")
                return None

        if not html:
            return None

        def _extract() -> str:
            doc = Document(html)
            summary_html = doc.summary()
            return md(summary_html)

        loop = asyncio.get_event_loop()
        try:
            markdown_text = await loop.run_in_executor(None, _extract)
        except Exception as e:
            await log_warning("readability", f"Extraction error for {url[:80]}: {e}")
            return None

        if not markdown_text or not markdown_text.strip():
            return None

        word_count = len(markdown_text.split())
        latency = (time.perf_counter() - t0) * 1000
        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        await log_info(
            "readability",
            f"Extracted {word_count} words from {url[:80]} in {fmt_ms(latency)}",
        )

        return ExtractedChunk(
            chunk_id=source_id,
            source_id=source_id,
            url=url,
            title=url.split("/")[-1].replace("-", " ").title() if "/" in url else url,
            content_markdown=markdown_text,
            word_count=word_count,
            extractor_used=self.name,
            extraction_latency_ms=latency,
        )
