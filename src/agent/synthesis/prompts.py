from datetime import datetime

from agent.models.result import ScoredChunk


def build_system_prompt() -> str:
    now = datetime.now()
    today = now.strftime("%B %d, %Y")
    current_year = now.year
    return (
        "You are an expert AI/ML research analyst. Your task is to synthesise "
        "information from multiple sources into a clear, well-structured "
        "Markdown research report.\n\n"
        "## CRITICAL DATE RULE\n"
        f"The current date is {today}. "
        "When the user query uses relative time periods (e.g. "
        "\"last 4 weeks\", \"recent\", \"past month\", \"latest\", "
        "\"this year\"), you MUST calculate the actual date range "
        f"relative to {today}. "
        "For example, \"last 4 weeks\" means from "
        f"{now.strftime('%B')} {now.day - 28}, {current_year} to {today}. "
        "\"Past year\" means from "
        f"{now.strftime('%B')} {now.day}, {current_year - 1} to {today}. "
        "DO NOT use dates from your training data. "
        "DO NOT reference dates earlier than 2026 for news queries. "
        "If the source material dates are older than the query's "
        "intended timeframe, note this discrepancy and synthesize "
        "based on what is available, but DO NOT fabricate more "
        "recent information.\n\n"
        "Rules:\n"
        "1. Output only valid Markdown.\n"
        "2. Use inline citations like [1], [2] referencing the Sources block "
        "at the end.\n"
        "3. Be clear, technical, and accessible.\n"
        "4. Only cite information that is present in the provided context. "
        "Do NOT fabricate facts or citations.\n"
        "5. If the context does not cover a requested section, note "
        '"Not covered by available sources."\n'
        "6. Organise the report with clear headings and subheadings.\n\n"
        "Report structure:\n"
        "## Executive Summary\n"
        "[2-3 most important points TL;DR covering the most important findings]\n\n"
        "## Key Findings\n"
        "[Detailed synthesis organised by subtopic]\n\n"
        "## Recent Developments\n"
        "[News, announcements, and timely updates]\n\n"
        "## Academic Highlights\n"
        "[Paper summaries with key results, only if academic sources are "
        "present]\n\n"
        "## Limitations & Gaps\n"
        "[What the sources do not cover; areas of uncertainty]\n\n"
        "## References\n"
        "[List numbered citations (each in new line) in format: [N] Title: URL \n]"
    )


def build_synthesis_prompt(
    query: str,
    mode: str,
    sub_queries: list[str],
    chunks: list[ScoredChunk],
    max_chunks: int,
) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    source_blocks: list[str] = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        block = (
            f"=== SOURCE [{i}] ===\n"
            f"Title: {chunk.title}\n"
            f"URL: {chunk.url}\n"
            f"Source: {chunk.metadata.get('source', 'unknown')}\n"
            f"Date: {chunk.metadata.get('published_at', 'unknown')}\n"
            f"---\n"
            f"{chunk.content_markdown[:2000]}\n"
            f"{'='*40}\n"
        )
        source_blocks.append(block)

    context_block = "\n\n".join(source_blocks)

    prompt = (
        f"Current date: {today}\n"
        f"Research Query: {query}\n"
        f"Mode: {mode}\n"
        f"Sub-queries: {', '.join(sub_queries)}\n\n"
        f"Below are the source materials retrieved for this research. "
        f"Synthesise them into a comprehensive Markdown report following "
        f"the required structure. "
        f"Remember: if the query asks about relative time periods like "
        f"\"last 4 weeks\", calculate them from {today}.\n\n"
        f"{context_block}\n\n"
        f"Generate the report now:"
    )
    return prompt


def build_classification_prompt(query: str) -> str:
    today = datetime.now().strftime("%B %d, %Y")
    return (
        'Classify the following research query into one of three modes: '
        '"academic", "general", or "hybrid".\n\n'
        'Academic queries focus on papers, research, surveys, arXiv, '
        'journals, conferences.\n'
        'General queries focus on news, announcements, products, blogs, '
        'tutorials.\n'
        'Hybrid queries contain elements of both.\n\n'
        f'Current date: {today}\n\n'
        'Respond with ONLY a valid JSON object:\n'
        '{{\n'
        '    "mode": "academic" | "general" | "hybrid",\n'
        '    "academic_weight": <float 0.0-1.0>,\n'
        '    "general_weight": <float 0.0-1.0>,\n'
        '    "sub_queries": ["<best sub-query 1>", "<best sub-query 2>", ...],\n'
        '    "explanation": "<brief reason>"\n'
        '}}\n\n'
        f'Query: {query}'
    )
