"""
Phase 5a — Paper Summary with chunk-level citations.

Takes a StructuredDocument (with populated chunks from Phase 4) and calls an
LLM to produce a structured summary where every bullet is traceable back to
the source chunks that support it.

Output format
-------------
The LLM is instructed to produce markdown with four fixed sections:

    ## One-liner
    ## Problem & Method
    ## Experiments & Results
    ## Limitations

Each bullet in the last three sections must end with one or more citation tags:

    - <claim> [chunk_id: abc123def456]

The parser extracts these tags, strips them from the display text, and
populates SummaryBullet.chunk_ids so callers can link back to source passages.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from papersketch_backend.document.models import Chunk, StructuredDocument
from papersketch_backend.llm.base import LLMUsage
from papersketch_backend.llm.registry import resolve


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SummaryBullet:
    """
    One bullet in the paper summary.

    text:      Display text with chunk citation tags removed.
    chunk_ids: Ordered list of chunk_ids that the LLM cited for this bullet.
    """
    text: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class PaperSummary:
    """
    Structured paper summary produced by one LLM call.

    one_liner:               Single sentence capturing the core contribution.
    problem_and_method:      Bullets about problem formulation and approach.
    experiments_and_results: Bullets about experiments and key metrics.
    limitations:             Bullets about limitations and assumptions.
    model_info:              Provider:model string (e.g. "openai:gpt-4o-mini").
    llm_ms:                  Wall-clock LLM latency in milliseconds.
    usage:                   Normalized token usage from the provider.
    """
    one_liner: str
    problem_and_method: list[SummaryBullet]
    experiments_and_results: list[SummaryBullet]
    limitations: list[SummaryBullet]
    model_info: str
    llm_ms: int
    usage: LLMUsage


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Matches "[chunk_id: abc123def456]" — 12 lowercase hex chars.
_CHUNK_ID_RE = re.compile(r"\[chunk_id:\s*([0-9a-f]{12})\]", re.IGNORECASE)

_PROMPT_TEMPLATE = """\
Summarize the academic paper whose content is provided below.

RULES:
- Every bullet MUST end with at least one citation tag: [chunk_id: XXXXXXXXXXXX]
- chunk_ids are exactly 12 lowercase hex characters — copy them verbatim from the content.
- Do NOT invent chunk_ids. Only cite ids that appear in the provided content.
- Be specific: include dataset names, metric values, and model names where present.
- Do not add opinions or inferred claims.

OUTPUT FORMAT — use exactly these section headers, no others:

## One-liner
<one sentence describing the core contribution of the paper>

## Problem & Method
- <claim about the problem or method> [chunk_id: XXXXXXXXXXXX]
- ...  (5–10 bullets)

## Experiments & Results
- <claim about an experiment or result, include numbers> [chunk_id: XXXXXXXXXXXX]
- ...  (5–10 bullets)

## Limitations
- <limitation or assumption> [chunk_id: XXXXXXXXXXXX]
- ...  (2–5 bullets)

--- PAPER CONTENT ---
{content}
"""


def _format_chunks_for_prompt(chunks: list[Chunk], max_chars: int) -> str:
    """
    Render body chunks grouped by section path for inclusion in the prompt.

    Caption chunks are skipped (captions are short and context-poor for
    synthesis tasks).  The output looks like:

        ### Introduction

        [chunk_id: abc123def456]
        Text of the first chunk.

        [chunk_id: 789abcdef012]
        Text of the second chunk.

        ### Method > Architecture
        ...

    Stops adding chunks once *max_chars* is reached.
    """
    lines: list[str] = []
    char_count = 0
    current_path: tuple[str, ...] = ()

    for chunk in chunks:
        if chunk.chunk_type == "caption":
            continue

        path = tuple(chunk.section_path)
        if path != current_path:
            header = " > ".join(chunk.section_path) if chunk.section_path else "Preamble"
            lines.append(f"\n### {header}\n")
            current_path = path

        piece = f"[chunk_id: {chunk.chunk_id}]\n{chunk.text}\n"
        if char_count + len(piece) > max_chars:
            lines.append("\n... (content truncated for length) ...\n")
            break
        lines.append(piece)
        char_count += len(piece)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Maps the H2 header text the LLM should produce → our internal key.
_SECTION_HEADERS: dict[str, str] = {
    "one-liner":               "one_liner",
    "one liner":               "one_liner",
    "problem & method":        "problem_method",
    "problem and method":      "problem_method",
    "experiments & results":   "experiments",
    "experiments and results": "experiments",
    "limitations":             "limitations",
}


def _parse_bullets(block: str) -> list[SummaryBullet]:
    """Parse bullet lines from a section block into SummaryBullet objects."""
    bullets: list[SummaryBullet] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        chunk_ids = _CHUNK_ID_RE.findall(body)
        clean = _CHUNK_ID_RE.sub("", body).strip(" .,;")
        if clean:
            bullets.append(SummaryBullet(text=clean, chunk_ids=chunk_ids))
    return bullets


def _parse_summary_response(
    text: str,
) -> tuple[str, list[SummaryBullet], list[SummaryBullet], list[SummaryBullet]]:
    """
    Parse the LLM's markdown output into structured components.

    Returns:
        (one_liner, problem_and_method, experiments_and_results, limitations)

    Parsing is lenient: unrecognised headers are ignored, and missing sections
    produce empty lists so the caller always receives a valid PaperSummary.
    """
    sections: dict[str, list[str]] = {}
    current_key: Optional[str] = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = current_lines
            normalized = line[3:].strip().lower()
            current_key = _SECTION_HEADERS.get(normalized)
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = current_lines

    def _join(key: str) -> str:
        return "\n".join(sections.get(key, []))

    one_liner_block = _join("one_liner").strip()
    # Strip leading bullet if the LLM added one
    one_liner = one_liner_block.lstrip("- ").split("\n")[0].strip()

    problem_method   = _parse_bullets(_join("problem_method"))
    experiments      = _parse_bullets(_join("experiments"))
    limitations      = _parse_bullets(_join("limitations"))

    return one_liner, problem_method, experiments, limitations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(
    doc: StructuredDocument,
    model_spec: str = "openai:gpt-4o-mini",
    max_context_chars: int = 60_000,
) -> PaperSummary:
    """
    Generate a structured paper summary with chunk-level citations.

    The full extraction pipeline (Phases 1–4) must have been run so that
    ``doc.chunks`` is populated.  The chunks are formatted into a prompt,
    sent to the specified LLM, and the response is parsed into a
    ``PaperSummary`` dataclass.

    Args:
        doc:               StructuredDocument with populated chunks.
        model_spec:        Provider:model string, e.g. "openai:gpt-4o-mini".
        max_context_chars: Maximum characters of chunk content to include in
                           the prompt.  Larger values give the LLM more
                           context but cost more tokens.

    Returns:
        PaperSummary with one_liner, three bullet lists, timing, and usage.
    """
    client, provider, model_name = resolve(model_spec)

    content = _format_chunks_for_prompt(doc.chunks, max_context_chars)
    prompt = _PROMPT_TEMPLATE.format(content=content)

    t0 = time.perf_counter()
    resp = client.summarize(prompt=prompt, model=model_name)
    llm_ms = int((time.perf_counter() - t0) * 1000)

    one_liner, problem_method, experiments, limitations = _parse_summary_response(resp.text)

    return PaperSummary(
        one_liner=one_liner,
        problem_and_method=problem_method,
        experiments_and_results=experiments,
        limitations=limitations,
        model_info=f"{provider}:{model_name}",
        llm_ms=llm_ms,
        usage=resp.usage,
    )
