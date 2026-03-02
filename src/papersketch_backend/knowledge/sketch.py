"""
Phase 5b — Paper Sketch: navigable skeleton of a paper.

Produces a PaperSketch dataclass that contains:
  - section_map:    one-sentence LLM summary per section (navigable TOC)
  - figure_index:   every detected Figure with a short caption summary
  - table_index:    every detected Table with a short caption summary
  - key_terms:      up to 10 technical terms extracted by the LLM,
                    each linked to the first Chunk that contains it

All section summaries are obtained in a single LLM call (cheaper than one
call per section).  The LLM is asked to respond in JSON so the output is
unambiguous to parse.  A plain-text fallback handles JSON decode failures
by returning empty lists for unresolvable fields.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from papersketch_backend.document.models import (
    Chunk,
    Figure,
    Section,
    StructuredDocument,
    Table,
)
from papersketch_backend.llm.base import LLMUsage
from papersketch_backend.llm.registry import resolve


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectionSummary:
    """One-sentence summary of a single section."""
    section_id: str
    title: str
    level: int
    one_liner: str


@dataclass
class FigureEntry:
    """
    A figure or table entry in the sketch index.

    Used for both figure_index and table_index so the sketch has a uniform
    structure regardless of element type.
    """
    label: str            # e.g. "Figure 3" or "Table 2"
    caption_summary: str  # short one-sentence description
    page: int             # 0-based page where the figure/table appears
    crop_url: Optional[str] = None


@dataclass
class TermEntry:
    """A key technical term extracted from the paper."""
    term: str
    first_page: int  # 0-based page of the first chunk that contains the term
    chunk_id: str    # chunk_id of that first occurrence


@dataclass
class PaperSketch:
    """
    Navigable skeleton of a paper.

    section_map:   Flat list of SectionSummary objects in document order —
                   a lightweight TOC the UI can render without the full text.
    figure_index:  All detected figures with short captions.
    table_index:   All detected tables with short captions.
    key_terms:     Up to 10 important technical terms with provenance.
    """
    title: Optional[str]
    section_map: list[SectionSummary]
    figure_index: list[FigureEntry]
    table_index: list[FigureEntry]
    key_terms: list[TermEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a research assistant. Given the structured content of an academic paper below, \
produce a JSON response in exactly the format shown.

OUTPUT — respond with ONLY valid JSON, no markdown fences, no extra commentary:
{{
  "section_summaries": [
    {{"section_id": "s1", "one_liner": "1-2 sentence summary of this section."}}
  ],
  "figure_summaries": [
    {{"label": "Figure 1", "caption_summary": "One sentence describing what this figure shows."}}
  ],
  "key_terms": [
    {{"term": "technical term", "section_id": "s1"}}
  ]
}}

RULES:
- section_id values must be exactly as given in the content (e.g. "s1", "s2").
- Include every section listed in the content.
- figure_summaries covers both figures and tables; use the exact label string.
- key_terms: list up to 10 important technical terms or acronyms. \
  Use the section_id where the term first appears.
- If a field cannot be determined, use an empty string.

--- PAPER CONTENT ---
{content}
"""


def _format_content_for_sketch(doc: StructuredDocument, max_chars: int) -> str:
    """
    Build a compact content block for the sketch prompt.

    Layout:
      [Section s1: Introduction (level 1)]
      <first ~300 chars of body chunks in this section>

      [Figure 1] Figure 1: Comparison of baselines on GLUE.
      [Table 2]  Table 2: Ablation study results.
    """
    # Build section_id -> body text mapping from chunks
    sec_text: dict[str, list[str]] = {}
    for chunk in doc.chunks:
        if chunk.chunk_type == "caption":
            continue
        # Determine which section this chunk belongs to.
        # Chunks carry section_path (titles), not section_id.
        # We match by last element of section_path against section.title.
        last_title = chunk.section_path[-1] if chunk.section_path else ""
        # Store under the title key; we resolve to section_id below.
        sec_text.setdefault(last_title, []).append(chunk.text)

    lines: list[str] = []
    char_count = 0

    # Sections
    for sec in doc.sections:
        texts = sec_text.get(sec.title, [])
        snippet = " ".join(texts)[:400].strip()
        entry = (
            f"\n[Section {sec.section_id}: {sec.title} (level {sec.level})]\n"
            f"{snippet}\n"
        )
        if char_count + len(entry) > max_chars:
            break
        lines.append(entry)
        char_count += len(entry)

    # Figures and tables (append after sections; these are short)
    for fig in doc.figures:
        cap_text = fig.caption.text if fig.caption else ""
        entry = f"\n[{fig.label}] {cap_text}\n"
        lines.append(entry)

    for tbl in doc.tables:
        cap_text = tbl.caption.text if tbl.caption else ""
        entry = f"\n[{tbl.label}] {cap_text}\n"
        lines.append(entry)

    return "".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Lenient JSON extractor: strip markdown code fences if the LLM added them.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present; return raw JSON string."""
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _resolve_term_chunk(term: str, chunks: list[Chunk]) -> Optional[tuple[int, str]]:
    """
    Find the first body chunk that contains *term* (case-insensitive).

    Returns (page, chunk_id) or None if not found.
    """
    term_lower = term.lower()
    for chunk in chunks:
        if chunk.chunk_type == "caption":
            continue
        if term_lower in chunk.text.lower():
            return chunk.page, chunk.chunk_id
    return None


def _parse_sketch_response(
    text: str,
    doc: StructuredDocument,
) -> tuple[list[SectionSummary], list[FigureEntry], list[FigureEntry], list[TermEntry]]:
    """
    Parse the LLM's JSON response into sketch components.

    Falls back to empty lists for any field that fails to parse, so the
    caller always receives a valid PaperSketch even on partial failures.
    """
    raw_json = _extract_json(text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # If JSON parsing fails entirely, return empty components.
        return [], [], [], []

    # Build section lookup: section_id -> Section
    sec_by_id: dict[str, Section] = {s.section_id: s for s in doc.sections}

    # --- section_map ---
    section_map: list[SectionSummary] = []
    for item in data.get("section_summaries", []):
        sid = item.get("section_id", "")
        one_liner = item.get("one_liner", "")
        sec = sec_by_id.get(sid)
        if sec is None:
            continue
        section_map.append(SectionSummary(
            section_id=sid,
            title=sec.title,
            level=sec.level,
            one_liner=one_liner,
        ))

    # --- figure/table indexes ---
    # Build quick label -> Figure/Table lookup
    fig_by_label: dict[str, Figure] = {f.label: f for f in doc.figures}
    tbl_by_label: dict[str, Table]  = {t.label: t for t in doc.tables}

    figure_index: list[FigureEntry] = []
    table_index:  list[FigureEntry] = []

    for item in data.get("figure_summaries", []):
        label   = item.get("label", "")
        summary = item.get("caption_summary", "")

        if label in fig_by_label:
            fig = fig_by_label[label]
            figure_index.append(FigureEntry(
                label=label,
                caption_summary=summary,
                page=fig.bbox.page,
                crop_url=fig.crop_url,
            ))
        elif label in tbl_by_label:
            tbl = tbl_by_label[label]
            table_index.append(FigureEntry(
                label=label,
                caption_summary=summary,
                page=tbl.bbox.page,
                crop_url=tbl.crop_url,
            ))
        # If the label doesn't match any known figure/table, skip silently.

    # --- key_terms ---
    key_terms: list[TermEntry] = []
    for item in data.get("key_terms", []):
        term = item.get("term", "").strip()
        if not term:
            continue
        result = _resolve_term_chunk(term, doc.chunks)
        if result is None:
            continue
        first_page, chunk_id = result
        key_terms.append(TermEntry(
            term=term,
            first_page=first_page,
            chunk_id=chunk_id,
        ))

    return section_map, figure_index, table_index, key_terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sketch(
    doc: StructuredDocument,
    model_spec: str = "openai:gpt-4o-mini",
    max_context_chars: int = 20_000,
) -> PaperSketch:
    """
    Generate a navigable PaperSketch from a StructuredDocument.

    A single LLM call produces one-sentence summaries for every section,
    short caption descriptions for every figure and table, and up to 10
    key technical terms — all in one JSON response.

    Args:
        doc:               StructuredDocument from the extraction + chunking pipeline.
        model_spec:        Provider:model string, e.g. "openai:gpt-4o-mini".
        max_context_chars: Maximum characters of content to send in the prompt.
                           The sketch prompt is compact, so 20k chars is usually
                           sufficient to cover all sections.

    Returns:
        PaperSketch with section_map, figure_index, table_index, key_terms.
        Fields that the LLM could not resolve are returned as empty lists.
    """
    client, provider, model_name = resolve(model_spec)

    content = _format_content_for_sketch(doc, max_context_chars)
    prompt  = _PROMPT_TEMPLATE.format(content=content)

    t0 = time.perf_counter()
    resp = client.summarize(prompt=prompt, model=model_name)
    _ = int((time.perf_counter() - t0) * 1000)  # llm_ms (available if needed)

    section_map, figure_index, table_index, key_terms = _parse_sketch_response(
        resp.text, doc
    )

    return PaperSketch(
        title=doc.title,
        section_map=section_map,
        figure_index=figure_index,
        table_index=table_index,
        key_terms=key_terms,
    )
