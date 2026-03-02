"""
Phase 6b — New structured-knowledge API endpoints.

Three endpoints that build on the Phase 2–5 pipeline:

  GET /api/v1/paper_structure
      Returns the full StructuredDocument (sections, chunks, figures, tables)
      as a JSON object.  Useful for debugging or building custom UIs that
      need the raw extraction.

  GET /api/v1/paper_summary
      Runs the extraction pipeline (or uses cache) then calls an LLM to
      produce a PaperSummary with chunk-level citations.

  GET /api/v1/paper_sketch
      Runs the extraction pipeline (or uses cache) then calls an LLM to
      produce a PaperSketch (section TOC, figure index, key terms).

The existing /papersketch_url and /papersketch_compare endpoints are
unchanged (full backward compatibility).

Caching
-------
StructuredDocuments are cached by pdf_sha256 under PAPERSKETCH_CACHE_DIR
(default: "cache").  Subsequent calls for the same PDF URL hit the cache
and skip re-extraction, so only the LLM call is repeated.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query

from papersketch_backend.api.deps import require_api_key
from papersketch_backend.settings import get_default_model
from papersketch_backend.pipeline.preprocess import preprocess_paper
from papersketch_backend.knowledge.summary import generate_summary
from papersketch_backend.knowledge.sketch import generate_sketch


router = APIRouter()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/paper_structure")
def paper_structure(
    url: str = Query(..., description="PDF or arXiv URL"),
    max_pages: int = Query(30, ge=1, le=60, description="Max PDF pages to extract"),
    x_api_key: str | None = Header(default=None),
) -> Any:
    """
    Return the full StructuredDocument for a paper.

    Includes sections, text blocks, figures, tables, and chunks — the raw
    output of the Phase 2–4 extraction pipeline.  Useful for debugging the
    extraction or building custom downstream tools.

    The document is cached by pdf_sha256 so repeat calls are cheap.
    """
    require_api_key(x_api_key)
    try:
        ctx = preprocess_paper(url=url, max_pages=max_pages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction pipeline failed: {exc}") from exc
    return dataclasses.asdict(ctx.document)


@router.get("/paper_summary")
def paper_summary(
    url: str = Query(..., description="PDF or arXiv URL"),
    model: str = Query(
        get_default_model(),
        description="Model spec, e.g. openai:gpt-4o-mini or gemini:gemini-2.0-flash",
    ),
    max_pages: int = Query(30, ge=1, le=60, description="Max PDF pages to extract"),
    max_context_chars: int = Query(
        60_000, ge=5_000, le=200_000,
        description="Max characters of chunk content sent to the LLM",
    ),
    x_api_key: str | None = Header(default=None),
) -> Any:
    """
    Generate a structured paper summary with chunk-level citations.

    Pipeline:
      URL → download PDF → structured extraction (cached) → LLM summary call
      → PaperSummary with one_liner, problem_and_method, experiments_and_results,
        limitations; every bullet cites source chunk_ids.

    The extraction result is cached by pdf_sha256; only the LLM call is
    repeated on subsequent requests for the same PDF.
    """
    require_api_key(x_api_key)
    try:
        ctx = preprocess_paper(url=url, max_pages=max_pages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction pipeline failed: {exc}") from exc

    try:
        summary = generate_summary(
            ctx.document,
            model_spec=model,
            max_context_chars=max_context_chars,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {exc}"
        ) from exc

    return dataclasses.asdict(summary)


@router.get("/paper_sketch")
def paper_sketch(
    url: str = Query(..., description="PDF or arXiv URL"),
    model: str = Query(
        get_default_model(),
        description="Model spec, e.g. openai:gpt-4o-mini or gemini:gemini-2.0-flash",
    ),
    max_pages: int = Query(30, ge=1, le=60, description="Max PDF pages to extract"),
    max_context_chars: int = Query(
        20_000, ge=2_000, le=100_000,
        description="Max characters of content sent to the LLM for sketch generation",
    ),
    x_api_key: str | None = Header(default=None),
) -> Any:
    """
    Generate a navigable paper sketch (section TOC + figure index + key terms).

    Pipeline:
      URL → download PDF → structured extraction (cached) → LLM sketch call
      → PaperSketch with title, section_map, figure_index, table_index,
        key_terms.

    All section summaries are obtained in a single LLM call.  The extraction
    result is cached by pdf_sha256.
    """
    require_api_key(x_api_key)
    try:
        ctx = preprocess_paper(url=url, max_pages=max_pages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction pipeline failed: {exc}") from exc

    try:
        sketch = generate_sketch(
            ctx.document,
            model_spec=model,
            max_context_chars=max_context_chars,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Sketch generation failed: {exc}"
        ) from exc

    return dataclasses.asdict(sketch)
