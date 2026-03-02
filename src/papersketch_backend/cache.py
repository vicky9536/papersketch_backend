"""
Phase 6c — JSON cache for StructuredDocument, keyed by pdf_sha256.

Files are written to  {cache_dir}/{sha256}.json  so repeated calls for the
same PDF skip the entire extraction pipeline.

Serialisation
-------------
dataclasses.asdict() handles the nested dataclass tree recursively.
The only subtlety is deserialization: we must reconstruct typed objects from
plain dicts.  Every constructor call mirrors the dataclass field order and
handles Optional fields that may be None in the stored JSON.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional

from papersketch_backend.document.models import (
    BBox,
    Caption,
    Chunk,
    Figure,
    Section,
    StructuredDocument,
    Table,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise(doc: StructuredDocument) -> dict[str, Any]:
    """Convert StructuredDocument to a plain dict (fully JSON-serialisable)."""
    return dataclasses.asdict(doc)


# ---------------------------------------------------------------------------
# Deserialisation helpers
# ---------------------------------------------------------------------------

def _bbox(d: Optional[dict]) -> Optional[BBox]:
    if d is None:
        return None
    return BBox(
        x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"], page=d["page"]
    )


def _caption(d: Optional[dict]) -> Optional[Caption]:
    if d is None:
        return None
    return Caption(
        label=d["label"],
        text=d["text"],
        bbox=_bbox(d["bbox"]),  # type: ignore[arg-type]
    )


def _section(d: dict) -> Section:
    return Section(
        section_id=d["section_id"],
        title=d["title"],
        level=d["level"],
        page_start=d["page_start"],
        page_end=d.get("page_end"),
    )


def _text_block(d: dict) -> TextBlock:
    return TextBlock(
        block_id=d["block_id"],
        text=d["text"],
        bbox=_bbox(d["bbox"]),  # type: ignore[arg-type]
        section_id=d.get("section_id", ""),
    )


def _figure(d: dict) -> Figure:
    return Figure(
        label=d["label"],
        bbox=_bbox(d["bbox"]),  # type: ignore[arg-type]
        caption=_caption(d.get("caption")),
        crop_path=d.get("crop_path"),
        crop_url=d.get("crop_url"),
    )


def _table(d: dict) -> Table:
    return Table(
        label=d["label"],
        bbox=_bbox(d["bbox"]),  # type: ignore[arg-type]
        caption=_caption(d.get("caption")),
        crop_path=d.get("crop_path"),
        crop_url=d.get("crop_url"),
    )


def _chunk(d: dict) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"],
        text=d["text"],
        section_path=d.get("section_path", []),
        page=d["page"],
        token_count=d["token_count"],
        source_block_ids=d.get("source_block_ids", []),
        bbox=_bbox(d.get("bbox")),
        chunk_type=d.get("chunk_type", "body"),
    )


def _deserialise(raw: dict) -> StructuredDocument:
    """Reconstruct a StructuredDocument from its plain-dict representation."""
    return StructuredDocument(
        pdf_sha256=raw["pdf_sha256"],
        title=raw.get("title"),
        sections=[_section(s) for s in raw.get("sections", [])],
        blocks=[_text_block(b) for b in raw.get("blocks", [])],
        figures=[_figure(f) for f in raw.get("figures", [])],
        tables=[_table(t) for t in raw.get("tables", [])],
        chunks=[_chunk(c) for c in raw.get("chunks", [])],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _cache_path(sha256: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{sha256}.json")


def load_cached(sha256: str, cache_dir: str) -> Optional[StructuredDocument]:
    """
    Return the cached StructuredDocument for *sha256*, or None on miss.

    Silently returns None if the cache file is missing, unreadable, or
    contains invalid JSON so a fresh extraction is triggered automatically.

    Args:
        sha256:    Hex SHA-256 of the downloaded PDF (the cache key).
        cache_dir: Directory that holds the JSON cache files.
    """
    path = _cache_path(sha256, cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return _deserialise(raw)
    except Exception:
        return None


def save_cached(doc: StructuredDocument, cache_dir: str) -> None:
    """
    Persist *doc* as a JSON file in *cache_dir*.

    Creates *cache_dir* if it does not exist.  Write errors are silently
    swallowed so a cache failure never breaks an API request.

    Args:
        doc:       StructuredDocument to persist.
        cache_dir: Directory for JSON cache files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(doc.pdf_sha256, cache_dir)
    try:
        payload = _serialise(doc)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except Exception:
        pass  # cache write failure is non-fatal
