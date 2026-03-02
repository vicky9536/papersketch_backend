"""
Phase 4 — Paragraph-based chunking with provenance metadata.

Converts a list of TextBlocks (in reading order) into retrievable Chunk
objects.  Each chunk carries its section path, page, bbox, and an estimated
token count so downstream prompts can do token-budgeted context selection.

Design decisions
----------------
- Token counting uses a word-count proxy (words × 1.3) to avoid importing
  a full tokeniser.  Swap to tiktoken by replacing _approx_tokens().
- Caption blocks (detected by matching against Figure/Table Caption bboxes)
  are always emitted as atomic, single-block chunks with chunk_type="caption".
- Body blocks are accumulated across consecutive blocks in the same section
  up to MAX_CHUNK_TOKENS.  A block that already exceeds MAX_CHUNK_TOKENS on
  its own is split at sentence boundaries with OVERLAP_TOKENS of rolling
  context so the LLM sees smooth transitions across split points.
- Section boundaries always force a flush: a new chunk never spans two
  different sections.
- All constants are tunable via environment variables without code changes.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

from papersketch_backend.document.models import BBox, Chunk, Figure, Section, Table, TextBlock
from papersketch_backend.document.sections import get_section_path


# ---------------------------------------------------------------------------
# Tunable constants (override via environment)
# ---------------------------------------------------------------------------

MAX_CHUNK_TOKENS  = int(os.environ.get("CHUNK_MAX_TOKENS",  512))
MIN_CHUNK_TOKENS  = int(os.environ.get("CHUNK_MIN_TOKENS",   64))
OVERLAP_TOKENS    = int(os.environ.get("CHUNK_OVERLAP_TOKENS", 64))


# ---------------------------------------------------------------------------
# Token-count proxy
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Approximate token count: word count × 1.3, minimum 1."""
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Section path map
# ---------------------------------------------------------------------------

def _build_section_path_map(sections: list[Section]) -> dict[str, list[str]]:
    """Pre-compute {section_id: [ancestor_title, …, own_title]} for all sections."""
    return {s.section_id: get_section_path(s.section_id, sections) for s in sections}


# ---------------------------------------------------------------------------
# Sentence splitting with overlap
# ---------------------------------------------------------------------------

# Sentence boundary: period / ! / ? followed by whitespace + capital letter.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u2018\u2019\u201c\u201d])")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into a list of sentences (non-empty, stripped)."""
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split a long block of text into overlapping token-bounded sub-chunks.

    Strategy
    --------
    1. Split the text into sentences.
    2. Greedily pack sentences into the current accumulator until the next
       sentence would push the total past *max_tokens*.
    3. On overflow: emit the accumulator as a sub-chunk, then seed the next
       accumulator with trailing sentences totalling ≤ *overlap_tokens* so
       that context bleeds across the boundary.

    If the text cannot be split (no sentence boundaries), the whole text is
    returned as a single element regardless of length.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        tok = _approx_tokens(sent)

        if current and current_tokens + tok > max_tokens:
            # Emit the current accumulator
            chunks.append(" ".join(current))

            # Seed overlap: walk backward through current to collect
            # trailing sentences up to overlap_tokens.
            overlap: list[str] = []
            overlap_total = 0
            for s in reversed(current):
                t = _approx_tokens(s)
                if overlap_total + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_total += t

            current = overlap
            current_tokens = overlap_total

        current.append(sent)
        current_tokens += tok

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Chunk-id derivation
# ---------------------------------------------------------------------------

def _make_chunk_id(block_id: str, sub_idx: int = 0) -> str:
    """
    Stable 12-hex-char chunk id.

    For unsplit blocks sub_idx=0; for blocks that were split,
    each sub-piece gets a distinct sub_idx so ids never collide.
    """
    raw = f"{block_id}:{sub_idx}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Caption-block identification
# ---------------------------------------------------------------------------

def _caption_bbox_key(bbox: Optional[BBox]) -> str:
    """Stable string key for a BBox, used to cross-reference Caption ↔ TextBlock."""
    if bbox is None:
        return ""
    return f"{bbox.page}:{bbox.x0:.1f}:{bbox.y0:.1f}:{bbox.x1:.1f}:{bbox.y1:.1f}"


def _find_caption_block_ids(
    blocks: list[TextBlock],
    figures: list[Figure],
    tables: list[Table],
    tol: float = 2.0,
) -> set[str]:
    """
    Return the set of block_ids whose bbox matches a Figure or Table Caption.

    Matching is done by comparing bbox coordinates within *tol* PDF points on
    each axis.  This works because Phase 3 (figures.py) constructs Caption
    objects directly from RawBlock.bbox, and TextBlock.bbox is preserved from
    the same RawBlock.

    Args:
        blocks:  TextBlock list from sections.build_section_tree().
        figures: Figure list from figures.extract_figures_tables().
        tables:  Table list from figures.extract_figures_tables().
        tol:     Coordinate tolerance in PDF user-space points.

    Returns:
        Set of block_id strings for blocks that are figure/table captions.
    """
    # Collect all caption bboxes
    caption_bboxes: list[BBox] = []
    for fig in figures:
        if fig.caption:
            caption_bboxes.append(fig.caption.bbox)
    for tbl in tables:
        if tbl.caption:
            caption_bboxes.append(tbl.caption.bbox)

    if not caption_bboxes:
        return set()

    caption_ids: set[str] = set()
    for blk in blocks:
        bb = blk.bbox
        for cb in caption_bboxes:
            if (
                bb.page == cb.page
                and abs(bb.x0 - cb.x0) < tol
                and abs(bb.y0 - cb.y0) < tol
                and abs(bb.x1 - cb.x1) < tol
                and abs(bb.y1 - cb.y1) < tol
            ):
                caption_ids.add(blk.block_id)
                break

    return caption_ids


# ---------------------------------------------------------------------------
# Internal piece representation
# ---------------------------------------------------------------------------

@dataclass
class _Piece:
    """Intermediate unit: either one TextBlock or a sub-split of a large one."""
    text: str
    block_id: str
    section_id: str
    page: int
    bbox: Optional[BBox]
    is_caption: bool
    sub_idx: int  # 0 for unsplit blocks; 0,1,2,… for split pieces


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def chunk_document(
    blocks: list[TextBlock],
    sections: list[Section],
    figures: list[Figure],
    tables: list[Table],
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Convert TextBlocks into retrievable Chunks with provenance metadata.

    Processing pipeline
    -------------------
    1. **Expand**: each TextBlock becomes one or more _Piece objects.
       - Caption blocks → one piece, flagged is_caption=True.
       - Body blocks ≤ max_tokens → one piece.
       - Body blocks > max_tokens → split by sentence boundary into
         overlapping sub-pieces (each ≤ max_tokens).

    2. **Accumulate + flush**: iterate pieces in order.
       - Caption pieces: flush any pending accumulator, then emit the
         caption as its own Chunk (chunk_type="caption").
       - Body pieces: merge into the running accumulator when the merged
         result stays within the same section and ≤ max_tokens.  Otherwise
         flush the accumulator and start a new one with the current piece.

    3. **Emit**: each flush produces one Chunk.  The chunk_id is a 12-hex
       SHA-256 derived from the first source block_id + sub_idx so that ids
       are stable across re-runs given the same PDF.

    The min_tokens constraint is handled implicitly: because we only flush
    when forced (overflow or section boundary), small blocks are naturally
    absorbed into their neighbours.

    Args:
        blocks:        TextBlock list in reading order (output of
                       sections.build_section_tree).
        sections:      Section list (same source; used for section_path).
        figures:       Figure list (output of figures.extract_figures_tables).
        tables:        Table list (same source).
        max_tokens:    Upper bound on tokens per emitted chunk.
        min_tokens:    Soft lower bound; small blocks are merged forward
                       when possible rather than emitted alone.
        overlap_tokens: Tokens of rolling context included at the start of
                        each sub-chunk produced by splitting a large block.

    Returns:
        List of Chunk objects in document reading order.
    """
    section_paths = _build_section_path_map(sections)
    caption_ids   = _find_caption_block_ids(blocks, figures, tables)

    # ------------------------------------------------------------------
    # Step 1: expand blocks into pieces
    # ------------------------------------------------------------------
    pieces: list[_Piece] = []

    for blk in blocks:
        is_cap = blk.block_id in caption_ids

        if is_cap:
            pieces.append(_Piece(
                text=blk.text,
                block_id=blk.block_id,
                section_id=blk.section_id,
                page=blk.bbox.page,
                bbox=blk.bbox,
                is_caption=True,
                sub_idx=0,
            ))
            continue

        tok = _approx_tokens(blk.text)
        if tok <= max_tokens:
            pieces.append(_Piece(
                text=blk.text,
                block_id=blk.block_id,
                section_id=blk.section_id,
                page=blk.bbox.page,
                bbox=blk.bbox,
                is_caption=False,
                sub_idx=0,
            ))
        else:
            sub_texts = _split_by_tokens(blk.text, max_tokens, overlap_tokens)
            for sub_idx, sub_text in enumerate(sub_texts):
                pieces.append(_Piece(
                    text=sub_text,
                    block_id=blk.block_id,
                    section_id=blk.section_id,
                    page=blk.bbox.page,
                    bbox=blk.bbox,
                    is_caption=False,
                    sub_idx=sub_idx,
                ))

    # ------------------------------------------------------------------
    # Step 2: accumulate pieces and flush into Chunks
    # ------------------------------------------------------------------
    chunks: list[Chunk] = []

    # Accumulator state
    acc_texts:      list[str]  = []
    acc_block_ids:  list[str]  = []
    acc_sub_idxs:   list[int]  = []
    acc_section_id: str        = ""
    acc_page:       int        = 0
    acc_bbox:       Optional[BBox] = None
    acc_tokens:     int        = 0

    def _flush() -> None:
        nonlocal acc_texts, acc_block_ids, acc_sub_idxs
        nonlocal acc_section_id, acc_page, acc_bbox, acc_tokens

        if not acc_texts:
            return

        merged_text = " ".join(acc_texts)
        tok = _approx_tokens(merged_text)
        cid = _make_chunk_id(acc_block_ids[0], acc_sub_idxs[0])

        # Deduplicate source block ids while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for bid in acc_block_ids:
            if bid not in seen:
                seen.add(bid)
                deduped.append(bid)

        chunks.append(Chunk(
            chunk_id=cid,
            text=merged_text,
            section_path=section_paths.get(acc_section_id, []),
            page=acc_page,
            token_count=tok,
            source_block_ids=deduped,
            bbox=acc_bbox,
            chunk_type="body",
        ))

        acc_texts      = []
        acc_block_ids  = []
        acc_sub_idxs   = []
        acc_section_id = ""
        acc_page       = 0
        acc_bbox       = None
        acc_tokens     = 0

    for piece in pieces:
        # ---- caption: always atomic ----
        if piece.is_caption:
            _flush()
            tok = _approx_tokens(piece.text)
            cid = _make_chunk_id(piece.block_id, piece.sub_idx)
            chunks.append(Chunk(
                chunk_id=cid,
                text=piece.text,
                section_path=section_paths.get(piece.section_id, []),
                page=piece.page,
                token_count=tok,
                source_block_ids=[piece.block_id],
                bbox=piece.bbox,
                chunk_type="caption",
            ))
            continue

        # ---- body piece ----
        tok = _approx_tokens(piece.text)

        if acc_texts:
            same_section   = piece.section_id == acc_section_id
            would_overflow = acc_tokens + tok > max_tokens

            if not same_section or would_overflow:
                _flush()

        # Initialise accumulator on first body piece (or after flush)
        if not acc_texts:
            acc_section_id = piece.section_id
            acc_page       = piece.page
            acc_bbox       = piece.bbox

        acc_texts.append(piece.text)
        acc_block_ids.append(piece.block_id)
        acc_sub_idxs.append(piece.sub_idx)
        acc_tokens += tok

    # Final flush
    _flush()

    return chunks
