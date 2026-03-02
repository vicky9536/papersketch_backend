from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass
from typing import Literal

import fitz  # PyMuPDF

from papersketch_backend.document.models import BBox


BlockType = Literal["heading", "body", "caption", "header_footer", "image", "other"]

# Matches "Figure 3", "Fig. 3", "Table 2" at the start of a block
_CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*\d+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Intermediate block (before section assignment and chunking)
# ---------------------------------------------------------------------------

@dataclass
class RawBlock:
    """
    A text block extracted from a single PDF page, with spatial and
    typographic metadata.  Produced by extract_page_blocks(); consumed
    by reading_order.py and sections.py.

    block_id:    stable 12-hex-char id derived from pdf_sha256 + position.
    page:        0-based page index.
    bbox:        bounding box in PDF user-space points.
    text:        full text of the block (lines joined with newline).
    avg_font_size: mean font size across all spans in the block.
    is_bold:     True if more than half of the spans are bold.
    block_type:  heuristic classification (see _classify()).
    page_width:  width of the page (needed for column detection).
    """
    block_id: str
    page: int
    bbox: BBox
    text: str
    avg_font_size: float
    is_bold: bool
    block_type: BlockType
    page_width: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_block_id(pdf_sha256: str, page: int, bbox: BBox) -> str:
    key = f"{pdf_sha256}:{page}:{bbox.x0:.1f}:{bbox.y0:.1f}:{bbox.x1:.1f}:{bbox.y1:.1f}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _span_stats(block: dict) -> tuple[float, bool]:
    """Return (avg_font_size, is_bold) from a PyMuPDF text-block dict."""
    sizes: list[float] = []
    bold_votes = 0
    total_spans = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(float(span.get("size", 0.0)))
            if span.get("flags", 0) & 16:  # bold bit in PyMuPDF flags
                bold_votes += 1
            total_spans += 1
    avg = statistics.mean(sizes) if sizes else 0.0
    is_bold = (bold_votes > total_spans / 2) if total_spans > 0 else False
    return avg, is_bold


def _block_text(block: dict) -> str:
    """Concatenate all span texts in a block, preserving line breaks."""
    lines_out: list[str] = []
    for line in block.get("lines", []):
        line_text = "".join(s.get("text", "") for s in line.get("spans", []))
        lines_out.append(line_text)
    return "\n".join(lines_out).strip()


def _classify(
    text: str,
    avg_font_size: float,
    is_bold: bool,
    bbox: BBox,
    median_font_size: float,
    page_height: float,
) -> BlockType:
    """
    Heuristic block classifier.

    Rules (applied in priority order):
      1. Near top/bottom edge → header_footer
      2. Large font + bold, or very large font → heading
      3. ALL-CAPS short text + bold → heading  (IEEE-style section headers)
      4. Starts with "Figure N" / "Table N" → caption
      5. Notably smaller font → other  (footnote, page number, reference)
      6. Everything else → body
    """
    # 1. Header / footer: within 6 % of page edge
    if page_height > 0 and (
        bbox.y1 < page_height * 0.06 or bbox.y0 > page_height * 0.94
    ):
        return "header_footer"

    # 2. Heading by font size
    if median_font_size > 0:
        if avg_font_size >= median_font_size * 1.1 and is_bold:
            return "heading"
        if avg_font_size >= median_font_size * 1.4:
            return "heading"

    # 3. ALL-CAPS heading (common in IEEE two-column papers)
    words = text.split()
    if (
        is_bold
        and 1 <= len(words) <= 10
        and text == text.upper()
        and text.isalpha()  # guard against numbers/punctuation
    ):
        return "heading"

    # 4. Caption
    if _CAPTION_RE.match(text):
        return "caption"

    # 5. Footnote / small annotation
    if median_font_size > 0 and avg_font_size < median_font_size * 0.85 and not is_bold:
        return "other"

    return "body"


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_page_blocks(
    page: fitz.Page,
    page_idx: int,
    pdf_sha256: str,
) -> list[RawBlock]:
    """
    Extract and classify all text blocks from a single PyMuPDF page.

    Returns RawBlock list in PyMuPDF's native order (top-to-bottom within
    each column).  Call reading_order.sort_blocks_reading_order() to recover
    the correct two-column reading order.
    """
    page_dict = page.get_text("dict")
    page_height = page.rect.height
    page_width = page.rect.width

    # First pass: collect raw data
    raw: list[tuple[BBox, str, float, bool]] = []
    for blk in page_dict.get("blocks", []):
        if blk.get("type") != 0:  # skip image blocks
            continue
        text = _block_text(blk)
        if not text.strip():
            continue
        avg_size, is_bold = _span_stats(blk)
        x0, y0, x1, y1 = blk["bbox"]
        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page_idx)
        raw.append((bbox, text, avg_size, is_bold))

    # Page-level median font size for relative classification
    sizes = [sz for (_, _, sz, _) in raw if sz > 0]
    median_size = statistics.median(sizes) if sizes else 10.0

    # Second pass: classify and build RawBlock objects
    result: list[RawBlock] = []
    for bbox, text, avg_size, is_bold in raw:
        btype = _classify(text, avg_size, is_bold, bbox, median_size, page_height)
        result.append(
            RawBlock(
                block_id=_make_block_id(pdf_sha256, page_idx, bbox),
                page=page_idx,
                bbox=bbox,
                text=text,
                avg_font_size=avg_size,
                is_bold=is_bold,
                block_type=btype,
                page_width=page_width,
            )
        )
    return result


def extract_all_blocks(
    pdf_path: str,
    pdf_sha256: str,
    max_pages: int = 30,
) -> list[RawBlock]:
    """
    Extract RawBlock objects from the full document (up to max_pages).

    Blocks are returned in page order but NOT yet sorted within each page
    for two-column reading order.  Pass the result to
    reading_order.sort_document_reading_order() next.
    """
    all_blocks: list[RawBlock] = []
    with fitz.open(pdf_path) as doc:
        n = min(doc.page_count, max_pages)
        for i in range(n):
            all_blocks.extend(extract_page_blocks(doc.load_page(i), i, pdf_sha256))
    return all_blocks
