"""
Phase 2b — Layout detection.

Two entry points are provided:

detect_layout_regions(page_image_path, page_idx)
    Image-based ML stub.  Returns [] until a model (surya, layoutparser, …)
    is plugged in.  Callers treat [] as "fall back to heuristics".

detect_document_layout(pdf_path, max_pages)
    PDF-based PyMuPDF implementation.  Processes the whole document and
    returns one list of LayoutRegion objects per page, covering:

        "figure"  — raster image blocks + merged vector-drawing clusters
        "text"    — body-text and heading blocks
        "caption" — blocks starting with "Figure N" / "Table N"
        "header"  — text within the top 6 % of the page
        "footer"  — text within the bottom 6 % of the page

    Phase 3 (figures.py) only consumes "figure" and "table" regions, so
    providing rich region typing here costs nothing and makes the data useful
    for future phases without changing any downstream interface.

Upgrade path
------------
When a real ML layout model is available:

    1. Implement detect_layout_regions() to call the model on the rendered PNG.
    2. In detect_document_layout(), call detect_layout_regions() per page
       instead of _detect_page_layout_pymupdf().

The rest of the pipeline (Phase 3 onwards) requires no changes.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Literal

import fitz  # PyMuPDF

from papersketch_backend.document.models import BBox


# ---------------------------------------------------------------------------
# Public dataclass and type alias
# ---------------------------------------------------------------------------

RegionType = Literal[
    "text",
    "figure",
    "table",
    "caption",
    "header",
    "footer",
    "equation",
    "other",
]


@dataclass(frozen=True)
class LayoutRegion:
    """
    A typed rectangular region detected on a page.

    region_type: semantic category of the region.
    bbox:        bounding box in PDF user-space points (same coordinate system
                 as PyMuPDF so it can be aligned with text-path results).
    confidence:  detection score [0, 1].  Heuristic detectors use 1.0;
                 ML-based detectors supply a real score.
    """
    region_type: RegionType
    bbox: BBox
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Caption prefix pattern — matches "Figure 3", "Fig. 3", "Table 2"
_CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*\d+", re.IGNORECASE)

# Pages whose y-coordinate falls within this fraction of the page height
# are treated as header / footer noise.
_EDGE_FRACTION = 0.06

# Nearby drawing paths are merged into clusters; clusters smaller than
# this area (PDF pts²) are dropped as decorative elements.
_MIN_FIGURE_AREA = 2_500.0

# Margin used when deciding whether two drawing rects are "nearby".
_DRAWING_MERGE_MARGIN = 15.0


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _merge_drawing_rects(
    rects: list[fitz.Rect],
    margin: float = _DRAWING_MERGE_MARGIN,
    min_area: float = _MIN_FIGURE_AREA,
) -> list[fitz.Rect]:
    """
    Greedily merge nearby rectangles and discard tiny ones.

    Two rects are merged when the first, expanded by *margin* on every side,
    intersects the second.  Iteration repeats until no more merges occur so
    that chain-connected clusters (A close to B, B close to C) are fully
    collapsed.
    """
    changed = True
    while changed:
        changed = False
        output: list[fitz.Rect] = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            cur = fitz.Rect(rects[i])
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                expanded = fitz.Rect(
                    cur.x0 - margin, cur.y0 - margin,
                    cur.x1 + margin, cur.y1 + margin,
                )
                if expanded.intersects(rects[j]):
                    cur = cur | rects[j]
                    used[j] = True
                    changed = True
            output.append(cur)
        rects = output

    return [r for r in rects if r.width * r.height >= min_area]


def _rect_overlaps_regions(r: fitz.Rect, regions: list[LayoutRegion]) -> bool:
    """Return True if *r* intersects any existing "figure" LayoutRegion."""
    for reg in regions:
        if reg.region_type != "figure":
            continue
        b = reg.bbox
        ix0 = max(r.x0, b.x0)
        iy0 = max(r.y0, b.y0)
        ix1 = min(r.x1, b.x1)
        iy1 = min(r.y1, b.y1)
        if max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) > 0.0:
            return True
    return False


def _block_text(block: dict) -> str:
    """Join all span texts from a PyMuPDF text-block dict."""
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return "".join(parts).strip()


def _span_stats(block: dict) -> tuple[float, bool]:
    """Return (avg_font_size, is_bold) from a PyMuPDF text-block dict."""
    sizes: list[float] = []
    bold = 0
    total = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sz = float(span.get("size", 0.0))
            if sz > 0:
                sizes.append(sz)
            if span.get("flags", 0) & 16:
                bold += 1
            total += 1
    avg = statistics.mean(sizes) if sizes else 0.0
    return avg, (bold > total / 2) if total > 0 else False


def _classify_text_region(
    text: str,
    y0: float,
    y1: float,
    page_height: float,
) -> RegionType:
    """
    Classify a text block into a layout region type.

    Rules (in priority order):
      1. Near top edge    → "header"
      2. Near bottom edge → "footer"
      3. Caption prefix   → "caption"
      4. Everything else  → "text"  (covers headings and body)
    """
    if page_height > 0:
        if y1 < page_height * _EDGE_FRACTION:
            return "header"
        if y0 > page_height * (1.0 - _EDGE_FRACTION):
            return "footer"
    if _CAPTION_RE.match(text):
        return "caption"
    return "text"


# ---------------------------------------------------------------------------
# Per-page PyMuPDF layout detector
# ---------------------------------------------------------------------------

def _detect_page_layout_pymupdf(page: fitz.Page, page_idx: int) -> list[LayoutRegion]:
    """
    Detect layout regions on a single page using PyMuPDF data.

    Sources
    -------
    Raster images (block type=1):
        Directly added as "figure" regions.

    Text blocks (block type=0):
        Classified into "header", "footer", "caption", or "text" using
        simple position and text-prefix heuristics.

    Vector drawings (page.get_drawings()):
        Individual path bboxes are merged into clusters.  Clusters that do
        not overlap any already-detected raster figure are added as "figure"
        regions (lower confidence=0.9 to signal they are heuristic).

    Returns
    -------
    List of LayoutRegion objects covering all detected regions.
    """
    regions: list[LayoutRegion] = []
    page_height = page.rect.height
    page_dict = page.get_text("dict")
    drawing_rects: list[fitz.Rect] = []

    for blk in page_dict.get("blocks", []):
        raw_bbox = blk.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = map(float, raw_bbox)
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            continue

        block_bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page_idx)

        if blk.get("type") == 1:
            # Raster image block
            if area >= _MIN_FIGURE_AREA:
                regions.append(LayoutRegion(
                    region_type="figure",
                    bbox=block_bbox,
                    confidence=1.0,
                ))

        elif blk.get("type") == 0:
            # Text block
            text = _block_text(blk)
            if not text:
                continue
            rtype = _classify_text_region(text, y0, y1, page_height)
            regions.append(LayoutRegion(
                region_type=rtype,
                bbox=block_bbox,
                confidence=1.0,
            ))

    # Vector drawings — collect all path bboxes then merge
    for drw in page.get_drawings():
        r = drw.get("rect")
        if r:
            r = fitz.Rect(r)
            if not r.is_empty and not r.is_infinite:
                drawing_rects.append(r)

    for merged_rect in _merge_drawing_rects(drawing_rects):
        # Skip if the merged cluster overlaps a raster-image figure region
        if _rect_overlaps_regions(merged_rect, regions):
            continue
        regions.append(LayoutRegion(
            region_type="figure",
            bbox=BBox(
                x0=merged_rect.x0, y0=merged_rect.y0,
                x1=merged_rect.x1, y1=merged_rect.y1,
                page=page_idx,
            ),
            confidence=0.9,
        ))

    return regions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_layout_regions(
    page_image_path: str,
    page_idx: int,
) -> list[LayoutRegion]:
    """
    Image-based layout detection stub (ML upgrade point).

    Returns an empty list until a model is integrated.  Callers treat []
    as "use the PyMuPDF-based fallback".

    To integrate a model (e.g. surya):

        from surya.detection import batch_detection
        regions = batch_detection([page_image_path], ...)
        return [LayoutRegion(region_type=..., bbox=..., confidence=...) for ...]

    Args:
        page_image_path: absolute path to the rendered PNG for this page.
        page_idx:        0-based page index (used to set BBox.page).
    """
    return []


def detect_document_layout(
    pdf_path: str,
    max_pages: int = 30,
) -> list[list[LayoutRegion]]:
    """
    Detect layout regions for all pages of a PDF using PyMuPDF.

    Returns a list with one entry per processed page, where each entry is a
    list of LayoutRegion objects for that page.  The outer list is always
    exactly min(page_count, max_pages) long, so callers can index it by
    page_idx safely.

    Region types produced
    ---------------------
    "figure"  — raster image blocks and merged vector-drawing clusters.
    "text"    — body text and heading blocks.
    "caption" — blocks whose text starts with "Figure N" or "Table N".
    "header"  — text in the top 6 % of the page.
    "footer"  — text in the bottom 6 % of the page.

    Phase 3 only consumes "figure" and "table" regions.  "table" is not
    produced here (requires ML); the caption-text matching in Phase 3 handles
    the Figure vs. Table classification instead.

    Args:
        pdf_path:  Local filesystem path to the PDF.
        max_pages: Maximum number of pages to process.

    Returns:
        list[list[LayoutRegion]] — one inner list per processed page.
    """
    result: list[list[LayoutRegion]] = []
    with fitz.open(pdf_path) as doc:
        n = min(doc.page_count, max_pages)
        for page_idx in range(n):
            page = doc.load_page(page_idx)
            result.append(_detect_page_layout_pymupdf(page, page_idx))
    return result
