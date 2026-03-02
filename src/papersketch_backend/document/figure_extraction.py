from __future__ import annotations

import os
import re
from typing import Optional

import fitz  # PyMuPDF

from papersketch_backend.document.models import BBox, Caption, Figure, Table
from papersketch_backend.document.layout import LayoutRegion
from papersketch_backend.document.text_blocks import RawBlock


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_FIGURE_RE  = re.compile(r"^(Figure|Fig\.)\s*(\d+)", re.IGNORECASE)
_TABLE_RE   = re.compile(r"^Table\s*(\d+)", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_bbox(r: fitz.Rect, page: int) -> BBox:
    return BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1, page=page)


def _label_slug(label: str) -> str:
    """'Figure 3' → 'figure_3'  (safe filename component)."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _merge_rects(
    rects: list[fitz.Rect],
    margin: float = 20.0,
    min_area: float = 2500.0,
) -> list[fitz.Rect]:
    """
    Greedily merge nearby rectangles into larger candidate regions.

    Two rects are merged when the first, expanded by `margin` on every side,
    intersects the second.  Iteration continues until no further merges occur
    so that chain-connected groups (A close to B, B close to C) are fully
    collapsed.  Regions smaller than `min_area` are discarded to filter out
    decorative elements (thin rules, tiny icons, etc.).
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


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

def _detect_rects_from_layout(
    layout_regions: list[LayoutRegion],
) -> tuple[list[fitz.Rect], list[bool]]:
    """
    Convert LayoutRegion objects into (rects, is_table_flags).

    Returns parallel lists: rects[i] is a table iff is_table[i] is True.
    """
    rects: list[fitz.Rect] = []
    is_table: list[bool] = []
    for r in layout_regions:
        if r.region_type not in ("figure", "table"):
            continue
        rects.append(fitz.Rect(r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1))
        is_table.append(r.region_type == "table")
    return rects, is_table


def _detect_rects_fallback(
    page: fitz.Page,
    drawing_merge_margin: float,
    min_figure_area: float,
) -> list[fitz.Rect]:
    """
    Heuristic figure-region detection when no ML layout output is available.

    Two sources are combined and then merged:
      1. Raster image blocks from get_text("dict") (block["type"] == 1).
      2. Vector drawing paths from page.get_drawings(), which captures
         matplotlib-style line charts, bar charts, diagrams, etc. that are
         stored as PDF path operators rather than embedded bitmaps.
    """
    raw: list[fitz.Rect] = []

    # 1. Raster images
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") == 1:
            r = fitz.Rect(blk["bbox"])
            if not r.is_empty and not r.is_infinite:
                raw.append(r)

    # 2. Vector drawings
    for drw in page.get_drawings():
        r = drw.get("rect")
        if r:
            r = fitz.Rect(r)
            if not r.is_empty and not r.is_infinite:
                raw.append(r)

    return _merge_rects(raw, margin=drawing_merge_margin, min_area=min_figure_area)


# ---------------------------------------------------------------------------
# Caption matching
# ---------------------------------------------------------------------------

def _find_caption(
    region: fitz.Rect,
    page_idx: int,
    caption_blocks: list[RawBlock],
    search_margin: float,
    used_ids: set[str],
) -> Optional[RawBlock]:
    """
    Find the nearest unused caption block associated with a detected region.

    Matching criteria:
      - Same page.
      - Horizontal overlap (the caption's x-range overlaps the region's
        x-range, with up to 50 pts tolerance for narrow columns).
      - Vertically within search_margin pts: captions below the region are
        preferred; captions above are accepted with a distance penalty so
        that "below" wins over equally-distant "above" captions.

    Returns the best match, or None if no caption qualifies.
    """
    best: Optional[RawBlock] = None
    best_dist = float("inf")

    for cb in caption_blocks:
        if cb.page != page_idx or cb.block_id in used_ids:
            continue

        # Horizontal overlap (allow 50 pt gap)
        h_overlap = min(region.x1, cb.bbox.x1) - max(region.x0, cb.bbox.x0)
        if h_overlap < -50:
            continue

        below_dist = cb.bbox.y0 - region.y1   # positive → caption is below
        above_dist = region.y0 - cb.bbox.y1   # positive → caption is above

        if 0 <= below_dist <= search_margin:
            dist = below_dist
        elif 0 <= above_dist <= search_margin:
            dist = above_dist + search_margin  # slight penalty for above
        else:
            continue

        if dist < best_dist:
            best_dist = dist
            best = cb

    return best


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def _crop_region(
    page: fitz.Page,
    rect: fitz.Rect,
    out_path: str,
    dpi: int,
    padding: float = 5.0,
) -> None:
    """Render a padded sub-region of a page and save as PNG."""
    padded = fitz.Rect(
        max(0.0, rect.x0 - padding),
        max(0.0, rect.y0 - padding),
        min(page.rect.width,  rect.x1 + padding),
        min(page.rect.height, rect.y1 + padding),
    )
    scale = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=padded,
        alpha=False,
    )
    pix.save(out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_figures_tables(
    pdf_path: str,
    pdf_sha256: str,
    raw_blocks: list[RawBlock],
    layout_pages: list[list[LayoutRegion]],
    base_url: str,
    static_dir: str = "static",
    dpi: int = 200,
    overwrite: bool = False,
    caption_search_margin: float = 80.0,
    min_figure_area: float = 2500.0,
    drawing_merge_margin: float = 20.0,
) -> tuple[list[Figure], list[Table]]:
    """
    Detect, caption-match, crop and return all figures and tables in a PDF.

    Detection pipeline (per page):
      Primary:  If layout_pages[page_idx] is non-empty (ML detector active),
                use LayoutRegion objects filtered by region_type.
      Fallback: Heuristic detection from raster image blocks and clustered
                vector drawing paths (PyMuPDF).

    Caption matching:
      For every detected region the nearest RawBlock with block_type ==
      "caption" is sought within caption_search_margin points vertically and
      with horizontal overlap.  The caption text is parsed to determine
      Figure vs Table and to extract the human-readable label ("Figure 3").
      Detections without a matching caption are silently discarded to avoid
      false positives from decorative elements.

    Cropping:
      Matched regions are rendered at `dpi` resolution to:
        {static_dir}/{pdf_sha256}/fig_{label_slug}.png
      and the public URL is:
        {base_url}/static/{pdf_sha256}/fig_{label_slug}.png
      Files are skipped (not re-rendered) when they already exist and
      overwrite=False.

    Args:
        pdf_path:              local path to the downloaded PDF.
        pdf_sha256:            hex SHA-256 of the PDF (used for file paths).
        raw_blocks:            all RawBlock objects from Phase 2a/2c.
        layout_pages:          per-page LayoutRegion lists from Phase 2b;
                               pass [] or a list of empty lists to use the
                               heuristic fallback for every page.
        base_url:              public base URL, e.g. "http://127.0.0.1:8001".
        static_dir:            local directory mounted at /static.
        dpi:                   render resolution for figure crops.
        overwrite:             re-render even if the crop file exists.
        caption_search_margin: max vertical distance (PDF pts) to search
                               for captions above/below a detected region.
        min_figure_area:       minimum bbox area (pts²) to keep a detected
                               region (filters tiny decorative elements).
        drawing_merge_margin:  margin (pts) used when clustering drawing paths.

    Returns:
        (figures, tables): lists of Figure and Table objects, each carrying
        bbox, Caption, crop_path, and crop_url.
    """
    base_url = (base_url or "").rstrip("/")
    out_dir = os.path.join(static_dir, pdf_sha256)
    _ensure_dir(out_dir)

    # Caption blocks are the anchor for matching
    caption_blocks = [b for b in raw_blocks if b.block_type == "caption"]

    figures: list[Figure] = []
    tables: list[Table] = []
    used_caption_ids: set[str] = set()

    with fitz.open(pdf_path) as doc:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)

            # ---- choose detection path ----
            page_layout = (
                layout_pages[page_idx]
                if page_idx < len(layout_pages)
                else []
            )

            if page_layout:
                all_rects, is_table_flags = _detect_rects_from_layout(page_layout)
            else:
                all_rects = _detect_rects_fallback(
                    page, drawing_merge_margin, min_figure_area
                )
                is_table_flags = [False] * len(all_rects)

            # ---- match each region to a caption ----
            for rect, hint_is_table in zip(all_rects, is_table_flags):
                cap_block = _find_caption(
                    rect, page_idx, caption_blocks,
                    caption_search_margin, used_caption_ids,
                )
                if cap_block is None:
                    continue  # no caption → skip to avoid false positives

                used_caption_ids.add(cap_block.block_id)
                cap_text = cap_block.text.strip()

                # Caption text overrides layout hint for figure vs table
                m_fig = _FIGURE_RE.match(cap_text)
                m_tbl = _TABLE_RE.match(cap_text)

                if m_fig:
                    label = m_fig.group(0)
                    is_table = False
                elif m_tbl:
                    label = m_tbl.group(0)
                    is_table = True
                else:
                    label = cap_text[:30]   # fallback: first 30 chars of caption
                    is_table = hint_is_table

                # ---- crop ----
                slug = _label_slug(label)
                filename = f"fig_{slug}.png"
                out_path = os.path.join(out_dir, filename)
                out_url  = f"{base_url}/static/{pdf_sha256}/{filename}"

                if overwrite or not (
                    os.path.exists(out_path) and os.path.getsize(out_path) > 0
                ):
                    _crop_region(page, rect, out_path, dpi)

                # ---- build Caption + Figure/Table ----
                caption = Caption(
                    label=label,
                    text=cap_text,
                    bbox=cap_block.bbox,
                )
                region_bbox = _to_bbox(rect, page_idx)

                if is_table:
                    tables.append(Table(
                        label=label,
                        bbox=region_bbox,
                        caption=caption,
                        crop_path=out_path,
                        crop_url=out_url,
                    ))
                else:
                    figures.append(Figure(
                        label=label,
                        bbox=region_bbox,
                        caption=caption,
                        crop_path=out_path,
                        crop_url=out_url,
                    ))

    return figures, tables
