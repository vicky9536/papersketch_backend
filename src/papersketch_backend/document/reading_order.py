from __future__ import annotations

from papersketch_backend.document.text_blocks import RawBlock


_FULL_WIDTH_THRESHOLD = 0.6  # block spans ≥ 60 % of page width → full-width


def _is_full_width(block: RawBlock, threshold: float = _FULL_WIDTH_THRESHOLD) -> bool:
    if block.page_width <= 0:
        return True  # unknown width → treat as full-width
    span = block.bbox.x1 - block.bbox.x0
    return (span / block.page_width) >= threshold


def sort_blocks_reading_order(
    blocks: list[RawBlock],
    full_width_threshold: float = _FULL_WIDTH_THRESHOLD,
) -> list[RawBlock]:
    """
    Sort blocks from a single page into correct reading order.

    Algorithm (handles single- and double-column layouts):
      1. Classify each block as full-width, left-column, or right-column.
      2. Full-width blocks (title, abstract, section headers spanning the page)
         divide the page into vertical zones.
      3. Within each zone: all left-column blocks (by y0) then all
         right-column blocks (by y0).

    This correctly handles the common academic paper layout:
      [Title / Abstract — full width]
      [Left column body]   [Right column body]

    And also handles full-width section headers that appear mid-page:
      [Left body above header] [Right body above header]
      [Section Header — full width]
      [Left body below header] [Right body below header]

    Limitation: if a full-width block appears at the same y-coordinate as
    column blocks, the full-width block is emitted first (conservative).
    """
    if not blocks:
        return []

    page_width = blocks[0].page_width
    mid_x = page_width / 2.0 if page_width > 0 else float("inf")

    full_width: list[RawBlock] = []
    left_col: list[RawBlock] = []
    right_col: list[RawBlock] = []

    for b in blocks:
        if _is_full_width(b, full_width_threshold):
            full_width.append(b)
        else:
            x_center = (b.bbox.x0 + b.bbox.x1) / 2.0
            (left_col if x_center <= mid_x else right_col).append(b)

    full_width.sort(key=lambda b: b.bbox.y0)
    left_col.sort(key=lambda b: b.bbox.y0)
    right_col.sort(key=lambda b: b.bbox.y0)

    result: list[RawBlock] = []
    prev_bottom = -1.0  # y-coordinate below the last emitted full-width block

    for fw in full_width:
        fw_top = fw.bbox.y0
        fw_bottom = fw.bbox.y1

        # Emit column blocks that sit above this full-width block
        result.extend(b for b in left_col if prev_bottom < b.bbox.y0 < fw_top)
        result.extend(b for b in right_col if prev_bottom < b.bbox.y0 < fw_top)

        # Emit the full-width block itself
        result.append(fw)
        prev_bottom = fw_bottom

    # Emit remaining column blocks after the last full-width block
    result.extend(b for b in left_col if b.bbox.y0 >= prev_bottom)
    result.extend(b for b in right_col if b.bbox.y0 >= prev_bottom)

    return result


def sort_document_reading_order(blocks: list[RawBlock]) -> list[RawBlock]:
    """
    Sort all blocks from the entire document into reading order.

    Groups blocks by page number, applies per-page two-column sorting,
    then concatenates in page order.
    """
    by_page: dict[int, list[RawBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    ordered: list[RawBlock] = []
    for page_idx in sorted(by_page):
        ordered.extend(sort_blocks_reading_order(by_page[page_idx]))
    return ordered
