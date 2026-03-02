from __future__ import annotations

from typing import Optional

from papersketch_backend.document.models import Section, TextBlock
from papersketch_backend.document.text_blocks import RawBlock


# ---------------------------------------------------------------------------
# Level inference
# ---------------------------------------------------------------------------

def _assign_level(font_size: float, seen_sizes: list[float]) -> int:
    """
    Infer heading level by comparing font_size to previously seen heading
    font sizes.

    Strategy: sort all known heading sizes descending; the rank of the
    current font size within that ordering is its level (1 = largest).
    Sizes within 5 % of each other are treated as the same level.
    """
    if not seen_sizes:
        return 1

    # Deduplicate with 5 % tolerance: build a list of "canonical" sizes
    canonical: list[float] = []
    for s in sorted(seen_sizes, reverse=True):
        if not canonical or s < canonical[-1] * 0.95:
            canonical.append(s)

    for rank, canon_size in enumerate(canonical, start=1):
        if font_size >= canon_size * 0.95:
            return rank

    return len(canonical) + 1  # smaller than anything seen → deepest level


# ---------------------------------------------------------------------------
# Section tree builder
# ---------------------------------------------------------------------------

def build_section_tree(
    ordered_blocks: list[RawBlock],
) -> tuple[list[Section], list[TextBlock]]:
    """
    Walk blocks in reading order; construct Section objects from heading
    blocks and assign each body/caption block to its enclosing section.

    Args:
        ordered_blocks: output of reading_order.sort_document_reading_order().

    Returns:
        sections:    flat list of Section objects in document order, with
                     page_end filled in.
        text_blocks: list of TextBlock (body + caption), each tagged with
                     the section_id of its enclosing section.

    Notes:
        - header_footer and other blocks are silently dropped.
        - If the document starts with body text before any heading, those
          blocks are assigned section_id="" (pre-intro / unclassified).
        - section_id format: "s1", "s2", "s3", … (flat sequential numbering).
          Level is stored separately in Section.level for downstream use.
    """
    sections: list[Section] = []
    text_blocks: list[TextBlock] = []

    current_section_id: str = ""   # "" = before the first heading
    heading_count = 0
    heading_sizes: list[float] = []  # font sizes of all headings seen so far

    for blk in ordered_blocks:
        # ---- skip noise ----
        if blk.block_type in ("header_footer", "other", "image"):
            continue

        # ---- open a new section ----
        if blk.block_type == "heading":
            # Close the previous section by filling in page_end
            if sections:
                prev = sections[-1]
                sections[-1] = Section(
                    section_id=prev.section_id,
                    title=prev.title,
                    level=prev.level,
                    page_start=prev.page_start,
                    page_end=blk.page,
                )

            heading_count += 1
            level = _assign_level(blk.avg_font_size, heading_sizes)
            heading_sizes.append(blk.avg_font_size)

            section_id = f"s{heading_count}"
            sections.append(
                Section(
                    section_id=section_id,
                    title=blk.text.strip(),
                    level=level,
                    page_start=blk.page,
                    page_end=None,
                )
            )
            current_section_id = section_id
            continue

        # ---- body or caption block ----
        text_blocks.append(
            TextBlock(
                block_id=blk.block_id,
                text=blk.text,
                bbox=blk.bbox,
                section_id=current_section_id,
            )
        )

    # Close the last open section
    if sections and ordered_blocks:
        last_page = max(b.page for b in ordered_blocks)
        prev = sections[-1]
        sections[-1] = Section(
            section_id=prev.section_id,
            title=prev.title,
            level=prev.level,
            page_start=prev.page_start,
            page_end=last_page,
        )

    return sections, text_blocks


# ---------------------------------------------------------------------------
# Helper: build section_path for a given section_id
# ---------------------------------------------------------------------------

def get_section_path(
    section_id: str,
    sections: list[Section],
) -> list[str]:
    """
    Return the ordered list of ancestor section titles for a given section_id.

    With flat sequential numbering ("s1", "s2", …) the "path" is simply
    the section title hierarchy inferred from level numbers.

    Example: if sections are
        s1  level=1  "Introduction"
        s2  level=1  "Method"
        s3  level=2  "Architecture"
        s4  level=2  "Training"
        s5  level=1  "Experiments"

    get_section_path("s4", sections) → ["Method", "Training"]
    """
    if not section_id:
        return []

    by_id: dict[str, Section] = {s.section_id: s for s in sections}
    target = by_id.get(section_id)
    if target is None:
        return []

    # Walk backwards through sections to find ancestors by level
    path: list[str] = [target.title]
    target_idx = next(i for i, s in enumerate(sections) if s.section_id == section_id)

    needed_level = target.level - 1
    for s in reversed(sections[:target_idx]):
        if needed_level <= 0:
            break
        if s.level == needed_level:
            path.append(s.title)
            needed_level -= 1

    path.reverse()
    return path
