from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Spatial primitive
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """
    Axis-aligned bounding box on a specific page.

    All coordinates are in PDF user-space points (same unit as PyMuPDF's
    fitz.Rect). x0/y0 = top-left, x1/y1 = bottom-right.
    page is 0-based.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    page: int  # 0-based page index

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BBox) -> float:
        """Intersection-over-union (0 if on different pages)."""
        if self.page != other.page:
            return 0.0
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """
    A logical section of the paper (heading + its content range).

    section_id: stable string identifier, e.g. "s1", "s1.2", "s3.1.1".
    level: 1 = top-level heading, 2 = subsection, 3 = sub-subsection.
    page_start / page_end: 0-based page indices.  page_end is None when the
    section extends to the last page (set during post-processing).
    """
    section_id: str
    title: str
    level: int
    page_start: int
    page_end: Optional[int] = None


@dataclass(frozen=True)
class TextBlock:
    """
    A single paragraph / text block extracted from a page.

    block_id: stable id derived from pdf_sha256 + page + bbox position.
    section_id: the Section this block belongs to (empty string = unassigned).
    """
    block_id: str
    text: str
    bbox: BBox
    section_id: str = ""


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Caption:
    """
    A figure or table caption.

    label: the detected label prefix, e.g. "Figure 3" or "Table 2".
    text:  full caption string including the label.
    """
    label: str
    text: str
    bbox: BBox


@dataclass(frozen=True)
class Figure:
    """
    A figure region detected in the PDF.

    label:      e.g. "Figure 3" (empty string if label could not be parsed).
    bbox:       bounding box of the figure region on its page.
    caption:    associated caption block, or None if not found.
    crop_path:  absolute local filesystem path to the cropped PNG, or None.
    crop_url:   public URL served via /static, or None.
    """
    label: str
    bbox: BBox
    caption: Optional[Caption] = None
    crop_path: Optional[str] = None
    crop_url: Optional[str] = None


@dataclass(frozen=True)
class Table:
    """
    A table region detected in the PDF.

    Same fields as Figure; kept as a separate type so callers can
    distinguish without inspecting the label string.
    """
    label: str
    bbox: BBox
    caption: Optional[Caption] = None
    crop_path: Optional[str] = None
    crop_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Retrieval chunks
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A retrievable text unit with full provenance metadata.

    chunk_id:        stable 12-hex-char id derived from content + position.
    text:            chunk text (may span multiple source TextBlocks).
    section_path:    ordered list of ancestor section titles, e.g.
                     ["3. Method", "3.2 Architecture"].
    page:            0-based page where the chunk starts.
    token_count:     estimated token count (word-proxy; see chunking.py).
    source_block_ids: ids of the TextBlocks this chunk was built from.
    bbox:            bounding box of the first source block (may be None
                     if the chunk was assembled from merged blocks on
                     different positions).
    """
    chunk_id: str
    text: str
    section_path: list[str]
    page: int
    token_count: int
    source_block_ids: list[str] = field(default_factory=list)
    bbox: Optional[BBox] = None
    chunk_type: str = "body"  # "body" | "caption"


# ---------------------------------------------------------------------------
# Top-level document container
# ---------------------------------------------------------------------------

@dataclass
class StructuredDocument:
    """
    Full structured representation of one paper.

    Produced by the extraction pipeline; consumed by chunking, summary,
    and sketch generation.

    pdf_sha256:  hex digest of the downloaded PDF (stable cache key).
    title:       best-effort title guess (may be None).
    sections:    flat list of Section objects in document order.
    blocks:      flat list of TextBlock objects in reading order.
    figures:     detected Figure objects (may include cropped PNGs).
    tables:      detected Table objects (may include cropped PNGs).
    chunks:      retrievable Chunk objects (populated by chunking phase).
    """
    pdf_sha256: str
    title: Optional[str]
    sections: list[Section] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
