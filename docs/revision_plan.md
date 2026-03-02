# Revision Plan: Structured Paper Understanding

Based on professor's recommendations. This plan upgrades the backend from a
"download → plain text → one-shot LLM" pipeline into a structured, retrievable
knowledge representation of each paper.

---

## Current State (Gaps)

| Concern | Current | Gap |
|---------|---------|-----|
| Text extraction | `page.get_text("text")` — flat string, no bbox | No position info; can't cite paragraph/section |
| Figure detection | Regex heuristics on page text | No actual figure/table bbox; captions not extracted |
| Reading order | Concatenate pages in order | Breaks on double-column layout |
| Chunking | Whole extracted text truncated at 24k chars | No structure-aware splitting; no chunk metadata |
| Summary citations | Free-form LLM markdown | No traceability back to source paragraphs |
| Paper Sketch | One LLM call, free-form | No navigable section map, no figure index |

---

## Target Architecture

```
PDF
 │
 ├─ [Text Path]    PyMuPDF get_text("dict") ──────────────────┐
 │                 → blocks with bbox, font, flags              │
 │                                                             ▼
 ├─ [Layout Path]  Render page PNG → layout detection    StructuredDocument
 │                 → typed regions: heading/body/fig/          │
 │                   table/caption/header/footer               │
 │                                                             │
 ├─ Align text blocks → layout regions (by bbox IOU)          │
 ├─ Recover reading order (double-column aware)               │
 ├─ Classify: Section / TextBlock / Figure / Table / Caption  │
 │                                                             │
 ├─ Figure/Table crop → save as PNG                           │
 ├─ Caption extraction → linked to figure/table               │
 │                                                             │
 ├─ Chunking (paragraph-based, token-bounded)                 │
 │   each chunk → { text, chunk_id, section_path,             │
 │                  page, bbox, type }                         │
 │                                                             ▼
 └─ Knowledge Artifacts
      ├─ Paper Summary  (structured, with chunk_id citations)
      └─ Paper Sketch   (section map + figure index + term glossary)
```

---

## Phase 1 — Structured Data Models

**Goal**: Define the data schema that all later phases produce and consume.

**New file**: `src/papersketch_backend/document/models.py`

```python
@dataclass
class BBox:
    x0: float; y0: float; x1: float; y1: float
    page: int  # 0-based

@dataclass
class Section:
    title: str
    level: int          # 1 = h1, 2 = h2, …
    page_start: int
    page_end: int | None
    section_id: str     # e.g. "s1.2"

@dataclass
class TextBlock:
    text: str
    bbox: BBox
    section_id: str     # which section it belongs to
    block_id: str       # stable unique id

@dataclass
class Caption:
    text: str
    bbox: BBox
    label: str          # "Figure 3" / "Table 2"

@dataclass
class Figure:
    label: str          # "Figure 3"
    caption: Caption | None
    bbox: BBox
    crop_path: str | None   # path to cropped PNG
    crop_url: str | None    # served URL

@dataclass
class Table:
    label: str
    caption: Caption | None
    bbox: BBox
    crop_path: str | None
    crop_url: str | None

@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: list[str]   # ["Introduction", "Motivation"]
    page: int
    bbox: BBox | None
    source_block_ids: list[str]
    token_count: int

@dataclass
class StructuredDocument:
    pdf_sha256: str
    title: str | None
    sections: list[Section]
    blocks: list[TextBlock]
    figures: list[Figure]
    tables: list[Table]
    chunks: list[Chunk]
```

**Impact on existing code**: None yet — models are additive.

---

## Phase 2 — Dual-Path PDF Extraction

**Goal**: Replace flat `get_text("text")` with structured block extraction that
preserves bbox, reading order, and block type.

### 2a — Text Path (bbox-aware)

**Modify**: `src/papersketch_backend/document/text.py`

Switch from `page.get_text("text")` to `page.get_text("dict")`.
PyMuPDF returns a block tree:

```
page → blocks → lines → spans
each block: { "type": 0=text/1=image, "bbox": (x0,y0,x1,y1),
              "lines": [ { "spans": [ { "text", "font", "size", "flags" } ] } ] }
```

Use `flags` (bold, italic) and font size to classify headings vs body text.

**New file**: `src/papersketch_backend/document/text_blocks.py`

Key functions:
- `extract_blocks(page) -> list[RawBlock]` — raw blocks with bbox + spans
- `classify_block(block, page_stats) -> BlockType` — heading/body/caption/other
  (heuristic: large font + bold = heading; small font below figure = caption)

### 2b — Layout Path (optional, progressive)

Render each page to PNG (already done in `document/render.py`), then run layout
detection.

**Start simple**: use PyMuPDF's own block detection (no ML dependency).
Later optionally plug in `surya` or `layoutparser` for better double-column
handling.

**New file**: `src/papersketch_backend/document/layout.py`

Key functions:
- `detect_layout_regions(page_png_path) -> list[LayoutRegion]`
  — each region: `{ type, bbox, confidence }`
- For initial implementation: delegate back to PyMuPDF block detection;
  swap in ML model later without changing the interface.

### 2c — Align & Reading Order

**New file**: `src/papersketch_backend/document/reading_order.py`

For double-column PDFs:
- Split page into left/right halves by x-midpoint
- Sort blocks: (column, y0) so left column is read before right column
- Merge across columns when a block spans full width (section title, abstract)

Key function: `sort_blocks_reading_order(blocks, page_width) -> list[RawBlock]`

### 2d — Section Tree Builder

**New file**: `src/papersketch_backend/document/sections.py`

Walk ordered blocks; when a heading block is seen, open a new `Section`.
Assign `section_id` to all subsequent body blocks until the next heading.

Key function: `build_section_tree(ordered_blocks) -> tuple[list[Section], list[TextBlock]]`

---

## Phase 3 — Figure and Table Extraction

**Goal**: Replace regex-heuristic figure page picking with actual figure/table
detection using bbox from the layout path.

**New file**: `src/papersketch_backend/document/figure_extraction.py`

Replaces: `src/papersketch_backend/document/figure_selection.py`
but mark deprecated).

### Steps

1. **Detect** figure/table bboxes from layout regions (type == "figure" or "table").
   Fallback: use PyMuPDF's image block list (`block["type"] == 1`).

2. **Match captions**: for each figure bbox, search the nearest text block
   below it (within ~50px) that matches `re.match(r"Figure\s*\d+", text)` or
   `re.match(r"Table\s*\d+", text)`.

3. **Crop images**: expand bbox by small margin, render at 200–300 DPI,
   save as `static/<sha256>/fig_<label>.png`.
   Reuse `render_pages_to_static` infrastructure but with sub-page crop.

4. **Return** `list[Figure]` and `list[Table]` with `crop_path`, `crop_url`,
   and linked `Caption`.

Key functions:
```python
def extract_figures_tables(
    pdf_path: str,
    sha256: str,
    layout_pages: list[list[LayoutRegion]],
    text_blocks: list[TextBlock],
    base_url: str,
    static_dir: str,
) -> tuple[list[Figure], list[Table]]
```

---

## Phase 4 — Paragraph-Based Chunking

**Goal**: Replace the 24k-char hard truncation with structure-aware chunks that
carry metadata for citation.

**New file**: `src/papersketch_backend/document/chunking.py`

### Strategy

```
for each TextBlock (in reading order):
    if block.token_count <= MAX_CHUNK_TOKENS:
        candidate = block
    else:
        split block by sentence boundary with OVERLAP tokens

    if candidate.token_count < MIN_CHUNK_TOKENS:
        merge with next block (same section)

    emit Chunk with:
        chunk_id   = sha256(pdf_sha256 + block_id)[:12]
        text       = candidate.text
        section_path = ancestor section titles
        page       = block.bbox.page
        bbox       = block.bbox
        token_count
```

**Constants** (tunable via env):
- `MAX_CHUNK_TOKENS = 512`
- `MIN_CHUNK_TOKENS = 64`
- `OVERLAP_TOKENS = 64`

**Token counting**: Use a simple word-count proxy (`len(text.split()) * 1.3`)
to avoid a hard tokenizer dependency. Swap in `tiktoken` later if needed.

**Caption chunks**: Captions are short but high-density — always emit as their
own chunk with `type="caption"`, linked to the figure/table `label`.

Key function:
```python
def chunk_document(
    blocks: list[TextBlock],
    sections: list[Section],
    figures: list[Figure],
    tables: list[Table],
    max_tokens: int = 512,
    min_tokens: int = 64,
    overlap_tokens: int = 64,
) -> list[Chunk]
```

---

## Phase 5 — Knowledge Artifacts (Summary & Sketch)

**Goal**: Generate two post-processing artifacts from the structured document,
with chunk-level citations in the summary.

**New module**: `src/papersketch_backend/knowledge/`

### 5a — Paper Summary (traceable)

**New file**: `src/papersketch_backend/knowledge/summary.py`

The LLM receives the structured chunks (not raw full text), organized by
section. The prompt asks the LLM to produce each bullet as:

```
- <claim> [chunk_id: abc123]
```

Sections in the summary output:
- One-sentence contribution
- Problem definition + method (5–10 bullets, each with `chunk_id`)
- Experiments + key results (metrics, comparisons — with `chunk_id`)
- Limitations & assumptions

The returned `PaperSummary` dataclass contains:
```python
@dataclass
class SummaryBullet:
    text: str
    chunk_ids: list[str]

@dataclass
class PaperSummary:
    one_liner: str
    problem_and_method: list[SummaryBullet]
    experiments_and_results: list[SummaryBullet]
    limitations: list[SummaryBullet]
    model_info: str
    llm_ms: int
    usage: LLMUsage
```

### 5b — Paper Sketch (navigable skeleton)

**New file**: `src/papersketch_backend/knowledge/sketch.py`

Produce a `PaperSketch` dataclass:
```python
@dataclass
class SectionSummary:
    section_id: str
    title: str
    level: int
    one_liner: str       # 1-2 sentence LLM summary of the section

@dataclass
class FigureEntry:
    label: str           # "Figure 3"
    caption_summary: str # short caption or LLM one-liner
    page: int
    crop_url: str | None

@dataclass
class TermEntry:
    term: str
    first_page: int
    chunk_id: str

@dataclass
class PaperSketch:
    title: str | None
    section_map: list[SectionSummary]    # the navigable TOC
    figure_index: list[FigureEntry]
    table_index: list[FigureEntry]
    key_terms: list[TermEntry]           # optional
```

**Generation strategy**: batch the section-by-section summaries in one LLM call
(cheaper than N calls) by passing structured section content in the prompt.

---

## Phase 6 — Pipeline & API Integration

### 6a — New `preprocess_paper` output

Extend `PreprocessResult` (or replace with `StructuredDocument`) to include:
- `sections`, `blocks`, `figures`, `tables`, `chunks`
- Keep `figure_md_lines` for backward compat with existing LLM summarize path

### 6b — New API endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/v1/papersketch_url` | unchanged (backward compat) |
| `GET /api/v1/papersketch_compare` | unchanged |
| `GET /api/v1/paper_structure` | `StructuredDocument` (sections, chunks, figures) |
| `GET /api/v1/paper_summary` | `PaperSummary` with chunk citations |
| `GET /api/v1/paper_sketch` | `PaperSketch` with section map + figure index |

**New route file**: `src/papersketch_backend/api/routes/structure.py`

### 6c — Storage / Caching

Add a JSON cache layer keyed by `pdf_sha256` so the structured document is not
re-extracted on every call.

**New file**: `src/papersketch_backend/cache.py`

```python
def load_cached(sha256: str, cache_dir: str) -> StructuredDocument | None
def save_cached(doc: StructuredDocument, cache_dir: str) -> None
```

---

## Implementation Order (Phased)

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| 1 | Data models (`document/models.py`) | None |
| 2a | Text path with bbox (`document/text_blocks.py`) | Phase 1 |
| 2c | Reading order (`document/reading_order.py`) | Phase 2a |
| 2d | Section tree (`document/sections.py`) | Phase 2c |
| 3 | Figure/table extraction (`document/figure_extraction.py`) | Phase 2a, 2d |
| 4 | Chunking (`document/chunking.py`) | Phase 2d, Phase 3 |
| 5a | Paper Summary (`knowledge/summary.py`) | Phase 4 |
| 5b | Paper Sketch (`knowledge/sketch.py`) | Phase 2d, Phase 3, Phase 4 |
| 6 | API endpoints + caching | All above |
| 2b | ML layout detection (optional upgrade) | Phase 2a interface |

Start with Phases 1–2a–2c–2d: this already eliminates the biggest gaps
(no bbox, broken double-column reading order) without any new dependencies.

---

## Files to Create

```
src/papersketch_backend/
├─ document/
│  ├─ __init__.py
│  ├─ models.py          # Phase 1  — all dataclasses
│  ├─ text_blocks.py     # Phase 2a — bbox-aware PyMuPDF extraction
│  ├─ layout.py          # Phase 2b — layout detection (start: PyMuPDF blocks)
│  ├─ reading_order.py   # Phase 2c — double-column sort
│  ├─ sections.py        # Phase 2d — section tree builder
│  ├─ figure_extraction.py # Phase 3  — figure/table detection + crop
│  └─ chunking.py        # Phase 4  — paragraph chunking with metadata
│
├─ knowledge/
│  ├─ __init__.py
│  ├─ summary.py         # Phase 5a — Paper Summary with chunk citations
│  └─ sketch.py          # Phase 5b — Paper Sketch (section map + figure index)
│
├─ cache.py              # Phase 6c — JSON cache keyed by sha256
└─ settings.py           # runtime model configuration helpers

src/papersketch_backend/api/routes/
└─ structure.py          # Phase 6b — new endpoints
```

## Files to Modify

```
src/papersketch_backend/document/text.py   # add bbox extraction (keep old API)
src/papersketch_backend/pipeline/preprocess.py  # use new extraction pipeline
src/papersketch_backend/pipeline/prompt.py      # use chunks not raw full_text
src/papersketch_backend/document/figure_selection.py # deprecate; proxy to structured extractor
src/papersketch_backend/api/app.py              # register structure router
```
