from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict

from papersketch_backend.cache import load_cached, save_cached
from papersketch_backend.document.fetch import DownloadedPDF, download_pdf
from papersketch_backend.document.models import StructuredDocument
from papersketch_backend.document.text_blocks import extract_all_blocks
from papersketch_backend.document.reading_order import sort_document_reading_order
from papersketch_backend.document.sections import build_section_tree
from papersketch_backend.document.figure_extraction import extract_figures_tables
from papersketch_backend.document.chunking import chunk_document
from papersketch_backend.document.layout import detect_document_layout


@dataclass(frozen=True)
class ProcessedPaper:
    downloaded: DownloadedPDF
    document: StructuredDocument
    preprocess_ms: int
    timings_ms: Dict[str, int]


def _infer_base_url() -> str:
    base = (os.getenv("BASE_URL") or "").strip().rstrip("/")
    return base or "http://127.0.0.1:8001"


def _get_cache_dir() -> str:
    return os.getenv("PAPERSKETCH_CACHE_DIR", "cache")


def _build_structured_document(
    downloaded: DownloadedPDF,
    *,
    max_pages: int = 30,
    render_dpi: int = 200,
    overwrite_images: bool = False,
) -> StructuredDocument:
    base_url = _infer_base_url()
    static_dir = os.getenv("STATIC_DIR", "static")

    raw_blocks = extract_all_blocks(
        downloaded.path,
        downloaded.sha256,
        max_pages=max_pages,
    )
    layout_pages = detect_document_layout(downloaded.path, max_pages=max_pages)
    ordered = sort_document_reading_order(raw_blocks)
    sections, text_blocks = build_section_tree(ordered)
    figures, tables = extract_figures_tables(
        pdf_path=downloaded.path,
        pdf_sha256=downloaded.sha256,
        raw_blocks=raw_blocks,
        layout_pages=layout_pages,
        base_url=base_url,
        static_dir=static_dir,
        dpi=render_dpi,
        overwrite=overwrite_images,
    )
    chunks = chunk_document(text_blocks, sections, figures, tables)

    title = None
    page0_headings = [b for b in raw_blocks if b.page == 0 and b.block_type == "heading"]
    if page0_headings:
        title = max(page0_headings, key=lambda b: b.avg_font_size).text.strip()
    else:
        page0_body = [b for b in raw_blocks if b.page == 0 and b.block_type == "body"]
        if page0_body:
            title = page0_body[0].text.split("\n")[0].strip()[:200]

    return StructuredDocument(
        pdf_sha256=downloaded.sha256,
        title=title,
        sections=sections,
        blocks=text_blocks,
        figures=figures,
        tables=tables,
        chunks=chunks,
    )


def preprocess_paper(
    *,
    url: str,
    max_pages: int = 30,
    render_dpi: int = 200,
    overwrite_images: bool = False,
    use_cache: bool = True,
) -> ProcessedPaper:
    """
    Unified preprocessing pipeline for all endpoints.

    Steps:
      - download PDF
      - load StructuredDocument from cache when available
      - otherwise run the structured extraction pipeline and save it

    Returns the downloaded PDF metadata, the structured document, and timing
    information for download/cache/extraction.
    """
    t0_total = time.perf_counter()
    timings: Dict[str, int] = {}

    t0 = time.perf_counter()
    downloaded = download_pdf(url)
    timings["download_ms"] = int((time.perf_counter() - t0) * 1000)

    cache_dir = _get_cache_dir()
    document = None

    if use_cache:
        t0 = time.perf_counter()
        document = load_cached(downloaded.sha256, cache_dir)
        timings["cache_read_ms"] = int((time.perf_counter() - t0) * 1000)

    if document is None:
        t0 = time.perf_counter()
        document = _build_structured_document(
            downloaded,
            max_pages=max_pages,
            render_dpi=render_dpi,
            overwrite_images=overwrite_images,
        )
        timings["extract_ms"] = int((time.perf_counter() - t0) * 1000)

        if use_cache:
            t0 = time.perf_counter()
            save_cached(document, cache_dir)
            timings["cache_write_ms"] = int((time.perf_counter() - t0) * 1000)
    else:
        timings["extract_ms"] = 0

    preprocess_ms = int((time.perf_counter() - t0_total) * 1000)

    return ProcessedPaper(
        downloaded=downloaded,
        document=document,
        preprocess_ms=preprocess_ms,
        timings_ms=timings,
    )
