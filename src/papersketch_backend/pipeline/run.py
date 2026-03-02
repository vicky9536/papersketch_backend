from __future__ import annotations

from typing import Any, Dict

from papersketch_backend.pipeline.preprocess import preprocess_paper
from papersketch_backend.pipeline.summarize import summarize_with_model


def run_single(
    *,
    url: str,
    lang: str,
    model_spec: str,
    max_pages: int = 30,
    max_chars: int = 24_000,
    render_dpi: int = 200,
    overwrite_images: bool = False,
) -> Dict[str, Any]:
    """
    Single-model pipeline:
      preprocess once (download/extract/cache) + one LLM call

    Returns:
      - paperSketch (markdown)
      - modelInfo
      - preprocess_ms, llm_ms, latency_ms
      - usage (tokens when available)
      - meta (timing breakdown + figure picks + pdf info)
    """
    # Preprocess (shared part)
    ctx = preprocess_paper(
        url=url,
        max_pages=max_pages,
        render_dpi=render_dpi,
        overwrite_images=overwrite_images,
    )

    # LLM-only step
    llm_out = summarize_with_model(
        ctx=ctx,
        lang=lang,
        model_spec=model_spec,
        max_context_chars=max_chars,
    )

    total_ms = int(ctx.preprocess_ms + llm_out["llm_ms"])

    return {
        "paperSketch": llm_out["paperSketch"],
        "version": "0.1.0",
        "modelInfo": llm_out["modelInfo"],
        "preprocess_ms": ctx.preprocess_ms,
        "llm_ms": llm_out["llm_ms"],
        "latency_ms": total_ms,
        "usage": llm_out["usage"],
        "meta": {
            # preprocess breakdown
            **ctx.timings_ms,
            # llm identity
            **(llm_out.get("meta") or {}),
            # pdf info
            "pdf_sha256": ctx.downloaded.sha256,
            "pdf_size_bytes": ctx.downloaded.size_bytes,
            "pdf_final_url": ctx.downloaded.final_url,
            "title_guess": ctx.document.title,
            "pages_extracted": len({b.bbox.page for b in ctx.document.blocks}),
            "chars_used": sum(len(chunk.text) for chunk in ctx.document.chunks if chunk.chunk_type != "caption"),
            "figure_pages": sorted({f.bbox.page + 1 for f in ctx.document.figures}),
            "figure_urls": [f.crop_url for f in ctx.document.figures if f.crop_url],
        },
    }
