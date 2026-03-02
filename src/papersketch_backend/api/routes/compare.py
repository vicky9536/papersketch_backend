from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Header

from papersketch_backend.api.deps import require_api_key
from papersketch_backend.evaluation import evaluate_compare_results
from papersketch_backend.pipeline.preprocess import preprocess_paper
from papersketch_backend.pipeline.summarize import summarize_with_model
from papersketch_backend.settings import get_model_list

router = APIRouter()


def parse_models(models_csv: str | None) -> list[str]:
    if models_csv:
        return [m.strip() for m in models_csv.split(",") if m.strip()]
    return get_model_list()


@router.get("/papersketch_compare")
def papersketch_compare(
    url: str = Query(..., description="PDF or arXiv URL"),
    lang: str = Query("en", description="Output language"),
    models: str | None = Query(
        default=None,
        description="Comma-separated model specs: openai:gpt-4o-mini,gemini:gemini-1.5-pro",
    ),
    max_pages: int = Query(30, ge=1, le=60, description="Max PDF pages to extract"),
    max_chars: int = Query(24_000, ge=5_000, le=300_000, description="Max chars of structured content sent to each LLM"),
    reference_summary: str | None = Query(default=None, description="Optional gold/reference summary for ROUGE-style comparison"),
    judge_model: str | None = Query(default=None, description="Optional stronger judge model, e.g. openai:gpt-4o"),
    render_dpi: int = Query(200, ge=72, le=300, description="DPI for rendered page images"),
    overwrite_images: bool = Query(False, description="Re-render images even if cached on disk"),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Compare multiple LLMs fairly:
      - preprocess ONCE (download/extract/cache)
      - run ONLY the LLM step for each model (measure llm_ms + tokens)
      - return shared preprocess_ms and per-model metrics
    """
    require_api_key(x_api_key)

    model_list = parse_models(models)

    # ---- shared preprocess (one time) ----
    ctx = preprocess_paper(
        url=url,
        max_pages=max_pages,
        render_dpi=render_dpi,
        overwrite_images=overwrite_images,
    )

    results: list[dict[str, Any]] = []
    for model_spec in model_list:
        status = "OK"
        err_msg = None

        try:
            llm_out = summarize_with_model(
                ctx=ctx,
                lang=lang,
                model_spec=model_spec,
                max_context_chars=max_chars,
            )
        except Exception as e:
            status = "ERROR"
            err_msg = str(e)
            llm_out = {
                "paperSketch": "",
                "modelInfo": model_spec,
                "llm_ms": None,
                "usage": {"prompt_tokens": None, "output_tokens": None, "total_tokens": None},
                "meta": {},
            }

        # Total time is shared preprocess + per-model LLM
        total_ms = None
        if llm_out.get("llm_ms") is not None:
            total_ms = int(ctx.preprocess_ms + llm_out["llm_ms"])

        results.append(
            {
                "model": model_spec,
                "modelInfo": llm_out.get("modelInfo", model_spec),
                "status": status,
                "error": err_msg,
                "preprocess_ms": ctx.preprocess_ms,
                "llm_ms": llm_out.get("llm_ms"),
                "latency_ms": total_ms,
                "usage": llm_out.get("usage") or {},
                "paperSketch": llm_out.get("paperSketch", ""),
                "meta": llm_out.get("meta") or {},
            }
        )

    evaluation = evaluate_compare_results(
        doc=ctx.document,
        results=results,
        reference_summary=reference_summary,
        judge_model=judge_model,
    )

    return {
        "paper": {"url": url, "lang": lang},
        "shared": {
            "preprocess_ms": ctx.preprocess_ms,
            "timings_ms": ctx.timings_ms,
            "pdf_sha256": ctx.downloaded.sha256,
            "pdf_size_bytes": ctx.downloaded.size_bytes,
            "pdf_final_url": ctx.downloaded.final_url,
            "title_guess": ctx.document.title,
            "pages_extracted": len({b.bbox.page for b in ctx.document.blocks}),
            "chars_used": sum(len(chunk.text) for chunk in ctx.document.chunks if chunk.chunk_type != "caption"),
            "figure_pages": sorted({f.bbox.page + 1 for f in ctx.document.figures}),
            "figure_urls": [f.crop_url for f in ctx.document.figures if f.crop_url],
        },
        "results": results,
        "evaluation": evaluation,
        "version": "0.1.0",
    }
