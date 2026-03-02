# Flask comparison UI entrypoint.
from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
from typing import List, Dict, Any

from flask import Flask, render_template, request

from papersketch_backend.evaluation import evaluate_compare_results
from papersketch_backend.pipeline.preprocess import preprocess_paper
from papersketch_backend.pipeline.summarize import summarize_with_model
from papersketch_backend.settings import get_model_list


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            models=get_model_list(),
            default_lang=os.getenv("DEFAULT_LANG", "en"),
            default_url=os.getenv("DEFAULT_PAPER_URL", ""),
            default_judge_model=os.getenv("PAPERSKETCH_JUDGE_MODEL", ""),
        )

    @app.post("/run")
    def run():
        url = (request.form.get("url") or "").strip()
        lang = (request.form.get("lang") or "en").strip()
        selection = (request.form.get("model") or "ALL").strip()

        # preprocessing / prompt knobs (optional fields in the form; safe defaults)
        max_pages = int(request.form.get("max_pages") or 30)
        max_chars = int(request.form.get("max_chars") or 24000)
        reference_summary = (request.form.get("reference_summary") or "").strip()
        judge_model = (request.form.get("judge_model") or os.getenv("PAPERSKETCH_JUDGE_MODEL") or "").strip()
        render_dpi = int(request.form.get("render_dpi") or 200)

        if not url:
            return render_template(
                "result.html",
                url=url,
                lang=lang,
                selection=selection,
                shared=None,
                results=[],
                error="Missing URL.",
                primary_sketch="",
            )

        models = get_model_list()
        run_models = models if selection == "ALL" else [selection]

        # ---- shared preprocess (one time) ----
        try:
            ctx = preprocess_paper(
                url=url,
                max_pages=max_pages,
                render_dpi=render_dpi,
                overwrite_images=False,
            )
        except Exception as e:
            return render_template(
                "result.html",
                url=url,
                lang=lang,
                selection=selection,
                shared=None,
                results=[],
                error=f"Preprocess failed: {e}",
                primary_sketch="",
            )

        shared: Dict[str, Any] = {
            "preprocess_ms": ctx.preprocess_ms,
            "timings_ms": ctx.timings_ms,
            "pdf_final_url": ctx.downloaded.final_url,
            "pdf_sha256": ctx.downloaded.sha256,
            "pdf_size_bytes": ctx.downloaded.size_bytes,
            "title_guess": ctx.document.title,
            "pages_extracted": len({b.bbox.page for b in ctx.document.blocks}),
            "chars_used": sum(len(chunk.text) for chunk in ctx.document.chunks if chunk.chunk_type != "caption"),
            "figure_pages": sorted({f.bbox.page + 1 for f in ctx.document.figures}),
            "figure_urls": [f.crop_url for f in ctx.document.figures if f.crop_url],
        }

        # ---- per-model LLM step ----
        results: List[Dict[str, Any]] = []
        primary_sketch = ""
        for mspec in run_models:
            try:
                out = summarize_with_model(
                    ctx=ctx,
                    lang=lang,
                    model_spec=mspec,
                    max_context_chars=max_chars,
                )
                total_ms = ctx.preprocess_ms + (out.get("llm_ms") or 0)
                row = {
                    "model": mspec,
                    "modelInfo": out.get("modelInfo", mspec),
                    "status": "OK",
                    "error": None,
                    "llm_ms": out.get("llm_ms"),
                    "preprocess_ms": ctx.preprocess_ms,
                    "total_ms": total_ms,
                    "usage": out.get("usage") or {},
                    "paperSketch": out.get("paperSketch", ""),
                }
                results.append(row)
                if not primary_sketch and row["paperSketch"]:
                    primary_sketch = row["paperSketch"]
            except Exception as e:
                results.append(
                    {
                        "model": mspec,
                        "modelInfo": mspec,
                        "status": "ERROR",
                        "error": str(e),
                        "llm_ms": None,
                        "preprocess_ms": ctx.preprocess_ms,
                        "total_ms": None,
                        "usage": {"prompt_tokens": None, "output_tokens": None, "total_tokens": None},
                        "paperSketch": "",
                    }
                )

        evaluation = evaluate_compare_results(
            doc=ctx.document,
            results=results,
            reference_summary=reference_summary,
            judge_model=judge_model or None,
        )

        return render_template(
            "result.html",
            url=url,
            lang=lang,
            selection=selection,
            shared=shared,
            results=results,
            evaluation=evaluation,
            error=None,
            primary_sketch=primary_sketch,
        )

    return app


app = create_app()


if __name__ == "__main__":
    # Run:
    #   export FLASK_UI_PORT=5000
    #   PYTHONPATH=src python -m papersketch_backend.ui.app
    #
    # Make sure your env vars are set (OPENAI_API_KEY / GEMINI_API_KEY, etc.)
    port = int(os.getenv("FLASK_UI_PORT", "5055"))
    app.run(host="0.0.0.0", port=port, debug=True)
