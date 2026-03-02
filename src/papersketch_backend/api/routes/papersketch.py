from __future__ import annotations

from fastapi import APIRouter, Query, Header

from papersketch_backend.api.deps import require_api_key
from papersketch_backend.pipeline.run import run_single
from papersketch_backend.settings import get_default_model


router = APIRouter()


@router.get("/papersketch_url")
def papersketch_url(
    url: str = Query(..., description="PDF or arXiv URL"),
    lang: str = Query("en", description="Output language: en / zh / ch"),
    model: str = Query(
        get_default_model(),
        description="Model spec, e.g. openai:gpt-4o-mini or gemini:gemini-2.0-flash",
    ),
    max_pages: int = Query(30, ge=1, le=60, description="Max PDF pages to extract"),
    max_chars: int = Query(24_000, ge=5_000, le=300_000, description="Max chars of structured content sent to the LLM"),
    x_api_key: str | None = Header(default=None),
):
    """
    Main endpoint your connector should call.

    Pipeline:
      URL -> unified structured extraction -> build prompt -> call selected LLM -> markdown output
    """
    require_api_key(x_api_key)

    result = run_single(
        url=url,
        lang=lang,
        model_spec=model,
        max_pages=max_pages,
        max_chars=max_chars,
    )
    return result
