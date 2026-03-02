from __future__ import annotations

import time
from typing import Any, Dict

from papersketch_backend.llm.registry import resolve
from papersketch_backend.pipeline.prompt import build_prompt
from papersketch_backend.pipeline.preprocess import ProcessedPaper


def summarize_with_model(
    *,
    ctx: ProcessedPaper,
    lang: str,
    model_spec: str,
    max_context_chars: int = 24_000,
) -> Dict[str, Any]:
    """
    Pure LLM step (measured):
      - build prompt from shared preprocess ctx
      - call chosen LLM
      - return paperSketch + usage + llm_ms + modelInfo
    """
    client, provider, model_name = resolve(model_spec)

    prompt = build_prompt(
        url=ctx.downloaded.final_url,
        lang=lang,
        document=ctx.document,
        max_context_chars=max_context_chars,
    )

    t0 = time.perf_counter()
    llm_resp = client.summarize(prompt=prompt, model=model_name)
    llm_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "paperSketch": llm_resp.text,
        "modelInfo": f"{provider}:{model_name}",
        "llm_ms": llm_ms,
        "usage": {
            "prompt_tokens": llm_resp.usage.prompt_tokens,
            "output_tokens": llm_resp.usage.output_tokens,
            "total_tokens": llm_resp.usage.total_tokens,
        },
        "meta": {
            "provider": provider,
            "model": model_name,
        },
    }
