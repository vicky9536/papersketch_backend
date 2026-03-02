from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from google import genai  # pip install google-genai

from .base import LLMClient, LLMResponse, LLMUsage


class GeminiClient(LLMClient):
    """
    Gemini (Google) LLM adapter using the Google Gen AI SDK.

    Environment variable recommended by Google docs:
      GEMINI_API_KEY

    Install:
      pip install -U google-genai
    :contentReference[oaicite:1]{index=1}
    """

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Please set it in your environment or .env file."
            )

        # The client can also pick up GEMINI_API_KEY automatically from env,
        # but passing it explicitly makes behavior clearer.
        self.client = genai.Client(api_key=api_key)

    def summarize(self, *, prompt: str, model: str) -> LLMResponse:
        """
        Generate a summary using a Gemini model.

        Args:
            prompt: Full prompt string (already assembled).
            model: Gemini model name, e.g. 'gemini-2.0-flash', 'gemini-1.5-pro', etc.

        Returns:
            LLMResponse with normalized text + best-effort token usage.
        """
        t0 = time.perf_counter()

        # For simple text generation, docs show:
        # response = client.models.generate_content(model="...", contents="...")
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        text = getattr(response, "text", "") or ""

        # Best-effort usage extraction:
        # Some Gemini/Vertex responses expose `usage_metadata` with token counts.
        # If it's missing, keep None (your compare table can show NA).
        usage_md = getattr(response, "usage_metadata", None)

        prompt_tokens = None
        output_tokens = None
        total_tokens = None

        if usage_md is not None:
            # The exact field names can vary by backend/version, so handle safely.
            prompt_tokens = getattr(usage_md, "prompt_token_count", None) or getattr(
                usage_md, "input_tokens", None
            )
            output_tokens = getattr(usage_md, "candidates_token_count", None) or getattr(
                usage_md, "output_tokens", None
            )
            total_tokens = getattr(usage_md, "total_token_count", None) or getattr(
                usage_md, "total_tokens", None
            )

        llm_usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        raw: Dict[str, Any] = {"latency_ms": latency_ms}
        # Include minimal identifiers if present
        for k in ("model_version", "response_id"):
            if hasattr(response, k):
                raw[k] = getattr(response, k)

        return LLMResponse(
            text=text,
            provider="gemini",
            model=model,
            usage=llm_usage,
            raw=raw,
        )
