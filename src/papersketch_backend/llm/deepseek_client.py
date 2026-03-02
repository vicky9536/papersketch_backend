from __future__ import annotations

import os
import time
from typing import Any, Dict

from openai import OpenAI

from .base import LLMClient, LLMResponse, LLMUsage


class DeepSeekClient(LLMClient):
    """
    DeepSeek adapter using the OpenAI-compatible API.

    DeepSeek docs:
      - OpenAI-compatible base_url: https://api.deepseek.com/v1
      - Auth: Bearer API key
    :contentReference[oaicite:2]{index=2}

    Env var:
      DEEPSEEK_API_KEY

    Typical model names:
      - deepseek-chat
      - deepseek-reasoner
    :contentReference[oaicite:3]{index=3}
    """

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Please set it in your environment or .env file."
            )

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def summarize(self, *, prompt: str, model: str) -> LLMResponse:
        """
        Uses OpenAI-compatible Chat Completions endpoint.

        Note: Some DeepSeek models may return extra fields like reasoning_content;
        we only use final message content here.
        :contentReference[oaicite:4]{index=4}
        """
        t0 = time.perf_counter()

        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic assistant. "
                        "You produce concise, structured paper summaries in markdown format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        text = (resp.choices[0].message.content or "").strip()

        usage = resp.usage
        llm_usage = LLMUsage(
            prompt_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )

        return LLMResponse(
            text=text,
            provider="deepseek",
            model=model,
            usage=llm_usage,
            raw={
                "latency_ms": latency_ms,
                "id": getattr(resp, "id", None),
                "object": getattr(resp, "object", None),
            },
        )
