from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from .base import LLMClient, LLMResponse, LLMUsage


class OpenAIClient(LLMClient):
    """
    OpenAI LLM adapter.

    Environment variable required:
      OPENAI_API_KEY
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Please set it in your environment or .env file."
            )

        # OpenAI Python SDK (new style client)
        self.client = OpenAI(api_key=api_key)

    def summarize(self, *, prompt: str, model: str) -> LLMResponse:
        """
        Generate a summary using an OpenAI chat model.

        Args:
            prompt: Full prompt string (already assembled).
            model: Model name, e.g. 'gpt-4o-mini', 'gpt-4.1-mini'.

        Returns:
            LLMResponse with normalized text + token usage.
        """
        t0 = time.perf_counter()

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic assistant. "
                        "You produce concise, structured paper summaries "
                        "in markdown format."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        choice = response.choices[0]
        text = choice.message.content or ""

        usage = response.usage
        llm_usage = LLMUsage(
            prompt_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )

        return LLMResponse(
            text=text,
            provider="openai",
            model=model,
            usage=llm_usage,
            raw={
                "latency_ms": latency_ms,
                "id": response.id,
                "object": response.object,
            },
        )
