from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class LLMUsage:
    """
    Normalized token usage across providers.

    Some providers may not return token usage. In that case, fields can be None.
    """
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMResponse:
    """
    Normalized response across providers.
    """
    text: str
    provider: str
    model: str
    usage: LLMUsage = LLMUsage()
    raw: Optional[Dict[str, Any]] = None  # provider-native payload (optional)


class LLMClient:
    """
    Provider-agnostic interface your pipeline will call.

    Implementations:
      - OpenAIClient in openai_client.py
      - GeminiClient in gemini_client.py
    """

    def summarize(self, *, prompt: str, model: str) -> LLMResponse:
        raise NotImplementedError("LLMClient.summarize must be implemented by subclasses.")
