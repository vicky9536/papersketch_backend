# src/papersketch_backend/llm/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "gemini", "deepseek"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str


def parse_model_spec(spec: str) -> ModelSpec:
    """
    Accepts:
      - "openai:gpt-4o-mini"
      - "gemini:1.5-pro"
      - "gpt-4o-mini"  (defaults to openai)
    """
    s = (spec or "").strip()
    if not s:
        return ModelSpec(provider="openai", model="gpt-4o-mini")

    if ":" not in s:
        return ModelSpec(provider="openai", model=s)

    provider_raw, model = s.split(":", 1)
    provider = provider_raw.strip().lower()
    model = model.strip()

    if provider not in ("openai", "gemini", "deepseek"):
        raise ValueError(f"Unknown provider '{provider}'. Use 'openai:', 'gemini:' or 'deepseek'.")

    if not model:
        raise ValueError("Model name is empty. Example: openai:gpt-4o-mini")

    return ModelSpec(provider=provider, model=model)  # type: ignore[arg-type]


def get_client(provider: Provider):
    """
    Lazy-import clients so you can install provider SDKs incrementally.
    """
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient()

    if provider == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient()
    
    if provider == "deepseek":
        from .deepseek_client import DeepSeekClient
        return DeepSeekClient()

    # Should never happen due to Provider typing
    raise ValueError(f"Unsupported provider: {provider}")


def resolve(spec: str):
    """
    Convenience: parse and return (client, provider, model_name).
    """
    ms = parse_model_spec(spec)
    client = get_client(ms.provider)
    return client, ms.provider, ms.model
