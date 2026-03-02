from __future__ import annotations

import os
from typing import List


def _parse_csv(env_value: str) -> List[str]:
    return [m.strip() for m in env_value.split(",") if m.strip()]


def get_model_list() -> List[str]:
    """
    Models available for comparison / UI dropdown.

    Env:
      PAPERSKETCH_MODELS="openai:...,gemini:...,deepseek:..."
    """
    env = (os.getenv("PAPERSKETCH_MODELS") or "").strip()
    if env:
        return _parse_csv(env)

    # Safe defaults (used only if env not set)
    return [
        "openai:gpt-4o-mini",
        "gemini:gemini-2.0-flash",
        "deepseek:deepseek-chat",
    ]


def get_default_model() -> str:
    """
    Default single-model choice.

    Env:
      PAPERSKETCH_DEFAULT_MODEL="openai:gpt-4o-mini"
    """
    return os.getenv("PAPERSKETCH_DEFAULT_MODEL", get_model_list()[0])
