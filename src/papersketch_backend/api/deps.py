from __future__ import annotations

import os

from fastapi import HTTPException


def require_api_key(x_api_key: str | None) -> None:
    """Validate the shared API key when configured."""
    expected = os.getenv("PAPERSKETCH_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
