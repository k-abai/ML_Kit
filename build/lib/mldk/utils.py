"""Utility helpers for file operations."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory for the provided path exists."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def now_ts() -> str:
    """
    Return a filesystem-friendly timestamp string (UTC).
    Example: '2026-01-17T12-34-56Z'
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
