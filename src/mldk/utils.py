"""Utility helpers for file operations."""

from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory for the provided path exists."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj
