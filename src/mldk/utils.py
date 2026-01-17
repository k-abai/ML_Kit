"""Utility helpers for file operations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
import importlib.util


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory for the provided path exists."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def now_ts() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def load_plugin(plugin_dir: str) -> ModuleType:
    """Load a custom plugin module from the provided directory."""
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists() or not plugin_path.is_dir():
        raise ValueError(f"Plugin directory '{plugin_dir}' does not exist.")

    plugin_file = plugin_path / "plugin.py"
    if not plugin_file.exists():
        raise ValueError(f"Plugin file '{plugin_file}' was not found.")

    spec = importlib.util.spec_from_file_location("mldk_custom_plugin", plugin_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load plugin from '{plugin_file}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr in ("train", "predict", "evaluate"):
        fn = getattr(module, attr, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"Plugin '{plugin_file}' must define callable '{attr}'.")

    return module
