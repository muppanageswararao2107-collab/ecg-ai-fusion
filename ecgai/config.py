"""Configuration loading and path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (or the given file) and return it as a dict."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def path(cfg: dict[str, Any], key: str) -> Path:
    """Return an absolute path for cfg['paths'][key] (relative paths are relative to ROOT)."""
    p = Path(cfg["paths"][key]).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()
