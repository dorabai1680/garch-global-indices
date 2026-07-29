"""Small helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def print_section(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n{title}\n{line}")


def df_to_markdown_preview(df: pd.DataFrame, max_rows: int = 12) -> str:
    try:
        return df.head(max_rows).to_string()
    except Exception:  # noqa: BLE001
        return str(df)
