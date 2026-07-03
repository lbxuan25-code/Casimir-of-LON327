"""I/O helpers for model sanity plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent_dir(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_metadata_json(path, metadata: dict) -> None:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    payload = {
        **metadata,
        "sanity_plot_only": True,
        "not_casimir_input": True,
    }
    output_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def save_figure(fig, output_path, metadata: dict | None = None) -> None:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    ensure_parent_dir(path)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if metadata is not None:
        write_metadata_json(path.with_suffix(".json"), metadata)
