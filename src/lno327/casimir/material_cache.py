"""Point-level cache helpers for real-material response/reflection grids."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import uuid
from typing import Any

import numpy as np

REUSABLE_CACHE_STATUSES = {"PASS", "MONITOR"}
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def to_jsonable(value: Any) -> Any:
    """Convert numpy/complex values into stable JSON-compatible objects."""

    if isinstance(value, complex | np.complexfloating):
        return {"re": float(np.real(value)), "im": float(np.imag(value)), "abs": float(abs(value))}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def from_jsonable_complex(value: Any) -> Any:
    """Restore complex objects from the cache JSON convention where possible."""

    if isinstance(value, dict) and {"re", "im"}.issubset(value):
        return complex(float(value["re"]), float(value["im"]))
    if isinstance(value, dict):
        return {key: from_jsonable_complex(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_jsonable_complex(item) for item in value]
    return value


def cache_filename_for_point_id(point_id: str) -> str:
    """Return a readable, reproducible cache filename for a point id."""

    safe = _SAFE_FILENAME_RE.sub("_", point_id).strip("._")
    if not safe:
        raise ValueError("point_id must contain at least one filename-safe character")
    return f"{safe}.json"


def cache_path_for_point(cache_dir: Path, point_id: str) -> Path:
    return Path(cache_dir) / cache_filename_for_point_id(point_id)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write complete JSON via a same-directory temp file and atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_point_cache(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def is_reusable_cache(
    payload: dict[str, Any] | None,
    *,
    point_id: str,
    response_config: dict[str, Any],
    lattice_convention: dict[str, Any],
) -> bool:
    if not payload:
        return False
    return (
        payload.get("point_id") == point_id
        and payload.get("response_config") == response_config
        and payload.get("lattice_convention") == lattice_convention
        and payload.get("status") in REUSABLE_CACHE_STATUSES
    )


def load_reusable_point_cache(
    cache_dir: Path,
    *,
    point_id: str,
    response_config: dict[str, Any],
    lattice_convention: dict[str, Any],
) -> dict[str, Any] | None:
    payload = read_point_cache(cache_path_for_point(cache_dir, point_id))
    if not is_reusable_cache(
        payload,
        point_id=point_id,
        response_config=response_config,
        lattice_convention=lattice_convention,
    ):
        return None
    return restore_runtime_row_from_cache(payload)


def write_point_cache(
    cache_dir: Path,
    row: dict[str, Any],
    *,
    response_config: dict[str, Any],
    lattice_convention: dict[str, Any],
) -> Path:
    point_id = str(row["point_id"])
    payload = {
        **row,
        "point_id": point_id,
        "response_config": dict(response_config),
        "lattice_convention": dict(lattice_convention),
    }
    path = cache_path_for_point(cache_dir, point_id)
    atomic_write_json(path, payload)
    return path


def restore_runtime_row_from_cache(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore cached rows enough for existing numpy-based summaries."""

    row = from_jsonable_complex(payload)
    for key in (
        "response_matrix",
        "sigma_model_xy",
        "sigma_tilde_xy",
        "sigma_tilde_LT",
        "reflection_tangential_E_LT",
        "reflection_TE_TM",
    ):
        if key in row:
            row[key] = np.asarray(row[key], dtype=complex)
    return row
