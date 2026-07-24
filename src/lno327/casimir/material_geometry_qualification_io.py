"""Auditable filesystem helpers for TODO 4 diagnostic qualification artifacts."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        jsonable(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, content)


def write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        jsonable(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    destination = Path(path)
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError(
                f"refusing to replace a different frozen artifact: {destination}"
            )
        return
    atomic_write_bytes(destination, content)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def tracked_tree_clean() -> bool:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet"],
        check=False,
    ).returncode
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        check=False,
    ).returncode
    return unstaged == 0 and staged == 0


def slug(value: str) -> str:
    return str(value).replace("/", "__").replace(":", "_")


def point_token(point_id: str) -> str:
    return hashlib.sha256(str(point_id).encode("utf-8")).hexdigest()[:20]


def load_json(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"required qualification artifact is missing: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "canonical_json",
    "jsonable",
    "load_json",
    "payload_sha256",
    "point_token",
    "slug",
    "source_commit",
    "tracked_tree_clean",
    "write_frozen_json",
]
