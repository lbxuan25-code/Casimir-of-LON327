"""Stable hashing helpers for scientific and execution run identities."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

_EXECUTION_ONLY_KEYS = frozenset(
    {
        "workers",
        "parallel_mode",
        "memory_budget_gb",
        "max_context_workers",
        "certifier_q_batch_size",
        "point_cache_path",
        "transverse_checkpoint_path",
    }
)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    """Return the SHA-256 digest of canonical JSON data."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _strip_execution_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_execution_fields(item)
            for key, item in value.items()
            if str(key) not in _EXECUTION_ONLY_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_strip_execution_fields(item) for item in value]
    return value


def scientific_config_payload(config_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove execution-only fields from a serialized full-Casimir config.

    The returned payload is the resume identity. Worker count, scheduling mode,
    memory allocation, batch size and cache path may change between attempts;
    physical and numerical acceptance inputs may not.
    """

    stripped = _strip_execution_fields(config_payload)
    if not isinstance(stripped, dict):  # pragma: no cover - defensive
        raise TypeError("serialized config must remain a JSON object")
    return stripped


def scientific_config_sha256(config_payload: Mapping[str, Any]) -> str:
    return sha256_json(scientific_config_payload(config_payload))


__all__ = [
    "canonical_json_bytes",
    "scientific_config_payload",
    "scientific_config_sha256",
    "sha256_json",
]
