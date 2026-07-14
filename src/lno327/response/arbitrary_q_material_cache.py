"""Readonly q-independent material cache for fixed periodic BZ grids."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.response.arbitrary_q_formal_policy import MATERIAL_CACHE_SCHEMA
from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
)
from lno327.response.finite_q_optimized import FiniteQMaterialWorkspace
from lno327.response.periodic_bz_grid import PeriodicBZGrid


def _public_state(value: Any) -> dict[str, Any]:
    if not hasattr(value, "__dict__"):
        return {}
    return {
        key: item
        for key, item in vars(value).items()
        if not key.startswith("_")
    }


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("cache fingerprints reject non-finite floats")
        scalar = 0.0 if value == 0.0 else value
        return {"float_hex": scalar.hex()}
    if isinstance(value, complex):
        return {
            "complex": [
                _stable_value(float(value.real)),
                _stable_value(float(value.imag)),
            ]
        }
    if isinstance(value, np.generic):
        return _stable_value(value.item())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "array_dtype": array.dtype.str,
            "array_shape": list(array.shape),
            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if is_dataclass(value):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _stable_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]

    explicit = getattr(value, "cache_fingerprint_payload", None)
    if callable(explicit):
        return {
            "object": f"{type(value).__module__}.{type(value).__qualname__}",
            "explicit_cache_fingerprint_payload": _stable_value(explicit()),
        }

    metadata = getattr(value, "metadata", None)
    payload: dict[str, Any] = {
        "object": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    if callable(metadata):
        payload["metadata"] = _stable_value(metadata())
    public = _public_state(value)
    if public:
        payload["public_state"] = _stable_value(public)
    if len(payload) > 1:
        return payload
    return {
        **payload,
        "repr": repr(value),
    }


def material_cache_fingerprint(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    config: object,
    options: object,
    grid: PeriodicBZGrid,
) -> str:
    payload = {
        "schema": MATERIAL_CACHE_SCHEMA,
        "spec": _stable_value(spec),
        "ansatz": _stable_value(ansatz),
        "pairing": _stable_value(pairing),
        "config": {
            "temperature_eV": _stable_value(
                float(getattr(config, "temperature_eV"))
            ),
            "fermi_level_eV": _stable_value(
                float(getattr(config, "fermi_level_eV"))
            ),
            "eta_eV": _stable_value(float(getattr(config, "eta_eV"))),
            "output_si": bool(getattr(config, "output_si")),
            "omega_excluded_because_material_cache_is_frequency_independent": True,
        },
        "options": _stable_value(options),
        "grid_fingerprint": grid.fingerprint,
        "phase_vertex": str(getattr(ansatz, "phase_vertex", "unknown")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MaterialChunkView:
    """Zero-copy duck-typed view accepted by the batched q-workspace builder."""

    spec: object
    ansatz: object
    k_points: np.ndarray
    k_weights: np.ndarray
    config: object
    pairing_params: object
    options: object
    collective_mode: str
    collective_mode_disabled_reason: str | None
    midpoint_energies: np.ndarray
    midpoint_states: np.ndarray
    midpoint_occupations: np.ndarray
    collective_counterterm_matrix: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def nk(self) -> int:
        return int(self.k_points.shape[0])

    @property
    def nb(self) -> int:
        return int(self.midpoint_energies.shape[1])


@dataclass(frozen=True)
class MaterialGridCache:
    schema_version: str
    fingerprint: str
    grid: PeriodicBZGrid
    workspace: FiniteQMaterialWorkspace
    build_seconds: float
    build_count: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != MATERIAL_CACHE_SCHEMA:
            raise ValueError("unsupported material cache schema")
        if self.workspace.nk != self.grid.point_count:
            raise ValueError("material cache workspace/grid point counts differ")
        for name in (
            "k_points",
            "k_weights",
            "midpoint_energies",
            "midpoint_states",
            "midpoint_occupations",
            "collective_counterterm_matrix",
        ):
            if np.asarray(getattr(self.workspace, name)).flags.writeable:
                raise ValueError(f"material cache array {name} must be readonly")

    @property
    def counterterm(self) -> np.ndarray:
        return np.asarray(self.workspace.collective_counterterm_matrix, dtype=complex)

    def chunk_view(self, start: int, stop: int) -> MaterialChunkView:
        first, last = int(start), int(stop)
        if first < 0 or last > self.grid.point_count or first >= last:
            raise ValueError("invalid material cache chunk bounds")
        zero_counterterm = np.zeros((2, 2), dtype=complex)
        zero_counterterm.setflags(write=False)
        metadata = MappingProxyType(
            {
                **dict(self.workspace.metadata),
                "workspace_kind": "arbitrary_q_material_chunk_view",
                "parent_material_cache_fingerprint": self.fingerprint,
                "chunk_start": first,
                "chunk_stop": last,
                "counterterm_omitted_from_chunk": True,
            }
        )
        return MaterialChunkView(
            spec=self.workspace.spec,
            ansatz=self.workspace.ansatz,
            k_points=self.workspace.k_points[first:last],
            k_weights=self.workspace.k_weights[first:last],
            config=self.workspace.config,
            pairing_params=self.workspace.pairing_params,
            options=self.workspace.options,
            collective_mode=self.workspace.collective_mode,
            collective_mode_disabled_reason=self.workspace.collective_mode_disabled_reason,
            midpoint_energies=self.workspace.midpoint_energies[first:last],
            midpoint_states=self.workspace.midpoint_states[first:last],
            midpoint_occupations=self.workspace.midpoint_occupations[first:last],
            collective_counterterm_matrix=zero_counterterm,
            metadata=metadata,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "build_seconds": float(self.build_seconds),
            "build_count": int(self.build_count),
            "readonly": True,
            "grid": self.grid.metadata(),
            "midpoint_eigensystem_count": int(self.workspace.nk),
            "goldstone_counterterm_cached_once": True,
        }


def build_material_grid_cache(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    config: object,
    options: object,
    grid: PeriodicBZGrid,
) -> MaterialGridCache:
    fingerprint = material_cache_fingerprint(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        options=options,
        grid=grid,
    )
    started = perf_counter()
    workspace = precompute_finite_q_material_workspace_batched(
        spec,
        ansatz,
        grid.points,
        grid.weights,
        config,
        pairing,
        options,
    )
    build_seconds = perf_counter() - started
    return MaterialGridCache(
        schema_version=MATERIAL_CACHE_SCHEMA,
        fingerprint=fingerprint,
        grid=grid,
        workspace=workspace,
        build_seconds=float(build_seconds),
    )


__all__ = [
    "MaterialChunkView",
    "MaterialGridCache",
    "build_material_grid_cache",
    "material_cache_fingerprint",
]
