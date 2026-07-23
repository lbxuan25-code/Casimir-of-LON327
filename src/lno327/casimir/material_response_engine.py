"""Geometry-free finite-temperature material-response certification engine.

This is the clean TODO 2 orchestration path. It accepts crystal-frame momentum,
material/model policy, Matsubara frequencies, and BZ convergence controls. It
never accepts plate angles, laboratory momentum, separation, reflection policy,
or outer quadrature state.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.material_response import (
    MaterialResponsePolicy,
    MaterialResponseSample,
    build_material_response_sample,
)
from lno327.casimir.material_response_certification import (
    CertifiedMaterialResponse,
    MaterialResponseConvergencePolicy,
    MaterialResponseLevelRecord,
    assess_material_response_level,
    certify_material_response_history,
)
from lno327.casimir.matsubara import matsubara_energy_eV
from lno327.casimir.microscopic_model import get_finite_q_microscopic_model
from lno327.workflows.arbitrary_q_matsubara import integrate_arbitrary_q_periodic_bz

MATERIAL_RESPONSE_ENGINE_SCHEMA = "material-response-engine-v1"
DEFAULT_RESPONSE_SHIFTS = ((0.5, 0.5), (0.25, 0.75), (0.75, 0.25))


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _readonly_q(value: np.ndarray) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_crystal must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_crystal must be nonzero")
    q.setflags(write=False)
    return q


def _normalize_shifts(
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    shifts: list[tuple[float, float]] = []
    for raw in values:
        if len(raw) != 2:
            raise ValueError("every BZ shift must contain two values")
        shift = (float(raw[0]) % 1.0, float(raw[1]) % 1.0)
        if not np.isfinite(shift).all():
            raise ValueError("BZ shifts must be finite")
        shifts.append(shift)
    normalized = tuple(shifts)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("at least two unique BZ shifts are required")
    return normalized


def _shift_label(index: int, shift: tuple[float, float]) -> str:
    return f"shift_{int(index)}:{float(shift[0]).hex()}:{float(shift[1]).hex()}"


@dataclass(frozen=True)
class MaterialResponseEngineConfig:
    """Complete geometry-independent policy for one response ladder."""

    pairing_name: str
    temperature_K: float = 10.0
    delta0_eV: float = 0.1
    eta_eV: float = 1e-8
    matsubara_indices: tuple[int, ...] = (0, 1)
    n_candidates: tuple[int, ...] = (128, 192, 256)
    shifts: tuple[tuple[float, float], ...] = DEFAULT_RESPONSE_SHIFTS
    required_consecutive_passes: int = 2
    envelope_levels: int = 3
    canonical_reduction_block_size: int = 4096
    runtime_chunk_size: int = 16384
    microscopic_model_name: str = "symmetry_bdg_2band"
    material_policy: MaterialResponsePolicy = MaterialResponsePolicy()
    convergence_policy: MaterialResponseConvergencePolicy = (
        MaterialResponseConvergencePolicy()
    )

    def __post_init__(self) -> None:
        pairing = str(self.pairing_name)
        if pairing not in {"spm", "dwave"}:
            raise ValueError("pairing_name must be 'spm' or 'dwave'")
        object.__setattr__(self, "pairing_name", pairing)
        object.__setattr__(
            self,
            "temperature_K",
            _finite_positive(self.temperature_K, "temperature_K"),
        )
        object.__setattr__(
            self,
            "delta0_eV",
            _finite_positive(self.delta0_eV, "delta0_eV"),
        )
        object.__setattr__(
            self,
            "eta_eV",
            _finite_nonnegative(self.eta_eV, "eta_eV"),
        )

        indices = tuple(sorted(set(int(value) for value in self.matsubara_indices)))
        if not indices or any(value < 0 for value in indices):
            raise ValueError("matsubara_indices must be nonempty and non-negative")
        object.__setattr__(self, "matsubara_indices", indices)

        levels = tuple(int(value) for value in self.n_candidates)
        if (
            len(levels) < 2
            or tuple(sorted(set(levels))) != levels
            or any(value <= 0 or value % 2 for value in levels)
        ):
            raise ValueError(
                "n_candidates must be strictly increasing unique positive even integers"
            )
        object.__setattr__(self, "n_candidates", levels)
        object.__setattr__(self, "shifts", _normalize_shifts(self.shifts))

        required = int(self.required_consecutive_passes)
        if required <= 0 or required >= len(levels):
            raise ValueError("required_consecutive_passes is incompatible with N ladder")
        object.__setattr__(self, "required_consecutive_passes", required)
        envelope_levels = int(self.envelope_levels)
        if envelope_levels < 3:
            raise ValueError("envelope_levels must be at least three")
        object.__setattr__(self, "envelope_levels", envelope_levels)

        for name in ("canonical_reduction_block_size", "runtime_chunk_size"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        model_name = str(self.microscopic_model_name)
        if not model_name:
            raise ValueError("microscopic_model_name must be non-empty")
        object.__setattr__(self, "microscopic_model_name", model_name)
        if not isinstance(self.material_policy, MaterialResponsePolicy):
            raise TypeError("material_policy must be a MaterialResponsePolicy")
        if not isinstance(
            self.convergence_policy, MaterialResponseConvergencePolicy
        ):
            raise TypeError(
                "convergence_policy must be a MaterialResponseConvergencePolicy"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "material-response-engine-config-v1",
            "pairing_name": self.pairing_name,
            "temperature_K": self.temperature_K,
            "delta0_eV": self.delta0_eV,
            "eta_eV": self.eta_eV,
            "matsubara_indices": list(self.matsubara_indices),
            "n_candidates": list(self.n_candidates),
            "shifts": [list(value) for value in self.shifts],
            "required_consecutive_passes": self.required_consecutive_passes,
            "envelope_levels": self.envelope_levels,
            "canonical_reduction_block_size": (
                self.canonical_reduction_block_size
            ),
            "runtime_chunk_size": self.runtime_chunk_size,
            "microscopic_model_name": self.microscopic_model_name,
            "material_policy": self.material_policy.as_dict(),
            "convergence_policy": self.convergence_policy.as_dict(),
            "q_input_basis": "crystal_xy",
            "geometry_inputs_present": False,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        }


@dataclass(frozen=True)
class MaterialFrequencyResult:
    """N/shift history and optional response certification for one Matsubara index."""

    matsubara_index: int
    xi_eV: float
    history: tuple[MaterialResponseLevelRecord, ...]
    certification: CertifiedMaterialResponse | None

    def __post_init__(self) -> None:
        index = int(self.matsubara_index)
        if index < 0:
            raise ValueError("matsubara_index must be non-negative")
        object.__setattr__(self, "matsubara_index", index)
        object.__setattr__(
            self,
            "xi_eV",
            _finite_nonnegative(self.xi_eV, "xi_eV"),
        )
        object.__setattr__(self, "history", tuple(self.history))

    @property
    def established(self) -> bool:
        return self.certification is not None


@dataclass(frozen=True)
class MaterialResponseEngineResult:
    """Geometry-free response-ladder output for one pairing and crystal q."""

    config: MaterialResponseEngineConfig
    q_crystal: np.ndarray
    frequencies: Mapping[int, MaterialFrequencyResult]
    evaluated_n_candidates: tuple[int, ...]
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_RESPONSE_ENGINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_RESPONSE_ENGINE_SCHEMA:
            raise ValueError(f"schema must be {MATERIAL_RESPONSE_ENGINE_SCHEMA!r}")
        object.__setattr__(self, "q_crystal", _readonly_q(self.q_crystal))
        frequency_map = dict(self.frequencies)
        if set(frequency_map) != set(self.config.matsubara_indices):
            raise ValueError("frequency result keys do not match requested indices")
        object.__setattr__(self, "frequencies", MappingProxyType(frequency_map))
        object.__setattr__(
            self,
            "evaluated_n_candidates",
            tuple(int(value) for value in self.evaluated_n_candidates),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def all_requested_certified(self) -> bool:
        return all(result.established for result in self.frequencies.values())


def evaluate_material_response_ladder(
    config: MaterialResponseEngineConfig,
    *,
    q_crystal: np.ndarray,
) -> MaterialResponseEngineResult:
    """Evaluate and certify requested frequencies without constructing geometry."""

    if not isinstance(config, MaterialResponseEngineConfig):
        raise TypeError("config must be a MaterialResponseEngineConfig")
    q = _readonly_q(q_crystal)
    model = get_finite_q_microscopic_model(config.microscopic_model_name)
    ansatz = model.build_ansatz(
        config.pairing_name,
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(config.delta0_eV)

    xi_by_n = {
        index: matsubara_energy_eV(index, config.temperature_K)
        for index in config.matsubara_indices
    }
    histories: dict[int, list[MaterialResponseLevelRecord]] = {
        index: [] for index in config.matsubara_indices
    }
    previous: dict[int, Mapping[str, MaterialResponseSample]] = {}
    certifications: dict[int, CertifiedMaterialResponse] = {}
    active = set(config.matsubara_indices)
    evaluated_levels: list[int] = []

    for n_grid in config.n_candidates:
        if not active:
            break
        active_indices = tuple(index for index in config.matsubara_indices if index in active)
        xi_values = np.asarray([xi_by_n[index] for index in active_indices], dtype=float)
        samples_at_level: dict[int, dict[str, MaterialResponseSample]] = {
            index: {} for index in active_indices
        }

        for shift_index, shift in enumerate(config.shifts):
            integrated = integrate_arbitrary_q_periodic_bz(
                spec=model.spec,
                ansatz=ansatz,
                pairing=pairing,
                xi_eV_values=xi_values,
                temperature_K=config.temperature_K,
                eta_eV=config.eta_eV,
                q_model=q,
                n=n_grid,
                shift=shift,
                canonical_reduction_block_size=(
                    config.canonical_reduction_block_size
                ),
                runtime_chunk_size=config.runtime_chunk_size,
            )
            label = _shift_label(shift_index, shift)
            for frequency_index, matsubara_index in enumerate(active_indices):
                samples_at_level[matsubara_index][label] = (
                    build_material_response_sample(
                        integrated,
                        frequency_index=frequency_index,
                        policy=config.material_policy,
                    )
                )

        evaluated_levels.append(n_grid)
        for matsubara_index in active_indices:
            current = samples_at_level[matsubara_index]
            assessment = assess_material_response_level(
                current_by_shift=current,
                previous_by_shift=previous.get(matsubara_index),
                policy=config.convergence_policy,
            )
            record = MaterialResponseLevelRecord(
                n_grid=n_grid,
                samples_by_shift=current,
                assessment=assessment,
            )
            histories[matsubara_index].append(record)
            previous[matsubara_index] = current
            certified = certify_material_response_history(
                histories[matsubara_index],
                policy=config.convergence_policy,
                required_consecutive_passes=(
                    config.required_consecutive_passes
                ),
                envelope_levels=config.envelope_levels,
            )
            if certified is not None:
                certifications[matsubara_index] = certified
                active.discard(matsubara_index)

    frequency_results = {
        index: MaterialFrequencyResult(
            matsubara_index=index,
            xi_eV=xi_by_n[index],
            history=tuple(histories[index]),
            certification=certifications.get(index),
        )
        for index in config.matsubara_indices
    }
    metadata = {
        "casimir_stage": "geometry_independent_material_response_certification",
        "q_input_basis": "crystal_xy",
        "exact_q_used_without_rounding": True,
        "geometry_inputs_present": False,
        "reflection_constructed": False,
        "two_plate_logdet_constructed": False,
        "outer_integration_performed": False,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    return MaterialResponseEngineResult(
        config=config,
        q_crystal=q,
        frequencies=frequency_results,
        evaluated_n_candidates=tuple(evaluated_levels),
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_RESPONSE_SHIFTS",
    "MATERIAL_RESPONSE_ENGINE_SCHEMA",
    "MaterialFrequencyResult",
    "MaterialResponseEngineConfig",
    "MaterialResponseEngineResult",
    "evaluate_material_response_ladder",
]
