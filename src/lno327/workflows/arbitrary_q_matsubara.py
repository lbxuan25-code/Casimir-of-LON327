"""Exact arbitrary-q Matsubara response on a fixed periodic BZ lattice."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_accumulator import (
    ArbitraryQAccumulationProfile,
    accumulate_arbitrary_q_primitives,
)
from lno327.response.arbitrary_q_material_cache import (
    MaterialGridCache,
    build_material_grid_cache,
    material_cache_fingerprint,
)
from lno327.response.periodic_bz_grid import (
    build_periodic_bz_grid,
    exact_float64_key,
)
from lno327.response.primitive_kernel import (
    OperatorWardReport,
    unpack_integrated_primitives,
)
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.finite_q_engine import FiniteQEngineOptions


@dataclass(frozen=True)
class ArbitraryQPeriodicBZResult:
    q_model: np.ndarray
    xi_eV_values: np.ndarray
    components: tuple[object, ...]
    rhs: tuple[PrimitiveWardRHS, ...]
    operator_ward: OperatorWardReport
    profile: ArbitraryQAccumulationProfile
    material_cache_fingerprint: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        q = np.array(self.q_model, dtype=float, copy=True)
        xi = np.array(self.xi_eV_values, dtype=float, copy=True)
        q.setflags(write=False)
        xi.setflags(write=False)
        object.__setattr__(self, "q_model", q)
        object.__setattr__(self, "xi_eV_values", xi)
        if q.shape != (2,):
            raise ValueError("q_model must have shape (2,)")
        if len(self.components) != xi.size or len(self.rhs) != xi.size:
            raise ValueError("components/rhs/frequency lengths differ")


class CrystalResponseCache:
    """Exact-float in-memory response cache; no q rounding or interpolation."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str, str], ArbitraryQPeriodicBZResult] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        material_fingerprint: str,
        q_model: np.ndarray,
        xi_values: np.ndarray,
    ) -> tuple[str, str, str]:
        return (
            str(material_fingerprint),
            exact_float64_key(np.asarray(q_model, dtype=float)),
            exact_float64_key(np.asarray(xi_values, dtype=float)),
        )

    def get(
        self,
        material_fingerprint: str,
        q_model: np.ndarray,
        xi_values: np.ndarray,
    ) -> ArbitraryQPeriodicBZResult | None:
        value = self._values.get(self.key(material_fingerprint, q_model, xi_values))
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, result: ArbitraryQPeriodicBZResult) -> None:
        key = self.key(
            result.material_cache_fingerprint,
            result.q_model,
            result.xi_eV_values,
        )
        self._values[key] = result

    def metadata(self) -> dict[str, int | str]:
        return {
            "schema": "CrystalResponseCache-v1",
            "entries": len(self._values),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "q_key": "canonicalized_ieee754_float64_bytes",
        }


def _validate_xi(values: Sequence[float] | np.ndarray) -> np.ndarray:
    xi = np.asarray(values, dtype=float)
    if xi.ndim != 1 or xi.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi).all() or np.any(xi < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    if np.count_nonzero(xi == 0.0) > 1:
        raise ValueError("exact zero may appear at most once")
    return xi


def rotate_lab_q_to_crystal(q_lab: np.ndarray, theta_rad: float) -> np.ndarray:
    q = np.asarray(q_lab, dtype=float)
    theta = float(theta_rad)
    if q.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(theta):
        raise ValueError("q_lab and theta must be finite")
    cosine = np.cos(theta)
    sine = np.sin(theta)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=float)
    return rotation @ q


def integrate_arbitrary_q_periodic_bz(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    q_model: np.ndarray,
    n: int = 256,
    shift: tuple[float, float] = (0.5, 0.5),
    canonical_reduction_block_size: int = 4096,
    runtime_chunk_size: int = 16384,
    material_cache: MaterialGridCache | None = None,
    response_cache: CrystalResponseCache | None = None,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
) -> ArbitraryQPeriodicBZResult:
    """Evaluate exact q without commensuration, rounding, or interpolation."""

    xi_values = _validate_xi(xi_eV_values)
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    pairing_name = str(getattr(ansatz, "name", ""))
    if pairing_name not in {"spm", "dwave"}:
        raise ValueError("arbitrary-q periodic BZ supports spm and dwave")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("arbitrary-q periodic BZ requires bond_endpoint_gauge")

    base_config = KuboConfig.from_kelvin(
        omega_eV=float(xi_values[0]),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    grid = build_periodic_bz_grid(int(n), shift)
    expected_fingerprint = material_cache_fingerprint(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        config=base_config,
        options=options,
        grid=grid,
    )
    cache_hit = material_cache is not None
    if material_cache is None:
        material_cache = build_material_grid_cache(
            spec=spec,
            ansatz=ansatz,
            pairing=pairing,
            config=base_config,
            options=options,
            grid=grid,
        )
    elif material_cache.fingerprint != expected_fingerprint:
        raise ValueError("provided material cache fingerprint does not match this run")

    if response_cache is not None:
        cached = response_cache.get(material_cache.fingerprint, q, xi_values)
        if cached is not None:
            return cached

    accumulated = accumulate_arbitrary_q_primitives(
        material_cache,
        q,
        xi_values,
        canonical_reduction_block_size=canonical_reduction_block_size,
        runtime_chunk_size=runtime_chunk_size,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
    )
    phase_policy = (
        "nearest_neighbor_bond_metric" if pairing_name == "dwave" else "q_independent"
    )
    integration_metadata: dict[str, object] = {
        "integration_strategy": "arbitrary_q_fixed_shifted_periodic_bz_lattice",
        "arbitrary_q_contract": "ArbitraryQPeriodicBZContract-v1",
        "exact_q_used_without_rounding": True,
        "translation_by_q_is_exact_orbit_permutation": False,
        "matsubara_batch_shared_nodes": True,
        "zero_and_positive_frequencies_share_eigensystems": bool(
            np.any(xi_values == 0.0) and np.any(xi_values > 0.0)
        ),
        "exact_zero_uses_divided_difference": bool(np.any(xi_values == 0.0)),
        "conductivity_division_for_zero_forbidden": True,
        "material_cache_hit": bool(cache_hit),
        "material_cache_build_count": int(material_cache.build_count),
        "material_cache_fingerprint": material_cache.fingerprint,
        "counterterm_add_count": int(accumulated.profile.counterterm_add_count),
        "canonical_reduction_block_size": int(
            canonical_reduction_block_size
        ),
        "runtime_chunk_size": int(runtime_chunk_size),
        "grid": grid.metadata(),
        "operator_ward": accumulated.operator_ward.as_dict(),
        "accumulation_profile": accumulated.profile.as_dict(),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    components, rhs = unpack_integrated_primitives(
        accumulated.packed,
        xi_values=xi_values,
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        q_model=q,
        options=options,
        phase_hessian_policy=phase_policy,
        integration_metadata=integration_metadata,
        rhs_source="arbitrary_q_fixed_periodic_bz_integral",
    )
    result = ArbitraryQPeriodicBZResult(
        q_model=q,
        xi_eV_values=xi_values,
        components=components,
        rhs=rhs,
        operator_ward=accumulated.operator_ward,
        profile=accumulated.profile,
        material_cache_fingerprint=material_cache.fingerprint,
        metadata=integration_metadata,
    )
    if response_cache is not None:
        response_cache.put(result)
    return result


@dataclass(frozen=True)
class TwoPlateAngleBatchResult:
    q_lab: np.ndarray
    theta_1_rad: float
    theta_2_rad_values: np.ndarray
    plate_1: ArbitraryQPeriodicBZResult
    plate_2: tuple[ArbitraryQPeriodicBZResult, ...]
    response_cache_metadata: dict[str, int | str]

    def __post_init__(self) -> None:
        q = np.array(self.q_lab, dtype=float, copy=True)
        theta = np.array(self.theta_2_rad_values, dtype=float, copy=True)
        q.setflags(write=False)
        theta.setflags(write=False)
        object.__setattr__(self, "q_lab", q)
        object.__setattr__(self, "theta_2_rad_values", theta)
        if len(self.plate_2) != theta.size:
            raise ValueError("plate-2 result count differs from angle count")


def integrate_two_plate_angle_batch(
    *,
    q_lab: np.ndarray,
    theta_1_rad: float,
    theta_2_rad_values: Sequence[float] | np.ndarray,
    material_cache: MaterialGridCache,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    canonical_reduction_block_size: int = 4096,
    runtime_chunk_size: int = 16384,
    response_cache: CrystalResponseCache | None = None,
) -> TwoPlateAngleBatchResult:
    """Evaluate one q_lab task with plate 1 reused across an angle batch."""

    q = np.asarray(q_lab, dtype=float)
    angles = np.asarray(theta_2_rad_values, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_lab must be a finite vector with shape (2,)")
    if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
        raise ValueError("theta_2_rad_values must be a nonempty finite vector")
    local_cache = response_cache or CrystalResponseCache()
    common = dict(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        n=material_cache.grid.n,
        shift=material_cache.grid.shift,
        canonical_reduction_block_size=canonical_reduction_block_size,
        runtime_chunk_size=runtime_chunk_size,
        material_cache=material_cache,
        response_cache=local_cache,
    )
    plate_1 = integrate_arbitrary_q_periodic_bz(
        q_model=rotate_lab_q_to_crystal(q, float(theta_1_rad)),
        **common,
    )
    plate_2 = tuple(
        integrate_arbitrary_q_periodic_bz(
            q_model=rotate_lab_q_to_crystal(q, float(theta)),
            **common,
        )
        for theta in angles
    )
    return TwoPlateAngleBatchResult(
        q_lab=q,
        theta_1_rad=float(theta_1_rad),
        theta_2_rad_values=angles,
        plate_1=plate_1,
        plate_2=plate_2,
        response_cache_metadata=local_cache.metadata(),
    )


__all__ = [
    "ArbitraryQPeriodicBZResult",
    "CrystalResponseCache",
    "TwoPlateAngleBatchResult",
    "integrate_arbitrary_q_periodic_bz",
    "integrate_two_plate_angle_batch",
    "rotate_lab_q_to_crystal",
]
