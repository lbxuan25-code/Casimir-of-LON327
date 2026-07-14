"""Exact arbitrary-q Matsubara response on a fixed periodic BZ lattice."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_accumulator import (
    ArbitraryQAccumulationProfile,
    accumulate_arbitrary_q_primitives,
    combine_operator_ward_reports,
)
from lno327.response.arbitrary_q_formal_policy import (
    PRIMITIVE_CONTRACT_VERSION,
    RESPONSE_CACHE_SCHEMA,
    VALIDATED_Q_COMPONENT_LIMIT,
    validate_q_domain,
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
from lno327.response.primitive_kernel_v2 import (
    OperatorWardReport,
    unpack_integrated_primitives,
)
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.finite_q_engine import FiniteQEngineOptions


@dataclass(frozen=True)
class ArbitraryQPeriodicBZResult:
    q_model: np.ndarray
    xi_eV_values: np.ndarray
    packed_primitives: np.ndarray
    components: tuple[object, ...]
    rhs: tuple[PrimitiveWardRHS, ...]
    operator_ward: OperatorWardReport
    profile: ArbitraryQAccumulationProfile
    material_cache_fingerprint: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        q = np.array(self.q_model, dtype=float, copy=True)
        xi = np.array(self.xi_eV_values, dtype=float, copy=True)
        packed = np.array(self.packed_primitives, dtype=complex, copy=True)
        q.setflags(write=False)
        xi.setflags(write=False)
        packed.setflags(write=False)
        object.__setattr__(self, "q_model", q)
        object.__setattr__(self, "xi_eV_values", xi)
        object.__setattr__(self, "packed_primitives", packed)
        if q.shape != (2,):
            raise ValueError("q_model must have shape (2,)")
        if len(self.components) != xi.size or len(self.rhs) != xi.size:
            raise ValueError("components/rhs/frequency lengths differ")


class CrystalResponseCache:
    """Policy-complete exact-float response cache; no q rounding or interpolation."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, ...], ArbitraryQPeriodicBZResult] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        material_fingerprint: str,
        q_model: np.ndarray,
        xi_values: np.ndarray,
        *,
        phase_policy: str,
        canonical_reduction_block_size: int,
        operator_ward_atol: float,
        operator_ward_rtol: float,
        primitive_contract_version: str = PRIMITIVE_CONTRACT_VERSION,
    ) -> tuple[str, ...]:
        return (
            RESPONSE_CACHE_SCHEMA,
            str(material_fingerprint),
            exact_float64_key(np.asarray(q_model, dtype=float)),
            exact_float64_key(np.asarray(xi_values, dtype=float)),
            str(phase_policy),
            str(int(canonical_reduction_block_size)),
            exact_float64_key(np.asarray([operator_ward_atol, operator_ward_rtol])),
            str(primitive_contract_version),
        )

    def get(
        self,
        material_fingerprint: str,
        q_model: np.ndarray,
        xi_values: np.ndarray,
        *,
        phase_policy: str,
        canonical_reduction_block_size: int,
        operator_ward_atol: float,
        operator_ward_rtol: float,
        primitive_contract_version: str = PRIMITIVE_CONTRACT_VERSION,
    ) -> ArbitraryQPeriodicBZResult | None:
        value = self._values.get(
            self.key(
                material_fingerprint,
                q_model,
                xi_values,
                phase_policy=phase_policy,
                canonical_reduction_block_size=canonical_reduction_block_size,
                operator_ward_atol=operator_ward_atol,
                operator_ward_rtol=operator_ward_rtol,
                primitive_contract_version=primitive_contract_version,
            )
        )
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, result: ArbitraryQPeriodicBZResult) -> None:
        metadata = result.metadata
        key = self.key(
            result.material_cache_fingerprint,
            result.q_model,
            result.xi_eV_values,
            phase_policy=str(metadata["post_integral_phase_hessian_policy"]),
            canonical_reduction_block_size=int(
                metadata["canonical_reduction_block_size"]
            ),
            operator_ward_atol=float(result.operator_ward.atol),
            operator_ward_rtol=float(result.operator_ward.rtol),
            primitive_contract_version=str(metadata["primitive_contract_version"]),
        )
        self._values[key] = result

    def metadata(self) -> dict[str, int | str]:
        return {
            "schema": RESPONSE_CACHE_SCHEMA,
            "entries": len(self._values),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "q_key": "canonicalized_ieee754_float64_bytes",
            "runtime_chunk_affects_key": "false",
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


def _phase_policy(pairing_name: str) -> str:
    return (
        "nearest_neighbor_bond_metric"
        if pairing_name == "dwave"
        else "q_independent"
    )


def _base_config(
    xi_values: np.ndarray,
    *,
    temperature_K: float,
    eta_eV: float,
) -> KuboConfig:
    return KuboConfig.from_kelvin(
        omega_eV=float(xi_values[0]),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )


def _finalize_result(
    *,
    packed: np.ndarray,
    q: np.ndarray,
    xi_values: np.ndarray,
    ansatz: object,
    pairing: object,
    base_config: KuboConfig,
    options: FiniteQEngineOptions,
    phase_policy: str,
    operator_ward: OperatorWardReport,
    profile: ArbitraryQAccumulationProfile,
    material_fingerprint: str,
    integration_metadata: dict[str, object],
) -> ArbitraryQPeriodicBZResult:
    components, rhs = unpack_integrated_primitives(
        packed,
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
    return ArbitraryQPeriodicBZResult(
        q_model=q,
        xi_eV_values=xi_values,
        packed_primitives=packed,
        components=components,
        rhs=rhs,
        operator_ward=operator_ward,
        profile=profile,
        material_cache_fingerprint=material_fingerprint,
        metadata=integration_metadata,
    )


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
    """Evaluate exact q without commensuration, rounding, wrapping, or interpolation."""

    xi_values = _validate_xi(xi_eV_values)
    q = validate_q_domain(np.asarray(q_model, dtype=float))
    pairing_name = str(getattr(ansatz, "name", ""))
    if pairing_name not in {"spm", "dwave"}:
        raise ValueError("arbitrary-q periodic BZ supports spm and dwave")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("arbitrary-q periodic BZ requires bond_endpoint_gauge")

    base_config = _base_config(
        xi_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    phase_policy = _phase_policy(pairing_name)

    cache_hit = material_cache is not None
    if material_cache is None:
        grid = build_periodic_bz_grid(int(n), shift)
        material_cache = build_material_grid_cache(
            spec=spec,
            ansatz=ansatz,
            pairing=pairing,
            config=base_config,
            options=options,
            grid=grid,
        )
    else:
        grid = material_cache.grid
        requested_shift = tuple(float(value) % 1.0 for value in shift)
        if int(n) != int(grid.n):
            raise ValueError("requested n differs from provided material cache grid")
        if requested_shift != tuple(float(value) for value in grid.shift):
            raise ValueError("requested shift differs from provided material cache grid")
        expected_fingerprint = material_cache_fingerprint(
            spec=spec,
            ansatz=ansatz,
            pairing=pairing,
            config=base_config,
            options=options,
            grid=grid,
        )
        if material_cache.fingerprint != expected_fingerprint:
            raise ValueError("provided material cache fingerprint does not match this run")

    if response_cache is not None:
        cached = response_cache.get(
            material_cache.fingerprint,
            q,
            xi_values,
            phase_policy=phase_policy,
            canonical_reduction_block_size=canonical_reduction_block_size,
            operator_ward_atol=operator_ward_atol,
            operator_ward_rtol=operator_ward_rtol,
        )
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
    integration_metadata: dict[str, object] = {
        "integration_strategy": "arbitrary_q_fixed_shifted_periodic_bz_lattice",
        "arbitrary_q_contract": "ArbitraryQPeriodicBZContract-v2",
        "primitive_contract_version": PRIMITIVE_CONTRACT_VERSION,
        "exact_q_used_without_rounding": True,
        "q_wrapping_forbidden": True,
        "validated_q_component_limit": VALIDATED_Q_COMPONENT_LIMIT,
        "translation_by_q_is_exact_orbit_permutation": False,
        "matsubara_batch_shared_nodes": True,
        "zero_and_positive_frequencies_share_eigensystems": bool(
            np.any(xi_values == 0.0) and np.any(xi_values > 0.0)
        ),
        "exact_zero_uses_divided_difference": bool(np.any(xi_values == 0.0)),
        "conductivity_division_for_zero_forbidden": True,
        "post_integral_phase_hessian_policy": phase_policy,
        "material_cache_hit": bool(cache_hit),
        "material_cache_build_count": int(material_cache.build_count),
        "material_cache_fingerprint": material_cache.fingerprint,
        "counterterm_add_count": int(accumulated.profile.counterterm_add_count),
        "canonical_reduction_block_size": int(
            canonical_reduction_block_size
        ),
        "runtime_chunk_size": int(runtime_chunk_size),
        "runtime_chunk_affects_numerical_definition": False,
        "grid": grid.metadata(),
        "operator_ward": accumulated.operator_ward.as_dict(),
        "accumulation_profile": accumulated.profile.as_dict(),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    result = _finalize_result(
        packed=accumulated.packed,
        q=q,
        xi_values=xi_values,
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        options=options,
        phase_policy=phase_policy,
        operator_ward=accumulated.operator_ward,
        profile=accumulated.profile,
        material_fingerprint=material_cache.fingerprint,
        integration_metadata=integration_metadata,
    )
    if response_cache is not None:
        response_cache.put(result)
    return result


def paired_average_arbitrary_q_results(
    first: ArbitraryQPeriodicBZResult,
    second: ArbitraryQPeriodicBZResult,
    *,
    ansatz: object,
    pairing: object,
    temperature_K: float,
    eta_eV: float,
) -> ArbitraryQPeriodicBZResult:
    """Average two audit shifts at the linear packed-primitive level."""

    if not np.array_equal(first.q_model, second.q_model):
        raise ValueError("paired shift results must use exactly the same q")
    if not np.array_equal(first.xi_eV_values, second.xi_eV_values):
        raise ValueError("paired shift results must use exactly the same frequencies")
    for key in (
        "primitive_contract_version",
        "post_integral_phase_hessian_policy",
        "canonical_reduction_block_size",
    ):
        if first.metadata.get(key) != second.metadata.get(key):
            raise ValueError(f"paired shift result policy differs for {key}")
    if (
        first.operator_ward.atol != second.operator_ward.atol
        or first.operator_ward.rtol != second.operator_ward.rtol
    ):
        raise ValueError("paired shift operator Ward tolerances differ")

    packed = 0.5 * (
        np.asarray(first.packed_primitives, dtype=complex)
        + np.asarray(second.packed_primitives, dtype=complex)
    )
    fingerprint = hashlib.sha256(
        (
            "paired-shift:"
            + first.material_cache_fingerprint
            + ":"
            + second.material_cache_fingerprint
        ).encode("utf-8")
    ).hexdigest()
    profile = replace(
        first.profile,
        material_cache_fingerprint=fingerprint,
        total_seconds=float(first.profile.total_seconds + second.profile.total_seconds),
    )
    operator = combine_operator_ward_reports(
        (first.operator_ward, second.operator_ward)
    )
    metadata = {
        **dict(first.metadata),
        "integration_strategy": "paired_shift_average_of_linear_packed_primitives",
        "paired_shift_primitive_average": True,
        "paired_shift_material_fingerprints": [
            first.material_cache_fingerprint,
            second.material_cache_fingerprint,
        ],
        "nonlinear_observable_average_forbidden": True,
        "operator_ward": operator.as_dict(),
    }
    base_config = _base_config(
        first.xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    return _finalize_result(
        packed=packed,
        q=np.asarray(first.q_model, dtype=float),
        xi_values=np.asarray(first.xi_eV_values, dtype=float),
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
        phase_policy=str(first.metadata["post_integral_phase_hessian_policy"]),
        operator_ward=operator,
        profile=profile,
        material_fingerprint=fingerprint,
        integration_metadata=metadata,
    )


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
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
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
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
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
    "paired_average_arbitrary_q_results",
    "rotate_lab_q_to_crystal",
]
