"""Exact arbitrary-q Matsubara response on a fixed periodic BZ lattice."""
from __future__ import annotations

from dataclasses import dataclass
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
    SUPPORTED_Q_COMPONENT_LIMIT,
    validate_q_domain,
)
from lno327.response.arbitrary_q_material_cache import (
    MaterialGridCache,
    build_material_grid_cache,
    material_cache_fingerprint,
)
from lno327.response.periodic_bz_grid import build_periodic_bz_grid, exact_float64_key
from lno327.response.primitive_kernel_v2 import OperatorWardReport, unpack_integrated_primitives
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
    profile: object
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


@dataclass(frozen=True)
class PairedShiftProfile:
    k_point_count: int
    frequency_count: int
    canonical_reduction_block_size: int
    runtime_chunk_size: int
    canonical_block_count: int
    runtime_chunk_count: int
    q_workspace_build_count: int
    shifted_eigensystem_build_count: int
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_pack_seconds: float
    operator_ward_seconds: float
    accumulation_seconds: float
    total_seconds: float
    counterterm_add_count: int
    material_cache_fingerprint: str
    source_material_cache_fingerprints: tuple[str, str]
    source_grid_fingerprints: tuple[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_schema": "PairedShiftProfile-v1",
            "k_point_count": int(self.k_point_count),
            "total_point_evaluations": int(self.k_point_count),
            "frequency_count": int(self.frequency_count),
            "canonical_reduction_block_size": int(self.canonical_reduction_block_size),
            "runtime_chunk_size": int(self.runtime_chunk_size),
            "canonical_block_count": int(self.canonical_block_count),
            "runtime_chunk_count": int(self.runtime_chunk_count),
            "q_workspace_build_count": int(self.q_workspace_build_count),
            "shifted_eigensystem_build_count": int(self.shifted_eigensystem_build_count),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "kubo_contraction_seconds": float(self.kubo_contraction_seconds),
            "primitive_pack_seconds": float(self.primitive_pack_seconds),
            "operator_ward_seconds": float(self.operator_ward_seconds),
            "accumulation_seconds": float(self.accumulation_seconds),
            "total_seconds": float(self.total_seconds),
            "counterterm_add_count": int(self.counterterm_add_count),
            "counterterm_source_evaluations": 2,
            "counterterm_effective_count_after_average": 1,
            "material_cache_fingerprint": self.material_cache_fingerprint,
            "source_material_cache_fingerprints": list(self.source_material_cache_fingerprints),
            "source_grid_fingerprints": list(self.source_grid_fingerprints),
        }


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
        self._values[
            self.key(
                result.material_cache_fingerprint,
                result.q_model,
                result.xi_eV_values,
                phase_policy=str(metadata["post_integral_phase_hessian_policy"]),
                canonical_reduction_block_size=int(metadata["canonical_reduction_block_size"]),
                operator_ward_atol=float(result.operator_ward.atol),
                operator_ward_rtol=float(result.operator_ward.rtol),
                primitive_contract_version=str(metadata["primitive_contract_version"]),
            )
        ] = result

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
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=float) @ q


def _phase_policy(pairing_name: str) -> str:
    return "nearest_neighbor_bond_metric" if pairing_name == "dwave" else "q_independent"


def _base_config(
    xi_values: np.ndarray, *, temperature_K: float, eta_eV: float
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
    profile: object,
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
    xi_values = _validate_xi(xi_eV_values)
    q = validate_q_domain(np.asarray(q_model, dtype=float))
    pairing_name = str(getattr(ansatz, "name", ""))
    if pairing_name not in {"spm", "dwave"}:
        raise ValueError("arbitrary-q periodic BZ supports spm and dwave")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("arbitrary-q periodic BZ requires bond_endpoint_gauge")
    base_config = _base_config(xi_values, temperature_K=temperature_K, eta_eV=eta_eV)
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
        "arbitrary_q_contract": "ArbitraryQPeriodicBZContract-v3",
        "primitive_contract_version": PRIMITIVE_CONTRACT_VERSION,
        "exact_q_used_without_rounding": True,
        "q_wrapping_forbidden": True,
        "principal_q_domain_kind": "syntactically_supported_not_numerically_qualified",
        "supported_q_component_limit": SUPPORTED_Q_COMPONENT_LIMIT,
        "numerically_qualified_q_envelope_established": False,
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
        "material_state_fingerprint": material_cache.material_state_fingerprint,
        "grid_fingerprint": material_cache.grid_fingerprint,
        "counterterm_add_count": int(accumulated.profile.counterterm_add_count),
        "canonical_reduction_block_size": int(canonical_reduction_block_size),
        "runtime_chunk_size": int(runtime_chunk_size),
        "runtime_chunk_affects_numerical_definition": False,
        "runtime_chunk_controls_q_workspace_batch": True,
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


def _profile_sum(first: object, second: object, fingerprint: str) -> PairedShiftProfile:
    integer_sum = lambda name: int(getattr(first, name)) + int(getattr(second, name))
    float_sum = lambda name: float(getattr(first, name)) + float(getattr(second, name))
    return PairedShiftProfile(
        k_point_count=integer_sum("k_point_count"),
        frequency_count=int(getattr(first, "frequency_count")),
        canonical_reduction_block_size=int(getattr(first, "canonical_reduction_block_size")),
        runtime_chunk_size=int(getattr(first, "runtime_chunk_size")),
        canonical_block_count=integer_sum("canonical_block_count"),
        runtime_chunk_count=integer_sum("runtime_chunk_count"),
        q_workspace_build_count=integer_sum("q_workspace_build_count"),
        shifted_eigensystem_build_count=integer_sum("shifted_eigensystem_build_count"),
        q_workspace_seconds=float_sum("q_workspace_seconds"),
        kubo_factor_seconds=float_sum("kubo_factor_seconds"),
        kubo_contraction_seconds=float_sum("kubo_contraction_seconds"),
        primitive_pack_seconds=float_sum("primitive_pack_seconds"),
        operator_ward_seconds=float_sum("operator_ward_seconds"),
        accumulation_seconds=float_sum("accumulation_seconds"),
        total_seconds=float_sum("total_seconds"),
        counterterm_add_count=1,
        material_cache_fingerprint=fingerprint,
        source_material_cache_fingerprints=(
            str(getattr(first, "material_cache_fingerprint")),
            str(getattr(second, "material_cache_fingerprint")),
        ),
        source_grid_fingerprints=("", ""),
    )


def paired_average_arbitrary_q_results(
    first: ArbitraryQPeriodicBZResult,
    second: ArbitraryQPeriodicBZResult,
    *,
    ansatz: object,
    pairing: object,
    temperature_K: float,
    eta_eV: float,
    require_formal_audit_pair: bool = True,
) -> ArbitraryQPeriodicBZResult:
    if not np.array_equal(first.q_model, second.q_model):
        raise ValueError("paired shift results must use exactly the same q")
    if not np.array_equal(first.xi_eV_values, second.xi_eV_values):
        raise ValueError("paired shift results must use exactly the same frequencies")
    for key in (
        "primitive_contract_version",
        "post_integral_phase_hessian_policy",
        "canonical_reduction_block_size",
        "runtime_chunk_size",
        "material_state_fingerprint",
    ):
        if first.metadata.get(key) != second.metadata.get(key):
            raise ValueError(f"paired shift result policy/state differs for {key}")
    if first.operator_ward.atol != second.operator_ward.atol or first.operator_ward.rtol != second.operator_ward.rtol:
        raise ValueError("paired shift operator Ward tolerances differ")

    grid_a = dict(first.metadata.get("grid", {}))
    grid_b = dict(second.metadata.get("grid", {}))
    for key in ("N", "bz_convention", "ordering", "point_count"):
        if grid_a.get(key) != grid_b.get(key):
            raise ValueError(f"paired shift grids differ for {key}")
    shift_a = tuple(float(v) % 1.0 for v in grid_a.get("shift", ()))
    shift_b = tuple(float(v) % 1.0 for v in grid_b.get("shift", ()))
    if len(shift_a) != 2 or len(shift_b) != 2:
        raise ValueError("paired shift grid metadata lacks two-dimensional shifts")
    expected_b = tuple((-value) % 1.0 for value in shift_a)
    if any(abs(a - b) > 64.0 * np.finfo(float).eps for a, b in zip(expected_b, shift_b)):
        raise ValueError("paired audit shifts are not related by inversion")
    if require_formal_audit_pair:
        formal = {(0.25, 0.75), (0.75, 0.25)}
        if {shift_a, shift_b} != formal:
            raise ValueError("formal paired audit requires shifts (1/4,3/4) and (3/4,1/4)")

    packed = 0.5 * (
        np.asarray(first.packed_primitives, dtype=complex)
        + np.asarray(second.packed_primitives, dtype=complex)
    )
    fingerprint = hashlib.sha256(
        (
            "paired-shift-v1:"
            + str(first.metadata["material_state_fingerprint"])
            + ":"
            + str(first.metadata["grid_fingerprint"])
            + ":"
            + str(second.metadata["grid_fingerprint"])
        ).encode("utf-8")
    ).hexdigest()
    profile = _profile_sum(first.profile, second.profile, fingerprint)
    object.__setattr__(
        profile,
        "source_grid_fingerprints",
        (str(first.metadata["grid_fingerprint"]), str(second.metadata["grid_fingerprint"])),
    )
    operator = combine_operator_ward_reports((first.operator_ward, second.operator_ward))
    paired_grid = {
        "grid_contract": "PairedShiftGrid-v1",
        "N": int(grid_a["N"]),
        "point_count_per_shift": int(grid_a["point_count"]),
        "total_point_evaluations": int(grid_a["point_count"]) + int(grid_b["point_count"]),
        "audit_shifts": [list(shift_a), list(shift_b)],
        "bz_convention": grid_a["bz_convention"],
        "ordering": grid_a["ordering"],
        "source_grid_fingerprints": list(profile.source_grid_fingerprints),
        "fingerprint": hashlib.sha256(
            (str(first.metadata["grid_fingerprint"]) + str(second.metadata["grid_fingerprint"])).encode("utf-8")
        ).hexdigest(),
    }
    metadata = {
        **dict(first.metadata),
        "integration_strategy": "paired_shift_average_of_linear_packed_primitives",
        "paired_shift_primitive_average": True,
        "paired_shift_profile_schema": "PairedShiftProfile-v1",
        "paired_shift_material_fingerprints": list(profile.source_material_cache_fingerprints),
        "material_cache_fingerprint": fingerprint,
        "grid_fingerprint": paired_grid["fingerprint"],
        "grid": paired_grid,
        "material_cache_hit": "paired_source_results",
        "material_cache_build_count": 2,
        "counterterm_add_count": 1,
        "accumulation_profile": profile.as_dict(),
        "nonlinear_observable_average_forbidden": True,
        "operator_ward": operator.as_dict(),
    }
    base_config = _base_config(first.xi_eV_values, temperature_K=temperature_K, eta_eV=eta_eV)
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
        q_model=rotate_lab_q_to_crystal(q, float(theta_1_rad)), **common
    )
    plate_2 = tuple(
        integrate_arbitrary_q_periodic_bz(
            q_model=rotate_lab_q_to_crystal(q, float(theta)), **common
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
    "PairedShiftProfile",
    "TwoPlateAngleBatchResult",
    "integrate_arbitrary_q_periodic_bz",
    "integrate_two_plate_angle_batch",
    "paired_average_arbitrary_q_results",
    "rotate_lab_q_to_crystal",
]
