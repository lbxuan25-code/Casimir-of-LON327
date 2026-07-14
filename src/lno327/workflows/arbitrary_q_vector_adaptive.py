"""Deterministic vector-adaptive cubature for exact arbitrary-q Matsubara response.

The backend shares the established arbitrary-q q-workspace and primitive packing
kernel. It changes only Brillouin-zone quadrature: rectangular cells are
integrated with paired low/high tensor-Gauss rules, all Matsubara frequencies and
all linear response/Ward primitives share one refinement tree, and the single
Schur/phase-Hessian post-processing step is applied only after accepted high-rule
cell primitives have been summed over the full BZ.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from math import ceil
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_accumulator import (
    ComplexKahanVector,
    combine_operator_ward_reports,
)
from lno327.response.arbitrary_q_formal_policy import (
    PRIMITIVE_CONTRACT_VERSION,
    SUPPORTED_Q_COMPONENT_LIMIT,
    validate_q_domain,
)
from lno327.response.arbitrary_q_material_cache import material_state_fingerprint
from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
    supports_batched_finite_q_material_workspace,
)
from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    _vectorized_kubo_factors,
)
from lno327.response.finite_q_q_workspace_batched import (
    _integrated_linear_terms_from_workspace_slice,
    precompute_finite_q_q_workspace_batched,
)
from lno327.response.periodic_bz_grid import exact_float64_key
from lno327.response.primitive_kernel import (
    OperatorWardReport,
    counterterm_primitive_vector,
    pack_integrated_primitives,
    primitive_vector_width,
    unpack_integrated_primitives,
)
from lno327.response.primitive_kernel_v2 import operator_ward_report_from_workspace
from lno327.response.ward_validation import primitive_ward_vectors_xy
from lno327.workflows.arbitrary_q_matsubara import (
    ArbitraryQPeriodicBZResult,
    TwoPlateAngleBatchResult,
    rotate_lab_q_to_crystal,
)
from lno327.workflows.dwave_vector_adaptive_cubature import (
    DWaveCubatureCell,
    cubature_cell_gauss_rule,
    initial_cubature_cells,
    subdivide_cubature_cell,
    vector_error_metrics,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

_FLOAT_EPS = np.finfo(float).eps
_ADAPTIVE_CONTRACT = "ArbitraryQVectorAdaptiveContract-v2"
_NODE_CACHE_SCHEMA = "HierarchicalMaterialNodeCache-v2"
_RESPONSE_CACHE_SCHEMA = "ArbitraryQVectorAdaptiveResponseCache-v2"


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class ArbitraryQVectorAdaptiveOptions:
    """Numerical definition and resource limits for vector-adaptive cubature."""

    coarse_grid: int = 6
    low_order: int = 2
    high_order: int = 3
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-9
    ward_error_tolerance: float = 1e-9
    max_level: int = 5
    max_iterations: int = 8
    refine_fraction: float = 0.15
    min_refine_cells: int = 4
    max_cells: int = 4000
    max_evaluation_points: int = 60_000
    cell_batch_size: int = 64

    def validate(self) -> None:
        if int(self.coarse_grid) <= 0:
            raise ValueError("coarse_grid must be positive")
        if int(self.low_order) <= 0 or int(self.high_order) <= int(self.low_order):
            raise ValueError("high_order must be greater than positive low_order")
        for name in (
            "relative_tolerance",
            "absolute_tolerance",
            "ward_error_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.max_level) < 0 or int(self.max_iterations) < 0:
            raise ValueError("max_level and max_iterations must be non-negative")
        if not 0.0 < float(self.refine_fraction) <= 1.0:
            raise ValueError("refine_fraction must lie in (0,1]")
        if int(self.min_refine_cells) <= 0:
            raise ValueError("min_refine_cells must be positive")
        if int(self.max_cells) <= 0 or int(self.max_evaluation_points) <= 0:
            raise ValueError("adaptive resource limits must be positive")
        if int(self.cell_batch_size) <= 0:
            raise ValueError("cell_batch_size must be positive")

    def numerical_definition(self) -> dict[str, object]:
        return {
            "contract": _ADAPTIVE_CONTRACT,
            "coarse_grid": int(self.coarse_grid),
            "low_order": int(self.low_order),
            "high_order": int(self.high_order),
            "rule_relation": "paired_nonembedded_tensor_gauss",
            "relative_tolerance": float(self.relative_tolerance).hex(),
            "absolute_tolerance": float(self.absolute_tolerance).hex(),
            "ward_error_tolerance": float(self.ward_error_tolerance).hex(),
            "max_level": int(self.max_level),
            "max_iterations": int(self.max_iterations),
            "refine_fraction": float(self.refine_fraction).hex(),
            "min_refine_cells": int(self.min_refine_cells),
            "max_cells": int(self.max_cells),
            "max_evaluation_points": int(self.max_evaluation_points),
            "cell_batch_size_excluded_from_numerical_definition": True,
        }

    @property
    def fingerprint(self) -> str:
        return _hash(self.numerical_definition())

    def as_dict(self) -> dict[str, object]:
        return {
            **self.numerical_definition(),
            "cell_batch_size": int(self.cell_batch_size),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class _NodeValue:
    energies: np.ndarray
    states: np.ndarray
    occupations: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "energies", _readonly(self.energies, float))
        object.__setattr__(self, "states", _readonly(self.states, complex))
        object.__setattr__(self, "occupations", _readonly(self.occupations, float))


class HierarchicalMaterialNodeCache:
    """Lazy q-independent midpoint cache keyed by exact adaptive Gauss nodes."""

    def __init__(
        self,
        *,
        spec: object,
        ansatz: object,
        pairing: object,
        config: KuboConfig,
        options: FiniteQEngineOptions,
    ) -> None:
        if not supports_batched_finite_q_material_workspace(spec, ansatz):
            raise ValueError("adaptive node cache requires batched material capabilities")
        self.spec = spec
        self.ansatz = ansatz
        self.pairing = pairing
        self.config = config
        self.options = options
        self.material_state_fingerprint = material_state_fingerprint(
            spec=spec,
            ansatz=ansatz,
            pairing=pairing,
            config=config,
            options=options,
        )
        self.fingerprint = _hash(
            {
                "schema": _NODE_CACHE_SCHEMA,
                "material_state_fingerprint": self.material_state_fingerprint,
            }
        )
        self._nodes: dict[str, _NodeValue] = {}
        self._counterterms: dict[str, np.ndarray] = {}
        self._pairing_params: object | None = None
        self._collective_mode: str | None = None
        self._collective_mode_disabled_reason: str | None = None
        self.node_hits = 0
        self.node_misses = 0
        self.material_batch_build_count = 0
        self.midpoint_eigh_call_count = 0
        self.counterterm_hits = 0
        self.counterterm_misses = 0

    @staticmethod
    def _point_key(point: np.ndarray) -> str:
        return exact_float64_key(np.asarray(point, dtype=float).reshape(2))

    @staticmethod
    def _counterterm_key(points: np.ndarray, weights: np.ndarray) -> str:
        values = np.concatenate(
            (
                np.asarray(points, dtype=float).reshape(-1),
                np.asarray(weights, dtype=float).reshape(-1),
            )
        )
        return exact_float64_key(values)

    def _build_missing(self, points: np.ndarray) -> None:
        ordered_missing: list[np.ndarray] = []
        seen: set[str] = set()
        for point in np.asarray(points, dtype=float):
            key = self._point_key(point)
            if key in self._nodes or key in seen:
                continue
            seen.add(key)
            ordered_missing.append(np.asarray(point, dtype=float))
        if not ordered_missing:
            return
        missing = np.stack(ordered_missing, axis=0)
        material_options = replace(self.options, collective_counterterm="none")
        dummy_weights = np.full(missing.shape[0], 1.0 / missing.shape[0], dtype=float)
        workspace = precompute_finite_q_material_workspace_batched(
            self.spec,
            self.ansatz,
            missing,
            dummy_weights,
            self.config,
            self.pairing,
            material_options,
        )
        self.material_batch_build_count += 1
        self.midpoint_eigh_call_count += 1
        if self._pairing_params is None:
            self._pairing_params = workspace.pairing_params
            self._collective_mode = workspace.collective_mode
            self._collective_mode_disabled_reason = (
                workspace.collective_mode_disabled_reason
            )
        for index, point in enumerate(missing):
            self._nodes[self._point_key(point)] = _NodeValue(
                energies=workspace.midpoint_energies[index],
                states=workspace.midpoint_states[index],
                occupations=workspace.midpoint_occupations[index],
            )
        self.node_misses += int(missing.shape[0])

    def material_workspace(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        *,
        include_counterterm: bool = False,
    ) -> FiniteQMaterialWorkspace:
        point_array = np.asarray(points, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        if point_array.ndim != 2 or point_array.shape[1] != 2 or point_array.shape[0] == 0:
            raise ValueError("adaptive material points must have shape (n,2)")
        if weight_array.shape != (point_array.shape[0],):
            raise ValueError("adaptive material weights must have shape (n,)")
        before = len(self._nodes)
        self._build_missing(point_array)
        self.node_hits += int(point_array.shape[0] - (len(self._nodes) - before))
        values = [self._nodes[self._point_key(point)] for point in point_array]
        counterterm = np.zeros((2, 2), dtype=complex)
        if include_counterterm:
            counterterm = self.counterterm(point_array, weight_array)
        if self._pairing_params is None or self._collective_mode is None:
            raise RuntimeError("adaptive node cache failed to initialize material state")
        return FiniteQMaterialWorkspace(
            spec=self.spec,
            ansatz=self.ansatz,
            k_points=point_array,
            k_weights=weight_array,
            config=self.config,
            pairing_params=self._pairing_params,
            options=self.options,
            collective_mode=self._collective_mode,
            collective_mode_disabled_reason=self._collective_mode_disabled_reason,
            midpoint_energies=np.stack([item.energies for item in values], axis=0),
            midpoint_states=np.stack([item.states for item in values], axis=0),
            midpoint_occupations=np.stack([item.occupations for item in values], axis=0),
            collective_counterterm_matrix=counterterm,
            metadata={
                "workspace_kind": "hierarchical_adaptive_material_node_view",
                "q_independent": True,
                "node_cache_schema": _NODE_CACHE_SCHEMA,
                "node_cache_fingerprint": self.fingerprint,
                "material_state_fingerprint": self.material_state_fingerprint,
                "node_count": int(point_array.shape[0]),
                "counterterm_included": bool(include_counterterm),
            },
        )

    def counterterm(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        self._build_missing(point_array)
        if self._pairing_params is None:
            raise RuntimeError("adaptive node cache has no resolved pairing parameters")
        key = self._counterterm_key(point_array, weight_array)
        cached = self._counterterms.get(key)
        if cached is not None:
            self.counterterm_hits += 1
            return np.asarray(cached, dtype=complex)
        value = np.asarray(
            self.ansatz.hs_counterterm(
                self.config,
                point_array,
                weight_array,
                self._pairing_params,
            ),
            dtype=complex,
        )
        if value.shape != (2, 2):
            raise ValueError("adaptive cell counterterm must have shape (2,2)")
        stored = _readonly(value, complex)
        self._counterterms[key] = stored
        self.counterterm_misses += 1
        return np.asarray(stored, dtype=complex)

    @property
    def pairing_params(self) -> object:
        if self._pairing_params is None:
            raise RuntimeError("adaptive node cache is not initialized")
        return self._pairing_params

    def metadata(self) -> dict[str, object]:
        return {
            "schema": _NODE_CACHE_SCHEMA,
            "fingerprint": self.fingerprint,
            "material_state_fingerprint": self.material_state_fingerprint,
            "entries": len(self._nodes),
            "node_hits": int(self.node_hits),
            "node_misses": int(self.node_misses),
            "material_batch_build_count": int(self.material_batch_build_count),
            "midpoint_eigh_call_count": int(self.midpoint_eigh_call_count),
            "counterterm_entries": len(self._counterterms),
            "counterterm_hits": int(self.counterterm_hits),
            "counterterm_misses": int(self.counterterm_misses),
            "exact_node_key": "canonicalized_ieee754_float64_bytes",
        }


_CACHE_COUNTER_KEYS = (
    "entries",
    "node_hits",
    "node_misses",
    "material_batch_build_count",
    "midpoint_eigh_call_count",
    "counterterm_entries",
    "counterterm_hits",
    "counterterm_misses",
    "counterterm_node_entries",
    "counterterm_node_hits",
    "counterterm_node_misses",
    "counterterm_q0_workspace_build_count",
    "counterterm_shifted_eigh_call_count",
)
_CACHE_TIME_KEYS = ("counterterm_q0_workspace_seconds",)


def material_node_cache_snapshot(cache: HierarchicalMaterialNodeCache) -> dict[str, float | int]:
    """Return stable numeric cache counters for per-call and worker telemetry."""

    metadata = cache.metadata()
    result: dict[str, float | int] = {}
    for key in _CACHE_COUNTER_KEYS:
        result[key] = int(metadata.get(key, 0))
    for key in _CACHE_TIME_KEYS:
        result[key] = float(metadata.get(key, 0.0))
    return result


def material_node_cache_delta(
    before: Mapping[str, float | int], after: Mapping[str, float | int]
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key in _CACHE_COUNTER_KEYS:
        result[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    for key in _CACHE_TIME_KEYS:
        result[key] = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
    return result


@dataclass(frozen=True)
class _PartitionedPrimitiveResult:
    packed_partitions: tuple[np.ndarray, ...]
    operator_ward: OperatorWardReport
    q_workspace_build_count: int
    shifted_eigh_call_count: int
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_pack_seconds: float


def _partition_workspace_view(workspace: object, start: int, stop: int) -> object:
    direct, phase_plus, ward_rhs = _integrated_linear_terms_from_workspace_slice(
        workspace, start, stop
    )
    zero = np.zeros((2, 2), dtype=complex)
    material = type("AdaptivePartitionMaterial", (), {})()
    material.collective_counterterm_matrix = zero
    view = type("AdaptivePartitionWorkspace", (), {})()
    view.direct_contact_contribution = direct
    view.phase_phase_direct_plus = phase_plus
    view.phase_phase_direct_minus = -phase_plus
    view.ward_rhs_vector = ward_rhs
    view.material = material
    return view


def _evaluate_partitioned_primitives(
    material: FiniteQMaterialWorkspace,
    q_model: np.ndarray,
    xi_values: np.ndarray,
    *,
    partitions: Sequence[tuple[int, int]],
    operator_ward_atol: float,
    operator_ward_rtol: float,
) -> _PartitionedPrimitiveResult:
    normalized = tuple((int(start), int(stop)) for start, stop in partitions)
    if not normalized or normalized[0][0] != 0 or normalized[-1][1] != material.nk:
        raise ValueError("adaptive partitions must cover the complete material batch")
    previous = 0
    for start, stop in normalized:
        if start != previous or stop <= start:
            raise ValueError("adaptive partitions must be contiguous and nonempty")
        previous = stop
    started = perf_counter()
    workspace = precompute_finite_q_q_workspace_batched(
        material, q_model, operator_diagnostics=True
    )
    q_seconds = perf_counter() - started
    operator = operator_ward_report_from_workspace(
        workspace, atol=operator_ward_atol, rtol=operator_ward_rtol
    )
    started = perf_counter()
    raw_factors = _vectorized_kubo_factors(workspace, xi_values)
    factor_seconds = perf_counter() - started
    packed_values: list[np.ndarray] = []
    contraction_seconds = 0.0
    pack_seconds = 0.0
    weights = np.asarray(material.k_weights, dtype=float)
    for start, stop in normalized:
        timer = perf_counter()
        weighted = 0.5 * weights[None, start:stop, None, None] * raw_factors[:, start:stop]
        blocks = np.einsum(
            "xkmn,kamn,kbmn->xab",
            weighted,
            workspace.left_vertices_band[start:stop],
            np.conjugate(workspace.right_vertices_band[start:stop]),
            optimize=True,
        )
        contraction_seconds += perf_counter() - timer
        timer = perf_counter()
        packed_values.append(
            np.asarray(
                pack_integrated_primitives(
                    workspace=_partition_workspace_view(workspace, start, stop),
                    blocks=blocks,
                    include_counterterm=False,
                ),
                dtype=complex,
            )
        )
        pack_seconds += perf_counter() - timer
    return _PartitionedPrimitiveResult(
        packed_partitions=tuple(packed_values),
        operator_ward=operator,
        q_workspace_build_count=int(workspace.metadata.get("q_workspace_build_count", 1)),
        shifted_eigh_call_count=int(workspace.metadata.get("shifted_eigh_call_count", 0)),
        q_workspace_seconds=float(q_seconds),
        kubo_factor_seconds=float(factor_seconds),
        kubo_contraction_seconds=float(contraction_seconds),
        primitive_pack_seconds=float(pack_seconds),
    )


def primitive_ward_residual_from_packed(
    packed: np.ndarray,
    *,
    xi_values: Sequence[float] | np.ndarray,
    q_model: np.ndarray,
    delta0_eV: float,
) -> np.ndarray:
    vector = np.asarray(packed, dtype=complex).reshape(-1)
    frequencies = np.asarray(xi_values, dtype=float)
    if vector.size != primitive_vector_width(int(frequencies.size)):
        raise ValueError("packed primitive width does not match frequency count")
    offset = 0
    direct = vector[offset : offset + 9].reshape(3, 3)
    offset += 9 + 4 + 2
    rhs = vector[offset : offset + 3]
    offset += 3
    residuals: list[np.ndarray] = []
    for frequency in frequencies:
        bubble = vector[offset : offset + 9].reshape(3, 3)
        offset += 9
        offset += 4
        em_collective_left = vector[offset : offset + 6].reshape(3, 2)
        offset += 6
        collective_em_right = vector[offset : offset + 6].reshape(2, 3)
        offset += 6
        u_left, u_right, w_left, w_right = primitive_ward_vectors_xy(
            float(frequency), q_model, float(delta0_eV)
        )
        k_ss = bubble + direct
        residuals.extend(
            (
                u_left @ k_ss + w_left @ collective_em_right - rhs,
                k_ss @ u_right + em_collective_left @ w_right - rhs,
            )
        )
    return np.concatenate(residuals)


@dataclass(frozen=True)
class _CellEvaluation:
    cell: DWaveCubatureCell
    low: np.ndarray
    high: np.ndarray
    low_ward: np.ndarray
    high_ward: np.ndarray

    def __post_init__(self) -> None:
        for name in ("low", "high", "low_ward", "high_ward"):
            object.__setattr__(self, name, _readonly(getattr(self, name), complex))


@dataclass(frozen=True)
class ArbitraryQVectorAdaptiveProfile:
    frequency_count: int
    iterations: int
    converged: bool
    stop_reason: str
    accepted_cell_count: int
    max_cell_level: int
    total_cell_evaluations: int
    total_point_evaluations: int
    low_rule_point_evaluations: int
    high_rule_point_evaluations: int
    q_workspace_build_count: int
    shifted_eigensystem_build_count: int
    midpoint_eigensystem_build_count: int
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_pack_seconds: float
    primitive_integration_seconds: float
    postprocess_seconds: float
    total_seconds: float
    conservative_error_ratio_max: float
    signed_error_ratio_max: float
    ward_error_ratio_conservative: float
    counterterm_add_count: int
    material_cache_fingerprint: str
    node_cache_hits: int
    node_cache_misses: int
    cache_delta: Mapping[str, float | int]
    cache_totals_after_call: Mapping[str, float | int]
    iteration_history: tuple[Mapping[str, object], ...]

    @property
    def k_point_count(self) -> int:
        return int(self.total_point_evaluations)

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_schema": "ArbitraryQVectorAdaptiveProfile-v2",
            "frequency_count": int(self.frequency_count),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "stop_reason": str(self.stop_reason),
            "accepted_cell_count": int(self.accepted_cell_count),
            "max_cell_level": int(self.max_cell_level),
            "total_cell_evaluations": int(self.total_cell_evaluations),
            "total_point_evaluations": int(self.total_point_evaluations),
            "low_rule_point_evaluations": int(self.low_rule_point_evaluations),
            "high_rule_point_evaluations": int(self.high_rule_point_evaluations),
            "q_workspace_build_count": int(self.q_workspace_build_count),
            "shifted_eigensystem_build_count": int(self.shifted_eigensystem_build_count),
            "midpoint_eigensystem_build_count": int(self.midpoint_eigensystem_build_count),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "kubo_contraction_seconds": float(self.kubo_contraction_seconds),
            "primitive_pack_seconds": float(self.primitive_pack_seconds),
            "primitive_integration_seconds": float(self.primitive_integration_seconds),
            "postprocess_seconds": float(self.postprocess_seconds),
            "total_seconds": float(self.total_seconds),
            "conservative_error_ratio_max": float(self.conservative_error_ratio_max),
            "signed_error_ratio_max": float(self.signed_error_ratio_max),
            "ward_error_ratio_conservative": float(self.ward_error_ratio_conservative),
            "counterterm_add_count": int(self.counterterm_add_count),
            "counterterm_cell_contribution_count": int(self.accepted_cell_count),
            "material_cache_fingerprint": self.material_cache_fingerprint,
            "node_cache_hits": int(self.node_cache_hits),
            "node_cache_misses": int(self.node_cache_misses),
            "cache_delta": dict(self.cache_delta),
            "cache_totals_after_call": dict(self.cache_totals_after_call),
            "iteration_history": [dict(row) for row in self.iteration_history],
        }


class AdaptiveConvergenceError(RuntimeError):
    pass


class _AdaptiveEvaluator:
    def __init__(
        self,
        *,
        node_cache: HierarchicalMaterialNodeCache,
        q_model: np.ndarray,
        xi_values: np.ndarray,
        options: ArbitraryQVectorAdaptiveOptions,
        operator_ward_atol: float,
        operator_ward_rtol: float,
    ) -> None:
        self.node_cache = node_cache
        self.q = np.asarray(q_model, dtype=float)
        self.xi = np.asarray(xi_values, dtype=float)
        self.options = options
        self.operator_ward_atol = float(operator_ward_atol)
        self.operator_ward_rtol = float(operator_ward_rtol)
        self.operator_reports: list[OperatorWardReport] = []
        self.total_cells = 0
        self.low_points = 0
        self.high_points = 0
        self.q_builds = 0
        self.shifted_builds = 0
        self.q_seconds = 0.0
        self.factor_seconds = 0.0
        self.contraction_seconds = 0.0
        self.pack_seconds = 0.0

    def evaluate_cells(
        self, cells: Sequence[DWaveCubatureCell]
    ) -> dict[DWaveCubatureCell, _CellEvaluation]:
        ordered = tuple(sorted(cells))
        output: dict[DWaveCubatureCell, _CellEvaluation] = {}
        batch_size = int(self.options.cell_batch_size)
        low_order = int(self.options.low_order)
        high_order = int(self.options.high_order)
        delta0 = float(getattr(self.node_cache.pairing_params, "delta0_eV")) if self.node_cache._pairing_params is not None else None
        for batch_start in range(0, len(ordered), batch_size):
            batch = tuple(ordered[batch_start : batch_start + batch_size])
            low_rules = [cubature_cell_gauss_rule(cell, low_order) for cell in batch]
            high_rules = [cubature_cell_gauss_rule(cell, high_order) for cell in batch]
            all_rules = [*low_rules, *high_rules]
            points = np.concatenate([item[0] for item in all_rules], axis=0)
            weights = np.concatenate([item[1] for item in all_rules], axis=0)
            partitions: list[tuple[int, int]] = []
            cursor = 0
            for rule in all_rules:
                stop = cursor + int(rule[0].shape[0])
                partitions.append((cursor, stop))
                cursor = stop
            material = self.node_cache.material_workspace(
                points, weights, include_counterterm=False
            )
            if delta0 is None:
                delta0 = float(getattr(self.node_cache.pairing_params, "delta0_eV"))
            result = _evaluate_partitioned_primitives(
                material,
                self.q,
                self.xi,
                partitions=partitions,
                operator_ward_atol=self.operator_ward_atol,
                operator_ward_rtol=self.operator_ward_rtol,
            )
            self.operator_reports.append(result.operator_ward)
            self.q_builds += result.q_workspace_build_count
            self.shifted_builds += result.shifted_eigh_call_count
            self.q_seconds += result.q_workspace_seconds
            self.factor_seconds += result.kubo_factor_seconds
            self.contraction_seconds += result.kubo_contraction_seconds
            self.pack_seconds += result.primitive_pack_seconds
            split = len(batch)
            low_packed = result.packed_partitions[:split]
            high_packed = result.packed_partitions[split:]
            for cell, low_rule, high_rule, low_value, high_value in zip(
                batch,
                low_rules,
                high_rules,
                low_packed,
                high_packed,
                strict=True,
            ):
                low = np.asarray(low_value, dtype=complex) + counterterm_primitive_vector(
                    self.node_cache.counterterm(low_rule[0], low_rule[1]),
                    frequency_count=int(self.xi.size),
                )
                high = np.asarray(high_value, dtype=complex) + counterterm_primitive_vector(
                    self.node_cache.counterterm(high_rule[0], high_rule[1]),
                    frequency_count=int(self.xi.size),
                )
                output[cell] = _CellEvaluation(
                    cell=cell,
                    low=low,
                    high=high,
                    low_ward=primitive_ward_residual_from_packed(
                        low, xi_values=self.xi, q_model=self.q, delta0_eV=float(delta0)
                    ),
                    high_ward=primitive_ward_residual_from_packed(
                        high, xi_values=self.xi, q_model=self.q, delta0_eV=float(delta0)
                    ),
                )
        self.total_cells += len(ordered)
        self.low_points += len(ordered) * low_order**2
        self.high_points += len(ordered) * high_order**2
        return output


def _error_state(
    active: Mapping[DWaveCubatureCell, _CellEvaluation],
    options: ArbitraryQVectorAdaptiveOptions,
) -> dict[str, Any]:
    ordered = [active[cell] for cell in sorted(active)]
    return vector_error_metrics(
        [item.low for item in ordered],
        [item.high for item in ordered],
        relative_tolerance=float(options.relative_tolerance),
        absolute_tolerance=float(options.absolute_tolerance),
        low_ward_vectors=[item.low_ward for item in ordered],
        high_ward_vectors=[item.high_ward for item in ordered],
        ward_threshold=float(options.ward_error_tolerance),
    )


def _converged(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["conservative_error_ratio_max"]) <= 1.0
        and float(metrics["ward_error_ratio_conservative"]) <= 1.0
    )


def _refinement_selection(
    active: Mapping[DWaveCubatureCell, _CellEvaluation],
    metrics: Mapping[str, Any],
    options: ArbitraryQVectorAdaptiveOptions,
    evaluated_points: int,
) -> tuple[DWaveCubatureCell, ...]:
    cells = sorted(active)
    scores = np.asarray(metrics["cell_scores"], dtype=float)
    candidates = [
        (float(score), cell)
        for cell, score in zip(cells, scores, strict=True)
        if int(cell.level) < int(options.max_level)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        return ()
    desired = max(
        int(options.min_refine_cells),
        int(ceil(float(options.refine_fraction) * len(active))),
    )
    room_by_cells = max((int(options.max_cells) - len(active)) // 3, 0)
    points_per_parent = 4 * (
        int(options.low_order) ** 2 + int(options.high_order) ** 2
    )
    room_by_points = max(
        (int(options.max_evaluation_points) - int(evaluated_points))
        // points_per_parent,
        0,
    )
    count = min(desired, len(candidates), room_by_cells, room_by_points)
    return tuple(item[1] for item in candidates[:count])


def _sum_high_primitives(
    active: Mapping[DWaveCubatureCell, _CellEvaluation], frequency_count: int
) -> np.ndarray:
    accumulator = ComplexKahanVector(primitive_vector_width(int(frequency_count)))
    for cell in sorted(active):
        accumulator.add(active[cell].high)
    return accumulator.value()


class ArbitraryQVectorAdaptiveResponseCache:
    """Exact-q cache including the complete adaptive numerical definition."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, ...], ArbitraryQPeriodicBZResult] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        node_cache_fingerprint: str,
        q_model: np.ndarray,
        xi_values: np.ndarray,
        options: ArbitraryQVectorAdaptiveOptions,
        *,
        phase_policy: str,
        operator_ward_atol: float,
        operator_ward_rtol: float,
    ) -> tuple[str, ...]:
        return (
            _RESPONSE_CACHE_SCHEMA,
            str(node_cache_fingerprint),
            exact_float64_key(np.asarray(q_model, dtype=float)),
            exact_float64_key(np.asarray(xi_values, dtype=float)),
            options.fingerprint,
            str(phase_policy),
            exact_float64_key(np.asarray([operator_ward_atol, operator_ward_rtol])),
            PRIMITIVE_CONTRACT_VERSION,
        )

    def get(self, *args: Any, **kwargs: Any) -> ArbitraryQPeriodicBZResult | None:
        value = self._values.get(self.key(*args, **kwargs))
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(
        self,
        result: ArbitraryQPeriodicBZResult,
        options: ArbitraryQVectorAdaptiveOptions,
    ) -> None:
        metadata = result.metadata
        self._values[
            self.key(
                result.material_cache_fingerprint,
                result.q_model,
                result.xi_eV_values,
                options,
                phase_policy=str(metadata["post_integral_phase_hessian_policy"]),
                operator_ward_atol=result.operator_ward.atol,
                operator_ward_rtol=result.operator_ward.rtol,
            )
        ] = result

    def metadata(self) -> dict[str, int | str]:
        return {
            "schema": _RESPONSE_CACHE_SCHEMA,
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


def _phase_policy(pairing_name: str) -> str:
    return "nearest_neighbor_bond_metric" if pairing_name == "dwave" else "q_independent"


def build_hierarchical_material_node_cache(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    temperature_K: float,
    eta_eV: float,
) -> HierarchicalMaterialNodeCache:
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    return HierarchicalMaterialNodeCache(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )


def prewarm_initial_adaptive_nodes(
    node_cache: HierarchicalMaterialNodeCache,
    options: ArbitraryQVectorAdaptiveOptions,
) -> dict[str, object]:
    """Build the deterministic initial low/high node union before POSIX fork."""

    options.validate()
    cells = tuple(initial_cubature_cells(int(options.coarse_grid)))
    rules = [
        cubature_cell_gauss_rule(cell, order)
        for order in (int(options.low_order), int(options.high_order))
        for cell in cells
    ]
    points = np.concatenate([rule[0] for rule in rules], axis=0)
    weights = np.concatenate([rule[1] for rule in rules], axis=0)
    before = material_node_cache_snapshot(node_cache)
    started = perf_counter()
    node_cache.material_workspace(points, weights, include_counterterm=False)
    seconds = perf_counter() - started
    after = material_node_cache_snapshot(node_cache)
    return {
        "seconds": float(seconds),
        "point_requests": int(points.shape[0]),
        "unique_nodes_after": int(after["entries"]),
        "cache_delta": material_node_cache_delta(before, after),
        "cache_snapshot_after": after,
    }


def _nonconvergence_message(profile: ArbitraryQVectorAdaptiveProfile) -> str:
    return (
        "vector-adaptive cubature did not converge: "
        f"stop_reason={profile.stop_reason}, cells={profile.accepted_cell_count}, "
        f"points={profile.total_point_evaluations}, contract_ratio="
        f"{profile.conservative_error_ratio_max:.3e}, ward_ratio="
        f"{profile.ward_error_ratio_conservative:.3e}"
    )


def integrate_arbitrary_q_vector_adaptive(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    q_model: np.ndarray,
    adaptive_options: ArbitraryQVectorAdaptiveOptions | None = None,
    node_cache: HierarchicalMaterialNodeCache | None = None,
    response_cache: ArbitraryQVectorAdaptiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
    require_converged: bool = True,
) -> ArbitraryQPeriodicBZResult:
    xi = _validate_xi(xi_eV_values)
    q = validate_q_domain(np.asarray(q_model, dtype=float))
    settings = adaptive_options or ArbitraryQVectorAdaptiveOptions()
    settings.validate()
    pairing_name = str(getattr(ansatz, "name", ""))
    if pairing_name not in {"spm", "dwave"}:
        raise ValueError("arbitrary-q vector adaptive supports spm and dwave")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("arbitrary-q vector adaptive requires bond_endpoint_gauge")
    base_config = KuboConfig.from_kelvin(
        omega_eV=float(xi[0]),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    engine_options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    phase_policy = _phase_policy(pairing_name)
    expected_state = material_state_fingerprint(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        config=base_config,
        options=engine_options,
    )
    cache = node_cache or build_hierarchical_material_node_cache(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    if cache.material_state_fingerprint != expected_state:
        raise ValueError("provided adaptive node cache does not match material state")
    if response_cache is not None:
        cached = response_cache.get(
            cache.fingerprint,
            q,
            xi,
            settings,
            phase_policy=phase_policy,
            operator_ward_atol=operator_ward_atol,
            operator_ward_rtol=operator_ward_rtol,
        )
        if cached is not None:
            if require_converged and not bool(cached.profile.converged):
                raise AdaptiveConvergenceError(_nonconvergence_message(cached.profile))
            return cached

    call_started = perf_counter()
    cache_before = material_node_cache_snapshot(cache)
    evaluator = _AdaptiveEvaluator(
        node_cache=cache,
        q_model=q,
        xi_values=xi,
        options=settings,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
    )
    initial = tuple(initial_cubature_cells(int(settings.coarse_grid)))
    initial_points = len(initial) * (
        int(settings.low_order) ** 2 + int(settings.high_order) ** 2
    )
    if len(initial) > int(settings.max_cells) or initial_points > int(
        settings.max_evaluation_points
    ):
        raise ValueError("adaptive resource limits are smaller than the initial grid")

    primitive_started = perf_counter()
    active = evaluator.evaluate_cells(initial)
    evaluated_points = evaluator.low_points + evaluator.high_points
    metrics = _error_state(active, settings)
    iterations = 0
    stop_reason = "converged"
    history: list[dict[str, object]] = []
    while True:
        scores = np.asarray(metrics["cell_scores"], dtype=float)
        history_row: dict[str, object] = {
            "iteration": int(iterations),
            "active_cells": len(active),
            "selected_cells": 0,
            "max_cell_score": float(np.max(scores)) if scores.size else 0.0,
            "median_cell_score": float(np.median(scores)) if scores.size else 0.0,
            "p90_cell_score": float(np.quantile(scores, 0.9)) if scores.size else 0.0,
            "conservative_error_ratio_max": float(
                metrics["conservative_error_ratio_max"]
            ),
            "ward_error_ratio_conservative": float(
                metrics["ward_error_ratio_conservative"]
            ),
            "evaluated_points": int(evaluated_points),
        }
        history.append(history_row)
        if _converged(metrics):
            stop_reason = "converged"
            break
        if iterations >= int(settings.max_iterations):
            stop_reason = "max_iterations"
            break
        selected = _refinement_selection(active, metrics, settings, evaluated_points)
        history_row["selected_cells"] = len(selected)
        if not selected:
            if len(active) >= int(settings.max_cells):
                stop_reason = "max_cells"
            elif evaluated_points >= int(settings.max_evaluation_points):
                stop_reason = "max_evaluation_points"
            else:
                stop_reason = "max_level_or_no_refinable_cells"
            break
        children: list[DWaveCubatureCell] = []
        before_points = evaluated_points
        for parent in selected:
            children.extend(subdivide_cubature_cell(parent))
            del active[parent]
        active.update(evaluator.evaluate_cells(children))
        evaluated_points = evaluator.low_points + evaluator.high_points
        history_row["new_points"] = int(evaluated_points - before_points)
        iterations += 1
        metrics = _error_state(active, settings)

    converged = _converged(metrics)
    packed = _sum_high_primitives(active, int(xi.size))
    operator = combine_operator_ward_reports(evaluator.operator_reports)
    primitive_seconds = perf_counter() - primitive_started
    cache_after_primitive = material_node_cache_snapshot(cache)
    cache_delta = material_node_cache_delta(cache_before, cache_after_primitive)

    integration_metadata: dict[str, object] = {
        "integration_strategy": "arbitrary_q_vector_adaptive_hierarchical_cubature",
        "arbitrary_q_contract": _ADAPTIVE_CONTRACT,
        "primitive_contract_version": PRIMITIVE_CONTRACT_VERSION,
        "exact_q_used_without_rounding": True,
        "q_wrapping_forbidden": True,
        "principal_q_domain_kind": "syntactically_supported_not_numerically_qualified",
        "supported_q_component_limit": SUPPORTED_Q_COMPONENT_LIMIT,
        "numerically_qualified_q_envelope_established": False,
        "translation_by_q_is_exact_orbit_permutation": False,
        "matsubara_batch_shared_nodes": True,
        "all_frequencies_share_one_adaptive_tree": True,
        "low_high_rules_share_one_q_workspace_per_cell_batch": True,
        "low_high_rule_relation": "paired_nonembedded_tensor_gauss",
        "zero_and_positive_frequencies_share_eigensystems": bool(
            np.any(xi == 0.0) and np.any(xi > 0.0)
        ),
        "exact_zero_uses_divided_difference": bool(np.any(xi == 0.0)),
        "conductivity_division_for_zero_forbidden": True,
        "post_integral_phase_hessian_policy": phase_policy,
        "primitive_vector_integrated_before_schur": True,
        "cell_schur_forbidden": True,
        "adaptive_options": settings.as_dict(),
        "adaptive_converged": bool(converged),
        "adaptive_stop_reason": stop_reason,
        "material_cache_fingerprint": cache.fingerprint,
        "material_state_fingerprint": cache.material_state_fingerprint,
        "grid_fingerprint": "hierarchical_adaptive_cells_no_fixed_grid",
        "grid": {
            "grid_contract": _ADAPTIVE_CONTRACT,
            "coarse_grid": int(settings.coarse_grid),
            "accepted_cells": len(active),
            "max_level": max((int(cell.level) for cell in active), default=0),
            "weights_equal": False,
            "full_bz_covered_by_disjoint_cells": True,
        },
        "node_cache": cache.metadata(),
        "cache_delta": cache_delta,
        "iteration_history": history,
        "counterterm_add_count": 1,
        "counterterm_integrated_per_accepted_cell_before_full_sum": True,
        "operator_ward": operator.as_dict(),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    post_started = perf_counter()
    components, rhs = unpack_integrated_primitives(
        packed,
        xi_values=xi,
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        q_model=q,
        options=engine_options,
        phase_hessian_policy=phase_policy,
        integration_metadata=integration_metadata,
        rhs_source="arbitrary_q_vector_adaptive_full_bz_integral",
    )
    postprocess_seconds = perf_counter() - post_started
    total_seconds = perf_counter() - call_started
    cache_after = material_node_cache_snapshot(cache)
    profile = ArbitraryQVectorAdaptiveProfile(
        frequency_count=int(xi.size),
        iterations=int(iterations),
        converged=bool(converged),
        stop_reason=stop_reason,
        accepted_cell_count=len(active),
        max_cell_level=max((int(cell.level) for cell in active), default=0),
        total_cell_evaluations=int(evaluator.total_cells),
        total_point_evaluations=int(evaluated_points),
        low_rule_point_evaluations=int(evaluator.low_points),
        high_rule_point_evaluations=int(evaluator.high_points),
        q_workspace_build_count=int(evaluator.q_builds),
        shifted_eigensystem_build_count=int(evaluator.shifted_builds),
        midpoint_eigensystem_build_count=int(
            cache_delta.get("midpoint_eigh_call_count", 0)
        ),
        q_workspace_seconds=float(evaluator.q_seconds),
        kubo_factor_seconds=float(evaluator.factor_seconds),
        kubo_contraction_seconds=float(evaluator.contraction_seconds),
        primitive_pack_seconds=float(evaluator.pack_seconds),
        primitive_integration_seconds=float(primitive_seconds),
        postprocess_seconds=float(postprocess_seconds),
        total_seconds=float(total_seconds),
        conservative_error_ratio_max=float(metrics["conservative_error_ratio_max"]),
        signed_error_ratio_max=float(metrics["signed_error_ratio_max"]),
        ward_error_ratio_conservative=float(metrics["ward_error_ratio_conservative"]),
        counterterm_add_count=1,
        material_cache_fingerprint=cache.fingerprint,
        node_cache_hits=int(cache_delta.get("node_hits", 0)),
        node_cache_misses=int(cache_delta.get("node_misses", 0)),
        cache_delta=cache_delta,
        cache_totals_after_call=cache_after,
        iteration_history=tuple(history),
    )
    integration_metadata["accumulation_profile"] = profile.as_dict()
    integration_metadata["node_cache"] = cache.metadata()
    result = ArbitraryQPeriodicBZResult(
        q_model=q,
        xi_eV_values=xi,
        packed_primitives=packed,
        components=components,
        rhs=rhs,
        operator_ward=operator,
        profile=profile,
        material_cache_fingerprint=cache.fingerprint,
        metadata=integration_metadata,
    )
    if require_converged and not converged:
        raise AdaptiveConvergenceError(_nonconvergence_message(profile))
    if response_cache is not None:
        response_cache.put(result, settings)
    return result


def integrate_two_plate_angle_batch_vector_adaptive(
    *,
    q_lab: np.ndarray,
    theta_1_rad: float,
    theta_2_rad_values: Sequence[float] | np.ndarray,
    node_cache: HierarchicalMaterialNodeCache,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    adaptive_options: ArbitraryQVectorAdaptiveOptions | None = None,
    response_cache: ArbitraryQVectorAdaptiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
    require_converged: bool = True,
) -> TwoPlateAngleBatchResult:
    q = np.asarray(q_lab, dtype=float)
    angles = np.asarray(theta_2_rad_values, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_lab must be a finite vector with shape (2,)")
    if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
        raise ValueError("theta_2_rad_values must be a nonempty finite vector")
    local_cache = response_cache or ArbitraryQVectorAdaptiveResponseCache()
    common = dict(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        adaptive_options=adaptive_options,
        node_cache=node_cache,
        response_cache=local_cache,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
        require_converged=require_converged,
    )
    plate_1 = integrate_arbitrary_q_vector_adaptive(
        q_model=rotate_lab_q_to_crystal(q, float(theta_1_rad)), **common
    )
    plate_2 = tuple(
        integrate_arbitrary_q_vector_adaptive(
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
    "AdaptiveConvergenceError",
    "ArbitraryQVectorAdaptiveOptions",
    "ArbitraryQVectorAdaptiveProfile",
    "ArbitraryQVectorAdaptiveResponseCache",
    "HierarchicalMaterialNodeCache",
    "build_hierarchical_material_node_cache",
    "integrate_arbitrary_q_vector_adaptive",
    "integrate_two_plate_angle_batch_vector_adaptive",
    "material_node_cache_delta",
    "material_node_cache_snapshot",
    "prewarm_initial_adaptive_nodes",
    "primitive_ward_residual_from_packed",
]
