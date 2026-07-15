"""Successive-high convergence controller for arbitrary-q vector-adaptive cubature.

This controller deliberately reuses the established adaptive cell evaluator, node
cache, primitive kernel, refinement selection, and full-BZ high-rule accumulation.
It changes only convergence semantics:

* local primitive low/high differences rank cells;
* same-iteration low/high primitive and Ward differences remain diagnostics;
* accepted high-rule full-BZ primitives must be stable for consecutive iterations;
* the current accepted high-rule result must pass the integrated Ward identity.

The module is diagnostic-only and does not authorize Casimir input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_accumulator import combine_operator_ward_reports
from lno327.response.arbitrary_q_formal_policy import (
    PRIMITIVE_CONTRACT_VERSION,
    SUPPORTED_Q_COMPONENT_LIMIT,
    validate_q_domain,
)
from lno327.response.arbitrary_q_material_cache import material_state_fingerprint
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.periodic_bz_grid import exact_float64_key
from lno327.response.primitive_kernel import unpack_integrated_primitives
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.arbitrary_q_matsubara import (
    ArbitraryQPeriodicBZResult,
    TwoPlateAngleBatchResult,
    rotate_lab_q_to_crystal,
)
from lno327.workflows.arbitrary_q_vector_adaptive import (
    AdaptiveConvergenceError,
    ArbitraryQVectorAdaptiveOptions,
    ArbitraryQVectorAdaptiveProfile,
    HierarchicalMaterialNodeCache,
    _AdaptiveEvaluator,
    _error_state,
    _phase_policy,
    _refinement_selection,
    _sum_high_primitives,
    _validate_xi,
    build_hierarchical_material_node_cache,
    initial_cubature_cells,
    material_node_cache_delta,
    material_node_cache_snapshot,
    subdivide_cubature_cell,
)
from lno327.workflows.dwave_vector_adaptive_cubature import vector_error_metrics
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

_FLOAT_EPS = np.finfo(float).eps
_SUCCESSIVE_CONTRACT = "ArbitraryQVectorAdaptiveContract-v3-successive-high"
_SUCCESSIVE_RESPONSE_CACHE_SCHEMA = "ArbitraryQVectorAdaptiveResponseCache-v3"


def _hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class ArbitraryQVectorAdaptiveSuccessiveOptions(ArbitraryQVectorAdaptiveOptions):
    """Adaptive definition with successive-high and integrated-Ward hard gates."""

    successive_stable_iterations: int = 2
    integrated_ward_tolerance: float = 1e-7
    integrated_ward_absolute_tolerance: float = 1e-12
    integrated_ward_condition_max: float = 1e12

    def validate(self) -> None:
        super().validate()
        if int(self.successive_stable_iterations) <= 0:
            raise ValueError("successive_stable_iterations must be positive")
        for name in (
            "integrated_ward_tolerance",
            "integrated_ward_absolute_tolerance",
            "integrated_ward_condition_max",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.integrated_ward_condition_max) <= 0.0:
            raise ValueError("integrated_ward_condition_max must be positive")

    def numerical_definition(self) -> dict[str, object]:
        definition = dict(super().numerical_definition())
        definition.update(
            {
                "contract": _SUCCESSIVE_CONTRACT,
                "convergence_estimator": "successive_accepted_high_rule_group_mixed",
                "successive_stable_iterations": int(self.successive_stable_iterations),
                "integrated_ward_gate": "current_high_rule_effective_ward_xy",
                "integrated_ward_tolerance": float(
                    self.integrated_ward_tolerance
                ).hex(),
                "integrated_ward_absolute_tolerance": float(
                    self.integrated_ward_absolute_tolerance
                ).hex(),
                "integrated_ward_condition_max": float(
                    self.integrated_ward_condition_max
                ).hex(),
                "same_iteration_low_high_is_diagnostic_only": True,
                "local_ward_is_diagnostic_only": True,
            }
        )
        return definition


@dataclass(frozen=True)
class ArbitraryQVectorAdaptiveSuccessiveProfile(ArbitraryQVectorAdaptiveProfile):
    successive_high_error_ratio_max: float = float("nan")
    successive_high_stable_streak: int = 0
    successive_high_required_streak: int = 2
    high_rule_integrated_ward_all_passed: bool = False
    high_rule_ward_effective_mixed_ratio_max: float = float("inf")
    high_rule_schur_condition_number_max: float = float("inf")
    high_rule_ward_by_frequency: tuple[Mapping[str, object], ...] = ()
    iteration_postprocess_seconds: float = 0.0
    final_postprocess_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        payload = dict(super().as_dict())
        payload.update(
            {
                "profile_schema": "ArbitraryQVectorAdaptiveProfile-v3-successive-high",
                "convergence_contract": _SUCCESSIVE_CONTRACT,
                "successive_high_error_ratio_max": float(
                    self.successive_high_error_ratio_max
                ),
                "successive_high_stable_streak": int(
                    self.successive_high_stable_streak
                ),
                "successive_high_required_streak": int(
                    self.successive_high_required_streak
                ),
                "high_rule_integrated_ward_all_passed": bool(
                    self.high_rule_integrated_ward_all_passed
                ),
                "high_rule_ward_effective_mixed_ratio_max": float(
                    self.high_rule_ward_effective_mixed_ratio_max
                ),
                "high_rule_schur_condition_number_max": float(
                    self.high_rule_schur_condition_number_max
                ),
                "high_rule_ward_by_frequency": [
                    dict(row) for row in self.high_rule_ward_by_frequency
                ],
                "iteration_postprocess_seconds": float(
                    self.iteration_postprocess_seconds
                ),
                "final_postprocess_seconds": float(self.final_postprocess_seconds),
                "same_iteration_low_high_hard_gate": False,
                "same_iteration_low_high_ward_hard_gate": False,
            }
        )
        return payload


@dataclass(frozen=True)
class _HighRuleWardState:
    passed: bool
    ratio_max: float
    condition_max: float
    rows: tuple[Mapping[str, object], ...]


class ArbitraryQVectorAdaptiveSuccessiveResponseCache:
    """Exact-q response cache isolated from legacy adaptive convergence semantics."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, ...], ArbitraryQPeriodicBZResult] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        node_cache_fingerprint: str,
        q_model: np.ndarray,
        xi_values: np.ndarray,
        options: ArbitraryQVectorAdaptiveSuccessiveOptions,
        *,
        phase_policy: str,
        operator_ward_atol: float,
        operator_ward_rtol: float,
    ) -> tuple[str, ...]:
        return (
            _SUCCESSIVE_RESPONSE_CACHE_SCHEMA,
            str(node_cache_fingerprint),
            exact_float64_key(np.asarray(q_model, dtype=float)),
            exact_float64_key(np.asarray(xi_values, dtype=float)),
            options.fingerprint,
            str(phase_policy),
            exact_float64_key(
                np.asarray([operator_ward_atol, operator_ward_rtol], dtype=float)
            ),
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
        options: ArbitraryQVectorAdaptiveSuccessiveOptions,
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
            "schema": _SUCCESSIVE_RESPONSE_CACHE_SCHEMA,
            "entries": len(self._values),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "q_key": "canonicalized_ieee754_float64_bytes",
        }


def successive_high_error_metrics(
    previous_high: np.ndarray,
    current_high: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Compare two accepted full-BZ high-rule primitive vectors by physical block."""

    return vector_error_metrics(
        (np.asarray(previous_high, dtype=complex),),
        (np.asarray(current_high, dtype=complex),),
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
    )


def update_successive_high_streak(previous_streak: int, ratio: float) -> int:
    """Advance or reset the consecutive-stability counter."""

    value = float(ratio)
    if np.isfinite(value) and value <= 1.0:
        return int(previous_streak) + 1
    return 0


def successive_high_controller_converged(
    *,
    stable_streak: int,
    required_streak: int,
    high_rule_ward_passed: bool,
) -> bool:
    return bool(
        int(stable_streak) >= int(required_streak) and bool(high_rule_ward_passed)
    )


def _iteration_unpack(
    packed: np.ndarray,
    *,
    xi: np.ndarray,
    ansatz: object,
    pairing: object,
    base_config: KuboConfig,
    q: np.ndarray,
    engine_options: FiniteQEngineOptions,
    phase_policy: str,
    iteration: int,
):
    return unpack_integrated_primitives(
        packed,
        xi_values=xi,
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        q_model=q,
        options=engine_options,
        phase_hessian_policy=phase_policy,
        integration_metadata={
            "integration_strategy": "adaptive_iteration_high_rule_ward_probe",
            "adaptive_contract": _SUCCESSIVE_CONTRACT,
            "adaptive_iteration": int(iteration),
            "diagnostic_only": True,
        },
        rhs_source="arbitrary_q_vector_adaptive_iteration_high_rule",
    )


def _evaluate_high_rule_ward(
    packed: np.ndarray,
    *,
    xi: np.ndarray,
    ansatz: object,
    pairing: object,
    base_config: KuboConfig,
    q: np.ndarray,
    engine_options: FiniteQEngineOptions,
    phase_policy: str,
    settings: ArbitraryQVectorAdaptiveSuccessiveOptions,
    iteration: int,
) -> _HighRuleWardState:
    try:
        components, rhs_values = _iteration_unpack(
            packed,
            xi=xi,
            ansatz=ansatz,
            pairing=pairing,
            base_config=base_config,
            q=q,
            engine_options=engine_options,
            phase_policy=phase_policy,
            iteration=iteration,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _HighRuleWardState(
            passed=False,
            ratio_max=float("inf"),
            condition_max=float("inf"),
            rows=(
                {
                    "n_index": -1,
                    "xi_eV": float("nan"),
                    "passed": False,
                    "effective_mixed_ratio_max": float("inf"),
                    "schur_condition_number": float("inf"),
                    "error": str(exc),
                },
            ),
        )

    rows: list[Mapping[str, object]] = []
    ratios: list[float] = []
    conditions: list[float] = []
    all_passed = True
    for index, (frequency, component, rhs) in enumerate(
        zip(xi, components, rhs_values, strict=True)
    ):
        try:
            kernel = effective_em_kernel_from_components(
                component,
                q_model=q,
                xi_eV=float(frequency),
            )
            report = validate_effective_ward_xy(
                kernel,
                rhs,
                residual_tolerance=float(settings.integrated_ward_tolerance),
                absolute_residual_tolerance=float(
                    settings.integrated_ward_absolute_tolerance
                ),
                condition_max=float(settings.integrated_ward_condition_max),
            )
            ratio = max(
                float(report.left.effective_mixed_ratio),
                float(report.right.effective_mixed_ratio),
            )
            condition = float(report.schur_condition_number)
            passed = bool(report.passed)
            error = ""
        except (ValueError, np.linalg.LinAlgError) as exc:
            ratio = float("inf")
            condition = float("inf")
            passed = False
            error = str(exc)
        ratios.append(ratio)
        conditions.append(condition)
        all_passed = all_passed and passed
        rows.append(
            {
                "n_index": int(index),
                "xi_eV": float(frequency),
                "passed": bool(passed),
                "effective_mixed_ratio_max": float(ratio),
                "schur_condition_number": float(condition),
                "error": error,
            }
        )
    return _HighRuleWardState(
        passed=bool(all_passed),
        ratio_max=max(ratios, default=float("inf")),
        condition_max=max(conditions, default=float("inf")),
        rows=tuple(rows),
    )


def _nonconvergence_message(profile: ArbitraryQVectorAdaptiveSuccessiveProfile) -> str:
    return (
        "vector-adaptive successive-high cubature did not converge: "
        f"stop_reason={profile.stop_reason}, cells={profile.accepted_cell_count}, "
        f"points={profile.total_point_evaluations}, successive_ratio="
        f"{profile.successive_high_error_ratio_max:.3e}, stable_streak="
        f"{profile.successive_high_stable_streak}/"
        f"{profile.successive_high_required_streak}, high_rule_ward_ratio="
        f"{profile.high_rule_ward_effective_mixed_ratio_max:.3e}"
    )


def integrate_arbitrary_q_vector_adaptive_successive(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    q_model: np.ndarray,
    adaptive_options: ArbitraryQVectorAdaptiveSuccessiveOptions | None = None,
    node_cache: HierarchicalMaterialNodeCache | None = None,
    response_cache: ArbitraryQVectorAdaptiveSuccessiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
    require_converged: bool = True,
) -> ArbitraryQPeriodicBZResult:
    xi = _validate_xi(xi_eV_values)
    q = validate_q_domain(np.asarray(q_model, dtype=float))
    settings = adaptive_options or ArbitraryQVectorAdaptiveSuccessiveOptions()
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
    stop_reason = "not_started"
    history: list[dict[str, object]] = []
    previous_high: np.ndarray | None = None
    current_high = _sum_high_primitives(active, int(xi.size))
    successive_ratio = float("nan")
    stable_streak = 0
    iteration_postprocess_seconds = 0.0
    high_ward = _HighRuleWardState(False, float("inf"), float("inf"), ())

    while True:
        current_high = _sum_high_primitives(active, int(xi.size))
        if previous_high is None:
            successive_ratio = float("nan")
            stable_streak = 0
        else:
            successive = successive_high_error_metrics(
                previous_high,
                current_high,
                relative_tolerance=float(settings.relative_tolerance),
                absolute_tolerance=float(settings.absolute_tolerance),
            )
            successive_ratio = float(successive["global_group_error_ratio_max"])
            stable_streak = update_successive_high_streak(
                stable_streak, successive_ratio
            )

        ward_started = perf_counter()
        high_ward = _evaluate_high_rule_ward(
            current_high,
            xi=xi,
            ansatz=ansatz,
            pairing=pairing,
            base_config=base_config,
            q=q,
            engine_options=engine_options,
            phase_policy=phase_policy,
            settings=settings,
            iteration=iterations,
        )
        iteration_postprocess_seconds += perf_counter() - ward_started

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
            "successive_high_error_ratio_max": float(successive_ratio),
            "successive_high_stable_streak": int(stable_streak),
            "successive_high_required_streak": int(
                settings.successive_stable_iterations
            ),
            "high_rule_integrated_ward_all_passed": bool(high_ward.passed),
            "high_rule_ward_effective_mixed_ratio_max": float(
                high_ward.ratio_max
            ),
            "high_rule_schur_condition_number_max": float(
                high_ward.condition_max
            ),
            "high_rule_ward_by_frequency": [dict(row) for row in high_ward.rows],
            "evaluated_points": int(evaluated_points),
        }
        history.append(history_row)

        if successive_high_controller_converged(
            stable_streak=stable_streak,
            required_streak=int(settings.successive_stable_iterations),
            high_rule_ward_passed=high_ward.passed,
        ):
            stop_reason = "converged_successive_high_and_integrated_ward"
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

        previous_high = np.array(current_high, dtype=complex, copy=True)
        children = []
        before_points = evaluated_points
        for parent in selected:
            children.extend(subdivide_cubature_cell(parent))
            del active[parent]
        active.update(evaluator.evaluate_cells(children))
        evaluated_points = evaluator.low_points + evaluator.high_points
        history_row["new_points"] = int(evaluated_points - before_points)
        iterations += 1
        metrics = _error_state(active, settings)

    converged = successive_high_controller_converged(
        stable_streak=stable_streak,
        required_streak=int(settings.successive_stable_iterations),
        high_rule_ward_passed=high_ward.passed,
    )
    packed = np.asarray(current_high, dtype=complex)
    operator = combine_operator_ward_reports(evaluator.operator_reports)
    primitive_seconds = perf_counter() - primitive_started
    cache_after_primitive = material_node_cache_snapshot(cache)
    cache_delta = material_node_cache_delta(cache_before, cache_after_primitive)

    integration_metadata: dict[str, object] = {
        "integration_strategy": "arbitrary_q_vector_adaptive_successive_high",
        "arbitrary_q_contract": _SUCCESSIVE_CONTRACT,
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
        "same_iteration_low_high_is_diagnostic_only": True,
        "successive_accepted_high_rule_is_numerical_hard_gate": True,
        "current_high_rule_integrated_ward_is_physical_hard_gate": True,
        "local_ward_is_diagnostic_only": True,
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
            "grid_contract": _SUCCESSIVE_CONTRACT,
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
        "high_rule_integrated_ward": {
            "all_passed": bool(high_ward.passed),
            "effective_mixed_ratio_max": float(high_ward.ratio_max),
            "schur_condition_number_max": float(high_ward.condition_max),
            "by_frequency": [dict(row) for row in high_ward.rows],
        },
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }

    final_post_started = perf_counter()
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
        rhs_source="arbitrary_q_vector_adaptive_successive_full_bz_integral",
    )
    final_postprocess_seconds = perf_counter() - final_post_started
    total_seconds = perf_counter() - call_started
    cache_after = material_node_cache_snapshot(cache)

    profile = ArbitraryQVectorAdaptiveSuccessiveProfile(
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
        postprocess_seconds=float(
            iteration_postprocess_seconds + final_postprocess_seconds
        ),
        total_seconds=float(total_seconds),
        conservative_error_ratio_max=float(metrics["conservative_error_ratio_max"]),
        signed_error_ratio_max=float(metrics["signed_error_ratio_max"]),
        ward_error_ratio_conservative=float(
            metrics["ward_error_ratio_conservative"]
        ),
        counterterm_add_count=1,
        material_cache_fingerprint=cache.fingerprint,
        node_cache_hits=int(cache_delta.get("node_hits", 0)),
        node_cache_misses=int(cache_delta.get("node_misses", 0)),
        cache_delta=cache_delta,
        cache_totals_after_call=cache_after,
        iteration_history=tuple(history),
        successive_high_error_ratio_max=float(successive_ratio),
        successive_high_stable_streak=int(stable_streak),
        successive_high_required_streak=int(settings.successive_stable_iterations),
        high_rule_integrated_ward_all_passed=bool(high_ward.passed),
        high_rule_ward_effective_mixed_ratio_max=float(high_ward.ratio_max),
        high_rule_schur_condition_number_max=float(high_ward.condition_max),
        high_rule_ward_by_frequency=tuple(high_ward.rows),
        iteration_postprocess_seconds=float(iteration_postprocess_seconds),
        final_postprocess_seconds=float(final_postprocess_seconds),
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


def integrate_two_plate_angle_batch_vector_adaptive_successive(
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
    adaptive_options: ArbitraryQVectorAdaptiveSuccessiveOptions | None = None,
    response_cache: ArbitraryQVectorAdaptiveSuccessiveResponseCache | None = None,
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
    local_cache = response_cache or ArbitraryQVectorAdaptiveSuccessiveResponseCache()
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
    plate_1 = integrate_arbitrary_q_vector_adaptive_successive(
        q_model=rotate_lab_q_to_crystal(q, float(theta_1_rad)), **common
    )
    plate_2 = tuple(
        integrate_arbitrary_q_vector_adaptive_successive(
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
    "ArbitraryQVectorAdaptiveSuccessiveOptions",
    "ArbitraryQVectorAdaptiveSuccessiveProfile",
    "ArbitraryQVectorAdaptiveSuccessiveResponseCache",
    "integrate_arbitrary_q_vector_adaptive_successive",
    "integrate_two_plate_angle_batch_vector_adaptive_successive",
    "successive_high_controller_converged",
    "successive_high_error_metrics",
    "update_successive_high_streak",
]
