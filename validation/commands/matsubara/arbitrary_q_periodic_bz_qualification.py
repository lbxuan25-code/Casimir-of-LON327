"""Diagnostic commensurate/arbitrary-q qualification core for periodic BZ.

The core never grants authorization.  The public gate verifies clean same-source
performance evidence and only then promotes a passed diagnostic result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.conventions import (
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import positive_matsubara_sheet_response_to_reflection
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.arbitrary_q_formal_policy import (
    EXECUTION_STRATEGY,
    MODEL_WORKLOAD_ID,
    QUALIFICATION_MATRIX_ID,
    THREAD_POLICY_ID,
    config_fingerprint,
    validate_numerical_formal_config,
)
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.arbitrary_q_matsubara import (
    ArbitraryQPeriodicBZResult,
    paired_average_arbitrary_q_results,
)
from lno327.workflows.arbitrary_q_parallel import ArbitraryQParallelEvaluator, QLabAngleTask
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_matsubara_orbit_gauss
from validation.lib.source_tree_provenance import source_tree_provenance

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_periodic_bz_qualification/qualification.json"
)


@dataclass(frozen=True)
class QualificationContextResult:
    pairing: str
    n: int
    shift: tuple[float, float]
    material_cache_metadata: Mapping[str, Any]
    execution_metadata: Mapping[str, Any]
    responses: Mapping[str, ArbitraryQPeriodicBZResult]
    two_plate_batch: object | None


def _formal_config(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "qualification_matrix_id": QUALIFICATION_MATRIX_ID,
        "model_workload_id": MODEL_WORKLOAD_ID,
        "pairings": list(a.pairings),
        "N_values": list(a.N_values),
        "reference_nk": int(a.reference_nk),
        "reference_order": int(a.reference_order),
        "reference_panel_count": int(a.reference_panel_count),
        "reference_workers": int(a.reference_workers),
        "reference_task_size": int(a.reference_task_size),
        "matsubara_indices": list(a.matsubara_indices),
        "primitive_tolerance": float(a.primitive_tolerance),
        "primitive_atol": float(a.primitive_atol),
        "reflection_tolerance": float(a.reflection_tolerance),
        "reflection_atol": float(a.reflection_atol),
        "logdet_tolerance": float(a.logdet_tolerance),
        "logdet_atol": float(a.logdet_atol),
        "diagonal_observable_tolerance": float(a.diagonal_observable_tolerance),
        "diagonal_observable_atol": float(a.diagonal_observable_atol),
        "ward_tolerance": float(a.ward_tolerance),
        "ward_absolute_tolerance": float(a.ward_absolute_tolerance),
        "temperature_K": float(a.temperature_K),
        "delta0_eV": float(a.delta0_eV),
        "eta_eV": float(a.eta_eV),
        "separation_nm": float(a.separation_nm),
        "canonical_block_size": int(a.canonical_block_size),
        "runtime_chunk_size": int(a.runtime_chunk_size),
        "primary_workers": int(a.workers),
        "audit_workers": 1,
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"])
    p.add_argument("--N-values", nargs="+", type=int, default=[256, 384, 512])
    p.add_argument("--reference-nk", type=int, default=1256)
    p.add_argument("--reference-order", type=int, default=384)
    p.add_argument("--reference-panel-count", type=int, default=16)
    p.add_argument("--reference-workers", type=int, default=8)
    p.add_argument("--reference-task-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1, 8])
    p.add_argument("--canonical-block-size", type=int, default=4096)
    p.add_argument("--runtime-chunk-size", type=int, default=16384)
    p.add_argument("--temperature-K", type=float, default=10.0)
    p.add_argument("--delta0-eV", type=float, default=0.1)
    p.add_argument("--eta-eV", type=float, default=1e-8)
    p.add_argument("--separation-nm", type=float, default=20.0)
    p.add_argument("--primitive-tolerance", type=float, default=1e-3)
    p.add_argument("--primitive-atol", type=float, default=1e-12)
    p.add_argument("--reflection-tolerance", type=float, default=3e-4)
    p.add_argument("--reflection-atol", type=float, default=1e-12)
    p.add_argument("--logdet-tolerance", type=float, default=3e-4)
    p.add_argument("--logdet-atol", type=float, default=1e-14)
    p.add_argument("--diagonal-observable-tolerance", type=float, default=1e-3)
    p.add_argument("--diagonal-observable-atol", type=float, default=1e-12)
    p.add_argument("--ward-tolerance", type=float, default=1e-7)
    p.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    p.add_argument("--diagnostic-nonformal", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args(argv)
    a.N_values = tuple(sorted(set(a.N_values)))
    a.matsubara_indices = tuple(sorted(set(a.matsubara_indices)))
    a.pairings = tuple(dict.fromkeys(a.pairings))
    if any(n <= 0 or n % 2 for n in a.N_values):
        p.error("positive even N values are required")
    if 0 not in a.matsubara_indices or not any(n > 0 for n in a.matsubara_indices):
        p.error("exact zero and at least one positive Matsubara index are required")
    if a.reference_panel_count <= 0 or a.reference_order % a.reference_panel_count:
        p.error("reference order must be divisible by positive panel count")
    if a.workers <= 0 or a.reference_workers <= 0:
        p.error("worker counts must be positive")
    for name in (
        "primitive_tolerance", "primitive_atol", "reflection_tolerance",
        "reflection_atol", "logdet_tolerance", "logdet_atol",
        "diagonal_observable_tolerance", "diagonal_observable_atol",
        "ward_tolerance", "ward_absolute_tolerance",
    ):
        value = float(getattr(a, name))
        if not np.isfinite(value) or value < 0.0:
            p.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    a.formal_policy = validate_numerical_formal_config(_formal_config(a))
    if not a.formal_policy.passed and not a.diagnostic_nonformal:
        p.error("configuration is looser than the frozen formal policy: " + "; ".join(a.formal_policy.violations))
    return a


def _xi(a: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        [0.0 if n == 0 else matsubara_energy_eV(n, a.temperature_K) for n in a.matsubara_indices],
        dtype=float,
    )


def _mixed(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    x, y = np.asarray(left, dtype=complex), np.asarray(right, dtype=complex)
    delta = float(np.linalg.norm(x - y))
    scale = max(float(np.linalg.norm(x)), float(np.linalg.norm(y)))
    threshold = float(atol) + float(rtol) * scale
    ratio = delta / max(threshold, np.finfo(float).tiny)
    return {
        "absolute": delta,
        "relative": delta / max(scale, np.finfo(float).tiny),
        "scale": scale,
        "atol": float(atol),
        "rtol": float(rtol),
        "mixed_threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _signature(component: object, rhs: object) -> np.ndarray:
    return np.concatenate([
        np.asarray(value, dtype=complex).reshape(-1)
        for value in (
            component.bare_bubble, component.direct, component.collective_bubble,
            component.collective_counterterm, component.em_collective_left,
            component.collective_em_right, component.gauge_restored, rhs.left, rhs.right,
        )
    ])


def _physical(result: object, q: np.ndarray, a: argparse.Namespace) -> list[dict[str, Any]]:
    config = OrbitAcceptancePhysicsConfig(
        separation_nm=a.separation_nm,
        ward_tolerance=a.ward_tolerance,
        ward_absolute_tolerance=a.ward_absolute_tolerance,
    )
    rows = []
    for n, frequency, component, rhs in zip(
        a.matsubara_indices, result.xi_eV_values, result.components, result.rhs, strict=True
    ):
        state = evaluate_matsubara_pipeline(
            components=component, rhs=rhs, q_model=q, xi_eV=float(frequency), config=config
        )
        rows.append({
            "n": int(n),
            "passed": bool(state["physical_passed"]),
            "ward": bool(state["ward_passed"]),
            "strict_static": bool(state["strict_static_ward_passed"]),
            "sheet": bool(state["sheet_validation_passed"]),
            "reflection_constructed": bool(state["reflection_constructed"]),
            "logdet_passed": bool(state["logdet_passed"]),
            "reflection": np.asarray(state["reflection"], dtype=complex),
            "logdet": float(state["logdet"]),
            "primary": np.asarray(state["primary_response"], dtype=complex),
            "error": str(state["error"]),
        })
    return rows


def _all_physical(result: object, states: Sequence[Mapping[str, Any]]) -> bool:
    operator = getattr(result, "operator_ward", None)
    return bool((operator is None or operator.passed) and all(bool(row["passed"]) for row in states))


def _q_cases(a: argparse.Namespace) -> dict[str, np.ndarray]:
    factor = 2.0 * np.pi / float(a.reference_nk)
    base = factor * np.asarray([6.0, 4.0])
    angle = np.deg2rad(17.0)
    rotation_minus = np.asarray([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return {
        "axis": factor * np.asarray([1.0, 0.0]),
        "generic": base,
        "near_diagonal": factor * np.asarray([25.0, 24.0]),
        "exact_diagonal": factor * np.asarray([6.0, 6.0]),
        "rotated_17deg": rotation_minus @ base,
    }


def _threadpool_runtime() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info  # type: ignore
    except ImportError:
        return [{"error": "threadpoolctl not installed", "num_threads": -1}]
    return [dict(item) for item in threadpool_info()]


def _evaluate_context(
    a: argparse.Namespace,
    *,
    pairing_name: str,
    n: int,
    shift: tuple[float, float],
    audit_only: bool = False,
) -> QualificationContextResult:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    xi = _xi(a)
    grid = build_periodic_bz_grid(int(n), shift)
    cache = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=KuboConfig.from_kelvin(
            omega_eV=float(xi[0]), temperature_K=a.temperature_K, eta_eV=a.eta_eV, output_si=False
        ),
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
        grid=grid,
    )
    q = _q_cases(a)
    if audit_only:
        tasks = (QLabAngleTask(0, q["generic"], 0.0, np.asarray([np.deg2rad(17.0)])),)
        workers = 1
    else:
        tasks = (
            QLabAngleTask(0, q["axis"], 0.0, np.asarray([0.0])),
            QLabAngleTask(1, q["generic"], 0.0, np.asarray([0.0, np.deg2rad(17.0)])),
            QLabAngleTask(2, q["near_diagonal"], 0.0, np.asarray([0.0])),
            QLabAngleTask(3, q["exact_diagonal"], 0.0, np.asarray([0.0])),
        )
        workers = min(int(a.workers), len(tasks))
    evaluator = ArbitraryQParallelEvaluator(
        material_cache=cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=a.temperature_K,
        eta_eV=a.eta_eV,
        process_workers=workers,
        canonical_reduction_block_size=a.canonical_block_size,
        runtime_chunk_size=a.runtime_chunk_size,
    )
    try:
        evaluated = evaluator.evaluate(tasks)
    finally:
        evaluator.close()
    execution_metadata = evaluator.metadata()
    if audit_only:
        responses = {
            "generic": evaluated[0].result.plate_1,
            "rotated_17deg": evaluated[0].result.plate_2[0],
        }
        two_plate = evaluated[0].result
    else:
        responses = {
            "axis": evaluated[0].result.plate_1,
            "generic": evaluated[1].result.plate_1,
            "rotated_17deg": evaluated[1].result.plate_2[1],
            "near_diagonal": evaluated[2].result.plate_1,
            "exact_diagonal": evaluated[3].result.plate_1,
        }
        two_plate = evaluated[1].result
    return QualificationContextResult(
        pairing=pairing_name,
        n=int(n),
        shift=tuple(float(value) for value in shift),
        material_cache_metadata=cache.metadata(),
        execution_metadata=execution_metadata,
        responses=responses,
        two_plate_batch=two_plate,
    )


def _reference(a: argparse.Namespace, *, pairing_name: str, indices: tuple[int, int]):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    mx, my = indices
    origins = 2 if int(np.gcd(abs(mx), abs(my))) % 2 else 1
    return integrate_matsubara_orbit_gauss(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=_xi(a),
        temperature_K=a.temperature_K,
        eta_eV=a.eta_eV,
        nk=a.reference_nk,
        mx=mx,
        my=my,
        transverse_order=a.reference_order,
        panel_count=a.reference_panel_count,
        shift_s=0.5,
        subgrid_average="auto",
        max_point_evaluations=a.reference_nk * origins * a.reference_order,
        transverse_workers=a.reference_workers,
        transverse_task_size=a.reference_task_size,
    )


def _commensurate_rows(a: argparse.Namespace, *, pairing_name: str, contexts: Sequence[QualificationContextResult]) -> list[dict[str, Any]]:
    cases = {"axis": (1, 0), "generic": (6, 4), "near_diagonal": (25, 24), "exact_diagonal": (6, 6)}
    rows = []
    for name, indices in cases.items():
        q = _q_cases(a)[name]
        reference = _reference(a, pairing_name=pairing_name, indices=indices)
        reference_states = _physical(reference, q, a)
        series = [context.responses[name] for context in contexts]
        states = [_physical(result, q, a) for result in series]
        diagonal = pairing_name == "dwave" and name == "exact_diagonal"
        obs_rtol = a.diagonal_observable_tolerance if diagonal else a.reflection_tolerance
        obs_atol = a.diagonal_observable_atol if diagonal else a.reflection_atol
        log_rtol = a.diagonal_observable_tolerance if diagonal else a.logdet_tolerance
        log_atol = a.diagonal_observable_atol if diagonal else a.logdet_atol
        all_physical = bool(
            all(bool(row["passed"]) for row in reference_states)
            and all(_all_physical(result, state) for result, state in zip(series, states, strict=True))
        )
        frequency_rows = []
        for index, n_value in enumerate(a.matsubara_indices):
            primitive = _mixed(
                _signature(reference.components[index], reference.rhs[index]),
                _signature(series[-1].components[index], series[-1].rhs[index]),
                atol=a.primitive_atol, rtol=a.primitive_tolerance,
            )
            reflection = _mixed(reference_states[index]["reflection"], states[-1][index]["reflection"], atol=obs_atol, rtol=obs_rtol)
            logdet = _mixed(reference_states[index]["logdet"], states[-1][index]["logdet"], atol=log_atol, rtol=log_rtol)
            refinements = [
                _mixed(states[j][index]["reflection"], states[j + 1][index]["reflection"], atol=obs_atol, rtol=obs_rtol)
                for j in range(len(states) - 1)
            ]
            log_refinements = [
                _mixed(states[j][index]["logdet"], states[j + 1][index]["logdet"], atol=log_atol, rtol=log_rtol)
                for j in range(len(states) - 1)
            ]
            passed = bool(
                all_physical and reflection["passed"] and logdet["passed"]
                and all(item["passed"] for item in (*refinements, *log_refinements))
                and (primitive["passed"] or diagonal)
            )
            frequency_rows.append({
                "n": int(n_value), "primitive": primitive, "reflection": reflection,
                "logdet": logdet, "refinement": refinements,
                "logdet_refinement": log_refinements,
                "response_unresolved_allowed": diagonal,
                "all_reference_and_refinement_physical": all_physical,
                "passed": passed,
            })
        rows.append({
            "pairing": pairing_name, "case": name, "q_indices": list(indices),
            "q_model": q.tolist(), "N_values": [context.n for context in contexts],
            "frequencies": frequency_rows,
            "passed": all(row["passed"] for row in frequency_rows),
        })
    return rows


def _arbitrary_row(
    a: argparse.Namespace,
    *,
    pairing_name: str,
    contexts: Sequence[QualificationContextResult],
    audit_a: QualificationContextResult,
    audit_b: QualificationContextResult,
) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    q = _q_cases(a)["rotated_17deg"]
    series = [context.responses["rotated_17deg"] for context in contexts]
    states = [_physical(result, q, a) for result in series]
    first, second = audit_a.responses["rotated_17deg"], audit_b.responses["rotated_17deg"]
    paired = paired_average_arbitrary_q_results(
        first, second, ansatz=ansatz, pairing=pairing,
        temperature_K=a.temperature_K, eta_eV=a.eta_eV,
    )
    audit_results = (first, second, paired)
    audit_states = [_physical(result, q, a) for result in audit_results]
    all_physical = bool(
        all(_all_physical(result, state) for result, state in zip(series, states, strict=True))
        and all(_all_physical(result, state) for result, state in zip(audit_results, audit_states, strict=True))
    )
    rows = []
    for index, n_value in enumerate(a.matsubara_indices):
        comparisons = [
            *[_mixed(states[j][index]["primary"], states[j + 1][index]["primary"], atol=a.primitive_atol, rtol=a.primitive_tolerance) for j in range(len(states) - 1)],
            *[_mixed(states[j][index]["reflection"], states[j + 1][index]["reflection"], atol=a.reflection_atol, rtol=a.reflection_tolerance) for j in range(len(states) - 1)],
            *[_mixed(states[j][index]["logdet"], states[j + 1][index]["logdet"], atol=a.logdet_atol, rtol=a.logdet_tolerance) for j in range(len(states) - 1)],
        ]
        spread = {
            "packed_primitive_a_vs_b": _mixed(first.packed_primitives, second.packed_primitives, atol=a.primitive_atol, rtol=a.primitive_tolerance),
            "reflection_a_vs_b": _mixed(audit_states[0][index]["reflection"], audit_states[1][index]["reflection"], atol=a.reflection_atol, rtol=a.reflection_tolerance),
            "logdet_a_vs_b": _mixed(audit_states[0][index]["logdet"], audit_states[1][index]["logdet"], atol=a.logdet_atol, rtol=a.logdet_tolerance),
        }
        primary_vs_paired = {
            "packed_primitive": _mixed(series[-1].packed_primitives, paired.packed_primitives, atol=a.primitive_atol, rtol=a.primitive_tolerance),
            "response": _mixed(states[-1][index]["primary"], audit_states[2][index]["primary"], atol=a.primitive_atol, rtol=a.primitive_tolerance),
            "reflection": _mixed(states[-1][index]["reflection"], audit_states[2][index]["reflection"], atol=a.reflection_atol, rtol=a.reflection_tolerance),
            "logdet": _mixed(states[-1][index]["logdet"], audit_states[2][index]["logdet"], atol=a.logdet_atol, rtol=a.logdet_tolerance),
        }
        passed = bool(all_physical and all(item["passed"] for item in (*comparisons, *spread.values(), *primary_vs_paired.values())))
        rows.append({
            "n": int(n_value), "refinement_comparisons": comparisons,
            "independent_shift_spread": spread,
            "primary_vs_paired_primitive_result": primary_vs_paired,
            "all_primary_audit_and_paired_physical": all_physical,
            "passed": passed,
        })
    return {
        "pairing": pairing_name, "case": "rotated_17deg", "q_model": q.tolist(),
        "N_values": [context.n for context in contexts],
        "audit_shifts": [list(audit_a.shift), list(audit_b.shift)],
        "paired_shift_definition": "0.5*(packed_A+packed_B) before all nonlinear processing",
        "frequencies": rows,
        "passed": all(row["passed"] for row in rows),
    }


def _plate_reflection(component: object, rhs: object, q_crystal: np.ndarray, q_lab: np.ndarray, theta: float, frequency: float, config: OrbitAcceptancePhysicsConfig):
    kernel = effective_em_kernel_from_components(component, q_model=q_crystal, xi_eV=frequency)
    ward = validate_effective_ward_xy(
        kernel, rhs, residual_tolerance=config.ward_tolerance,
        absolute_residual_tolerance=config.ward_absolute_tolerance,
        condition_max=config.condition_max,
    )
    if frequency == 0.0:
        strict = validate_strict_static_ward_closure(
            kernel, ward, energy_scale_eV=config.static_energy_scale_eV,
            primitive_tolerance=config.static_primitive_tolerance,
            amplitude_tolerance=config.static_amplitude_tolerance,
            phase_tolerance=config.static_phase_tolerance,
            effective_direct_tolerance=config.static_effective_direct_tolerance,
            effective_residual_tolerance=config.static_effective_residual_tolerance,
            longitudinal_tolerance=config.static_longitudinal_tolerance,
            condition_max=config.condition_max,
        )
        sheet = static_matsubara_kernel_to_sheet_response(
            kernel, ward, energy_scale_eV=config.static_energy_scale_eV,
            degeneracy=config.degeneracy, reality_tolerance=config.static_reality_tolerance,
            longitudinal_tolerance=config.static_longitudinal_tolerance,
            mixing_tolerance=config.static_mixing_tolerance,
            passivity_tolerance=config.static_passivity_tolerance,
        )
        reflection = static_sheet_response_to_reflection(
            sheet, q_lab_model=q_lab, theta_rad=theta,
            lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
            require_physical=True,
        )
        return reflection, bool(ward.passed and strict.passed and sheet.validation.passed)
    sheet = positive_matsubara_kernel_to_sheet_response(kernel, degeneracy=config.degeneracy)
    validation = validate_positive_matsubara_sheet_response(sheet)
    reflection = positive_matsubara_sheet_response_to_reflection(
        sheet, q_lab_model=q_lab, theta_rad=theta,
        lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        require_physical=True,
    )
    return reflection, bool(ward.passed and validation.passed)


def _rotated_plate(batch: object) -> ArbitraryQPeriodicBZResult:
    angles = np.asarray(batch.theta_2_rad_values, dtype=float)
    target = np.deg2rad(17.0)
    matches = np.flatnonzero(np.isclose(angles, target, rtol=0.0, atol=1e-14))
    if matches.size != 1:
        raise RuntimeError("two-plate batch does not contain exactly one 17-degree plate")
    return batch.plate_2[int(matches[0])]


def _two_plate_states(
    a: argparse.Namespace,
    plate_1: ArbitraryQPeriodicBZResult,
    plate_2: ArbitraryQPeriodicBZResult,
) -> list[dict[str, Any]]:
    q_lab = _q_cases(a)["generic"]
    theta = np.deg2rad(17.0)
    config = OrbitAcceptancePhysicsConfig(
        separation_nm=a.separation_nm,
        ward_tolerance=a.ward_tolerance,
        ward_absolute_tolerance=a.ward_absolute_tolerance,
    )
    both_operator = bool(plate_1.operator_ward.passed and plate_2.operator_ward.passed)
    rows = []
    for index, frequency in enumerate(_xi(a)):
        try:
            r1, p1 = _plate_reflection(
                plate_1.components[index], plate_1.rhs[index], plate_1.q_model,
                q_lab, 0.0, float(frequency), config,
            )
            r2, p2 = _plate_reflection(
                plate_2.components[index], plate_2.rhs[index], plate_2.q_model,
                q_lab, theta, float(frequency), config,
            )
            point = passive_sheet_logdet(r1, r2, separation_m=a.separation_nm * 1e-9)
            logdet = float(point.logdet)
            passed = bool(both_operator and p1 and p2 and np.isfinite(logdet))
            error = ""
        except (ValueError, np.linalg.LinAlgError) as exc:
            logdet, passed, error = float("nan"), False, str(exc)
        rows.append({
            "n": int(a.matsubara_indices[index]), "logdet": logdet,
            "both_operator_identities_passed": both_operator,
            "passed": passed, "error": error,
        })
    return rows


def _two_plate_row(
    a: argparse.Namespace,
    *,
    pairing_name: str,
    contexts: Sequence[QualificationContextResult],
    audit_a: QualificationContextResult,
    audit_b: QualificationContextResult,
) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    primary_pairs = []
    for context in contexts:
        if context.two_plate_batch is None:
            raise RuntimeError("primary context did not retain two-plate batch")
        primary_pairs.append((context.two_plate_batch.plate_1, _rotated_plate(context.two_plate_batch)))
    if audit_a.two_plate_batch is None or audit_b.two_plate_batch is None:
        raise RuntimeError("audit context did not retain two-plate batch")
    a_pair = (audit_a.two_plate_batch.plate_1, _rotated_plate(audit_a.two_plate_batch))
    b_pair = (audit_b.two_plate_batch.plate_1, _rotated_plate(audit_b.two_plate_batch))
    paired_1 = paired_average_arbitrary_q_results(
        a_pair[0], b_pair[0], ansatz=ansatz, pairing=pairing,
        temperature_K=a.temperature_K, eta_eV=a.eta_eV,
    )
    paired_2 = paired_average_arbitrary_q_results(
        a_pair[1], b_pair[1], ansatz=ansatz, pairing=pairing,
        temperature_K=a.temperature_K, eta_eV=a.eta_eV,
    )
    primary_states = [_two_plate_states(a, p1, p2) for p1, p2 in primary_pairs]
    audit_states = (
        _two_plate_states(a, *a_pair),
        _two_plate_states(a, *b_pair),
        _two_plate_states(a, paired_1, paired_2),
    )
    rows = []
    for index, n_value in enumerate(a.matsubara_indices):
        refinement = [
            _mixed(primary_states[j][index]["logdet"], primary_states[j + 1][index]["logdet"], atol=a.logdet_atol, rtol=a.logdet_tolerance)
            for j in range(len(primary_states) - 1)
        ]
        shift_spread = _mixed(
            audit_states[0][index]["logdet"], audit_states[1][index]["logdet"],
            atol=a.logdet_atol, rtol=a.logdet_tolerance,
        )
        primary_vs_paired = _mixed(
            primary_states[-1][index]["logdet"], audit_states[2][index]["logdet"],
            atol=a.logdet_atol, rtol=a.logdet_tolerance,
        )
        all_physical = bool(
            all(state[index]["passed"] for state in primary_states)
            and all(state[index]["passed"] for state in audit_states)
        )
        passed = bool(
            all_physical and all(item["passed"] for item in refinement)
            and shift_spread["passed"] and primary_vs_paired["passed"]
        )
        rows.append({
            "n": int(n_value),
            "primary_logdet_by_N": [state[index]["logdet"] for state in primary_states],
            "N_refinement": refinement,
            "audit_a_logdet": audit_states[0][index]["logdet"],
            "audit_b_logdet": audit_states[1][index]["logdet"],
            "paired_primitive_two_plate_logdet": audit_states[2][index]["logdet"],
            "audit_a_vs_b": shift_spread,
            "primary_final_vs_paired": primary_vs_paired,
            "all_primary_audit_and_paired_two_plate_physical": all_physical,
            "passed": passed,
        })
    return {
        "pairing": pairing_name,
        "q_lab": _q_cases(a)["generic"].tolist(),
        "theta_deg": [0.0, 17.0],
        "N_values": [context.n for context in contexts],
        "audit_shifts": [list(audit_a.shift), list(audit_b.shift)],
        "paired_definition": "pair each plate at packed-primitive level, then construct one nonlinear two-plate logdet",
        "frequencies": rows,
        "passed": all(row["passed"] for row in rows),
    }


def _q_coverage(a: argparse.Namespace, passed: bool) -> dict[str, Any]:
    values = list(_q_cases(a).values())
    return {
        "coverage_kind": "discrete_qualification_matrix_not_continuous_outer_envelope",
        "qualification_matrix_id": QUALIFICATION_MATRIX_ID,
        "tested_q_vectors": [value.tolist() for value in values],
        "tested_q_max_norm": max(float(np.linalg.norm(value)) for value in values),
        "tested_q_max_component": max(float(np.max(np.abs(value))) for value in values),
        "tested_angle_classes": ["axis", "generic", "near_diagonal", "exact_diagonal", "rotated_17deg"],
        "discrete_matrix_passed": bool(passed),
        "continuous_angle_coverage_established": False,
        "outer_tail_requirement_bound": False,
        "qualified_outer_q_envelope_established": False,
        "principal_supported_domain_is_not_claimed_as_qualified": True,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    a = _args(argv)
    provenance = source_tree_provenance()
    if not provenance.worktree_clean and not a.diagnostic_nonformal:
        provenance.require_clean()
    threadpools = _threadpool_runtime()
    actual_threadpool_passed = bool(threadpools) and all(int(row.get("num_threads", -1)) == 1 for row in threadpools)
    if not actual_threadpool_passed and not a.diagnostic_nonformal:
        raise RuntimeError("formal qualification requires actual single-thread BLAS threadpools")

    commensurate: list[dict[str, Any]] = []
    arbitrary: list[dict[str, Any]] = []
    two_plate: list[dict[str, Any]] = []
    context_records: list[dict[str, Any]] = []
    for pairing_name in a.pairings:
        primary_contexts = [
            _evaluate_context(a, pairing_name=pairing_name, n=n, shift=(0.5, 0.5))
            for n in a.N_values
        ]
        audit_a = _evaluate_context(a, pairing_name=pairing_name, n=a.N_values[-1], shift=(0.25, 0.75), audit_only=True)
        audit_b = _evaluate_context(a, pairing_name=pairing_name, n=a.N_values[-1], shift=(0.75, 0.25), audit_only=True)
        for context in (*primary_contexts, audit_a, audit_b):
            context_records.append({
                "pairing": context.pairing, "N": context.n, "shift": list(context.shift),
                "material_cache": dict(context.material_cache_metadata),
                "execution": dict(context.execution_metadata),
            })
        commensurate.extend(_commensurate_rows(a, pairing_name=pairing_name, contexts=primary_contexts))
        arbitrary.append(_arbitrary_row(a, pairing_name=pairing_name, contexts=primary_contexts, audit_a=audit_a, audit_b=audit_b))
        two_plate.append(_two_plate_row(a, pairing_name=pairing_name, contexts=primary_contexts, audit_a=audit_a, audit_b=audit_b))

    metric_passed = all(row["passed"] for row in (*commensurate, *arbitrary, *two_plate))
    formal_eligible = bool(
        metric_passed and a.formal_policy.passed and provenance.worktree_clean
        and actual_threadpool_passed and not a.diagnostic_nonformal
    )
    formal_config = _formal_config(a)
    payload = {
        "schema": "arbitrary-q-periodic-bz-diagnostic-result-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **provenance.as_dict(),
        "actual_threadpools": threadpools,
        "actual_threadpool_passed": actual_threadpool_passed,
        "config": formal_config,
        "config_fingerprint": config_fingerprint(formal_config),
        **a.formal_policy.as_dict(),
        "formal_evidence_eligible": formal_eligible,
        "diagnostic_nonformal_requested": bool(a.diagnostic_nonformal),
        "qualification_execution_contexts": context_records,
        "commensurate_regression": commensurate,
        "arbitrary_q_refinement_and_shift_audit": arbitrary,
        "two_plate_common_lab_basis_convergence": two_plate,
        "numerical_q_coverage": _q_coverage(a, metric_passed),
        "arbitrary_q_microscopic_contract": "diagnostic_result_passed" if metric_passed else "diagnostic_result_failed",
        "authorization_source": "none_direct_core_execution_never_authorizes_outer_integration",
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": bool(metric_passed),
    }
    _write(a.output, payload)
    print(json.dumps({"output": str(a.output), "passed": metric_passed}, indent=2))
    if not metric_passed:
        raise SystemExit("arbitrary-q periodic BZ diagnostic qualification failed")


if __name__ == "__main__":
    main()
