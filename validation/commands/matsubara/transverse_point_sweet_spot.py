"""Unified single-point transverse-integration sweet-spot search.

For each requested external lab-frame momentum, pairing, and Matsubara index, this
command evaluates complete shifted even-N periodic Brillouin-zone grids. It stops
an individual point as soon as the two-plate Casimir logdet is stable across
adjacent N levels and independent full-period shifts while every hard physical
gate remains closed.

CPU execution uses exactly one process-parallel layer.  Automatic mode chooses
between q/angle parallelism with one fork-shared readonly material cache and
independent pairing/shift material-context parallelism capped by a memory budget.
Nested process pools and multithreaded BLAS are forbidden.

This is the only public single-point transverse-convergence command. It does not
perform an outer q integral or Matsubara sum and cannot authorize production input.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from multiprocessing import get_all_start_methods, get_context
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.conventions import (
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import (
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.arbitrary_q_parallel import (
    ArbitraryQParallelEvaluator,
    QLabAngleTask,
    actual_threadpool_record,
    validate_single_thread_blas_environment,
)
from lno327.workflows.cpu_parallel import (
    CPUParallelPlan,
    choose_cpu_parallel_plan,
    estimate_context_bytes,
    numpy_array_bytes,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.matsubara import matsubara_energy_eV

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/transverse_point_sweet_spot/diagnostic.json"
)
DEFAULT_SHIFTS = ((0.5, 0.5), (0.25, 0.75), (0.75, 0.25))


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _mixed_scalar(
    previous: float,
    current: float,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = float(previous)
    right = float(current)
    if not np.isfinite(left) or not np.isfinite(right):
        return {
            "finite": False,
            "absolute": float("nan"),
            "relative": float("nan"),
            "mixed_threshold": float("nan"),
            "mixed_ratio": float("nan"),
            "passed": False,
        }
    absolute = abs(right - left)
    scale = max(abs(left), abs(right))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "finite": True,
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
        "mixed_threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _scalar_spread(
    values: Sequence[float],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        return {
            "finite": False,
            "minimum": float("nan"),
            "maximum": float("nan"),
            "absolute": float("nan"),
            "relative": float("nan"),
            "mixed_threshold": float("nan"),
            "mixed_ratio": float("nan"),
            "passed": False,
        }
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    absolute = maximum - minimum
    scale = float(np.max(np.abs(array)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "finite": True,
        "minimum": minimum,
        "maximum": maximum,
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
        "mixed_threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _parse_q_points(raw: Sequence[Sequence[str]]) -> tuple[dict[str, Any], ...]:
    points: list[dict[str, Any]] = []
    labels: set[str] = set()
    for fields in raw:
        if len(fields) != 3:
            raise ValueError("each --q-point requires LABEL QX QY")
        label = str(fields[0])
        if not label or label in labels:
            raise ValueError("q-point labels must be nonempty and unique")
        q = np.asarray([float(fields[1]), float(fields[2])], dtype=float)
        if not np.isfinite(q).all() or float(np.linalg.norm(q)) == 0.0:
            raise ValueError(f"q point {label!r} must be finite and nonzero")
        labels.add(label)
        points.append({"label": label, "q_lab": q})
    if not points:
        raise ValueError("at least one --q-point is required")
    return tuple(points)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-point",
        action="append",
        nargs=3,
        metavar=("LABEL", "QX", "QY"),
        required=True,
        help="repeatable lab-frame external momentum",
    )
    parser.add_argument(
        "--pairings",
        nargs="+",
        choices=("spm", "dwave"),
        default=["spm", "dwave"],
    )
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--N-candidates",
        nargs="+",
        type=int,
        default=[128, 192, 256, 384, 512, 640, 768],
    )
    parser.add_argument(
        "--shift",
        action="append",
        nargs=2,
        type=float,
        metavar=("SX", "SY"),
        help="repeatable complete-period grid shift; defaults to primary/A/B",
    )
    parser.add_argument("--plate-angles-deg", nargs=2, type=float, default=[0.0, 17.0])
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="total process budget; 0 uses the current CPU affinity",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "q", "context"),
        default="auto",
        help="select one non-nested process-parallel axis",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=0.0,
        help="context-parallel memory budget; 0 uses 70%% of available memory",
    )
    parser.add_argument(
        "--max-context-workers",
        type=int,
        default=0,
        help="hard cap for simultaneous pairing/shift material contexts; 0 is automatic",
    )
    parser.add_argument(
        "--memory-safety-factor",
        type=float,
        default=1.5,
        help="multiplier applied to measured or fallback context memory",
    )
    parser.add_argument(
        "--fallback-context-bytes-per-point",
        type=float,
        default=16_384.0,
        help="conservative pre-telemetry material-context memory estimate",
    )
    parser.add_argument("--canonical-block", type=int, default=4096)
    parser.add_argument("--runtime-chunk", type=int, default=16384)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--static-energy-scale-eV", type=float, default=1.0)
    parser.add_argument("--static-reality-tolerance", type=float, default=1e-8)
    parser.add_argument("--static-longitudinal-tolerance", type=float, default=1e-6)
    parser.add_argument("--static-mixing-tolerance", type=float, default=1e-6)
    parser.add_argument("--static-passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--logdet-rtol", type=float, default=1e-3)
    parser.add_argument("--logdet-atol", type=float, default=1e-14)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        args.q_points = _parse_q_points(args.q_point)
    except ValueError as exc:
        parser.error(str(exc))

    args.pairings = tuple(dict.fromkeys(str(value) for value in args.pairings))
    args.matsubara_indices = tuple(
        sorted(set(int(value) for value in args.matsubara_indices))
    )
    args.N_candidates = tuple(int(value) for value in args.N_candidates)
    args.shifts = tuple(
        tuple(float(component) for component in value)
        for value in (args.shift or DEFAULT_SHIFTS)
    )
    args.plate_angles_rad = tuple(
        float(np.deg2rad(value)) for value in args.plate_angles_deg
    )

    if not args.matsubara_indices or any(value < 0 for value in args.matsubara_indices):
        parser.error("--matsubara-indices must be nonempty and non-negative")
    if len(args.N_candidates) < 3:
        parser.error("--N-candidates must contain at least three levels")
    if (
        tuple(sorted(set(args.N_candidates))) != args.N_candidates
        or any(value <= 0 or value % 2 for value in args.N_candidates)
    ):
        parser.error("--N-candidates must be strictly increasing unique positive even integers")
    if len(args.shifts) < 2 or len(set(args.shifts)) != len(args.shifts):
        parser.error("at least two unique --shift values are required")
    if not all(np.isfinite(value) for shift in args.shifts for value in shift):
        parser.error("all shifts must be finite")
    if args.required_consecutive_passes <= 0:
        parser.error("--required-consecutive-passes must be positive")
    if args.required_consecutive_passes >= len(args.N_candidates):
        parser.error("consecutive-pass requirement leaves no usable N ladder")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.max_context_workers < 0:
        parser.error("--max-context-workers must be non-negative")
    if not np.isfinite(args.memory_budget_gb) or args.memory_budget_gb < 0.0:
        parser.error("--memory-budget-gb must be finite and non-negative")
    if not np.isfinite(args.memory_safety_factor) or args.memory_safety_factor < 1.0:
        parser.error("--memory-safety-factor must be finite and at least one")
    if (
        not np.isfinite(args.fallback_context_bytes_per_point)
        or args.fallback_context_bytes_per_point <= 0.0
    ):
        parser.error("--fallback-context-bytes-per-point must be finite and positive")
    if args.canonical_block <= 0 or args.runtime_chunk <= 0:
        parser.error("block and chunk sizes must be positive")
    if not np.isfinite(args.temperature_K) or args.temperature_K <= 0.0:
        parser.error("--temperature-K must be finite and positive")
    if not np.isfinite(args.delta0_eV) or args.delta0_eV <= 0.0:
        parser.error("--delta0-eV must be finite and positive")
    if not np.isfinite(args.eta_eV) or args.eta_eV < 0.0:
        parser.error("--eta-eV must be finite and non-negative")
    if not np.isfinite(args.degeneracy) or args.degeneracy <= 0.0:
        parser.error("--degeneracy must be finite and positive")
    if not np.isfinite(args.separation_nm) or args.separation_nm <= 0.0:
        parser.error("--separation-nm must be finite and positive")
    if not all(np.isfinite(value) for value in args.plate_angles_rad):
        parser.error("--plate-angles-deg values must be finite")
    for name in (
        "ward_tolerance",
        "ward_absolute_tolerance",
        "condition_max",
        "static_reality_tolerance",
        "static_longitudinal_tolerance",
        "static_mixing_tolerance",
        "static_passivity_tolerance",
        "logdet_rtol",
        "logdet_atol",
    ):
        value = _finite_nonnegative(getattr(args, name), name)
        if name == "condition_max" and value == 0.0:
            parser.error("--condition-max must be positive")
    return args


def _plate_state(
    result: object,
    *,
    frequency_index: int,
    q_lab: np.ndarray,
    theta_rad: float,
    xi_eV: float,
    args: argparse.Namespace | SimpleNamespace,
) -> tuple[object, dict[str, Any]]:
    component = result.components[frequency_index]
    rhs = result.rhs[frequency_index]
    q_crystal = np.asarray(result.q_model, dtype=float)
    kernel = effective_em_kernel_from_components(
        component,
        q_model=q_crystal,
        xi_eV=float(xi_eV),
    )
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=float(args.ward_tolerance),
        absolute_residual_tolerance=float(args.ward_absolute_tolerance),
        condition_max=float(args.condition_max),
    )

    if float(xi_eV) == 0.0:
        strict = validate_strict_static_ward_closure(
            kernel,
            ward,
            energy_scale_eV=float(args.static_energy_scale_eV),
            primitive_tolerance=1e-6,
            amplitude_tolerance=1e-6,
            phase_tolerance=1e-6,
            effective_direct_tolerance=1e-6,
            effective_residual_tolerance=1e-6,
            longitudinal_tolerance=float(args.static_longitudinal_tolerance),
            condition_max=float(args.condition_max),
        )
        sheet = static_matsubara_kernel_to_sheet_response(
            kernel,
            ward,
            energy_scale_eV=float(args.static_energy_scale_eV),
            degeneracy=float(args.degeneracy),
            reality_tolerance=float(args.static_reality_tolerance),
            longitudinal_tolerance=float(args.static_longitudinal_tolerance),
            mixing_tolerance=float(args.static_mixing_tolerance),
            passivity_tolerance=float(args.static_passivity_tolerance),
        )
        reflection = static_sheet_response_to_reflection(
            sheet,
            q_lab_model=q_lab,
            theta_rad=float(theta_rad),
            lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
            require_physical=True,
        )
        primary = np.diag([float(sheet.chi_bar), float(sheet.dbar_t)]).astype(complex)
        validation = sheet.validation
        state = {
            "operator_ward_passed": bool(result.operator_ward.passed),
            "ward_passed": bool(ward.passed),
            "ward_effective_mixed_ratio_max": max(
                ward.left.effective_mixed_ratio,
                ward.right.effective_mixed_ratio,
            ),
            "schur_condition_number": float(ward.schur_condition_number),
            "strict_static_ward_passed": bool(strict.passed),
            "strict_static_hard_gate": False,
            "sheet_validation_passed": bool(validation.passed),
            "reflection_constructed": True,
            "primary_norm": float(np.linalg.norm(primary)),
            "reflection_norm": float(np.linalg.norm(reflection.matrix_lt)),
            "chi_bar": float(sheet.chi_bar),
            "dbar_t": float(sheet.dbar_t),
            "static_longitudinal_residual": float(
                validation.relative_longitudinal_gauge_residual
            ),
            "static_longitudinal_tolerance": float(validation.longitudinal_tolerance),
            "static_longitudinal_warning": bool(validation.longitudinal_warning),
            "relative_imaginary_norm": float(validation.relative_imaginary_norm),
            "relative_density_transverse_mixing": float(
                validation.relative_density_transverse_mixing
            ),
        }
    else:
        sheet = positive_matsubara_kernel_to_sheet_response(
            kernel,
            degeneracy=float(args.degeneracy),
        )
        validation = validate_positive_matsubara_sheet_response(sheet)
        reflection = positive_matsubara_sheet_response_to_reflection(
            sheet,
            q_lab_model=q_lab,
            theta_rad=float(theta_rad),
            lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
            require_physical=True,
        )
        primary = np.asarray(sheet.matrix_tilde, dtype=complex)
        state = {
            "operator_ward_passed": bool(result.operator_ward.passed),
            "ward_passed": bool(ward.passed),
            "ward_effective_mixed_ratio_max": max(
                ward.left.effective_mixed_ratio,
                ward.right.effective_mixed_ratio,
            ),
            "schur_condition_number": float(ward.schur_condition_number),
            "strict_static_ward_passed": False,
            "strict_static_hard_gate": False,
            "sheet_validation_passed": bool(validation.passed),
            "reflection_constructed": True,
            "primary_norm": float(np.linalg.norm(primary)),
            "reflection_norm": float(np.linalg.norm(reflection.matrix_lt)),
            "chi_bar": float("nan"),
            "dbar_t": float("nan"),
            "static_longitudinal_residual": float("nan"),
            "static_longitudinal_tolerance": float("nan"),
            "static_longitudinal_warning": False,
            "relative_imaginary_norm": float(validation.relative_imaginary_norm),
            "relative_density_transverse_mixing": float("nan"),
        }

    state["hard_physical_passed"] = bool(
        state["operator_ward_passed"]
        and state["ward_passed"]
        and state["sheet_validation_passed"]
        and state["reflection_constructed"]
    )
    state["q_crystal"] = q_crystal.tolist()
    return reflection, state


def _two_plate_state(
    batch: object,
    *,
    frequency_index: int,
    n: int,
    xi_eV: float,
    args: argparse.Namespace | SimpleNamespace,
) -> dict[str, Any]:
    q_lab = np.asarray(batch.q_lab, dtype=float)
    theta_1, theta_2 = args.plate_angles_rad
    plate_1_result = batch.plate_1
    plate_2_result = batch.plate_2[0]
    try:
        reflection_1, plate_1 = _plate_state(
            plate_1_result,
            frequency_index=frequency_index,
            q_lab=q_lab,
            theta_rad=float(theta_1),
            xi_eV=float(xi_eV),
            args=args,
        )
        reflection_2, plate_2 = _plate_state(
            plate_2_result,
            frequency_index=frequency_index,
            q_lab=q_lab,
            theta_rad=float(theta_2),
            xi_eV=float(xi_eV),
            args=args,
        )
        point = passive_sheet_logdet(
            reflection_1,
            reflection_2,
            separation_m=float(args.separation_nm) * 1e-9,
        )
        logdet = float(point.logdet)
        logdet_passed = bool(np.isfinite(logdet))
        hard_passed = bool(
            plate_1["hard_physical_passed"]
            and plate_2["hard_physical_passed"]
            and logdet_passed
        )
        error = ""
    except (ValueError, np.linalg.LinAlgError) as exc:
        plate_1 = locals().get("plate_1", {})
        plate_2 = locals().get("plate_2", {})
        logdet = float("nan")
        logdet_passed = False
        hard_passed = False
        error = str(exc)

    return {
        "n": int(n),
        "xi_eV": float(xi_eV),
        "two_plate_logdet": logdet,
        "logdet_passed": logdet_passed,
        "hard_physical_passed": hard_passed,
        "plate_1": plate_1,
        "plate_2": plate_2,
        "longitudinal_is_hard_gate": False,
        "error": error,
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _current_pss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(getattr(psutil.Process().memory_full_info(), "pss", 0))
    except (ImportError, OSError, AttributeError):
        return 0


def _evaluate_shift(
    *,
    pairing_name: str,
    n_grid: int,
    shift: tuple[float, float],
    active_by_label: dict[str, tuple[int, ...]],
    q_by_label: dict[str, np.ndarray],
    args: argparse.Namespace | SimpleNamespace,
    q_workers: int,
) -> dict[str, Any]:
    context_started = perf_counter()
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(float(args.delta0_eV))

    grid_started = perf_counter()
    grid = build_periodic_bz_grid(int(n_grid), shift)
    grid_seconds = perf_counter() - grid_started
    material_started = perf_counter()
    cache = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=KuboConfig.from_kelvin(
            omega_eV=0.0,
            temperature_K=float(args.temperature_K),
            eta_eV=float(args.eta_eV),
            output_si=False,
        ),
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
        grid=grid,
    )
    material_seconds = perf_counter() - material_started
    cache_array_bytes = numpy_array_bytes(cache.workspace)

    grouped: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for label, active_indices in active_by_label.items():
        if active_indices:
            grouped[tuple(active_indices)].append(label)

    point_states: dict[str, dict[str, Any]] = {label: {} for label in active_by_label}
    group_records: list[dict[str, Any]] = []
    for indices, labels in sorted(grouped.items()):
        xi_values = np.asarray(
            [matsubara_energy_eV(index, args.temperature_K) for index in indices],
            dtype=float,
        )
        tasks = tuple(
            QLabAngleTask(
                index=position,
                q_lab=q_by_label[label],
                theta_1_rad=float(args.plate_angles_rad[0]),
                theta_2_rad_values=np.asarray(
                    [float(args.plate_angles_rad[1])],
                    dtype=float,
                ),
            )
            for position, label in enumerate(labels)
        )
        workers = min(max(int(q_workers), 1), len(tasks))
        evaluator = ArbitraryQParallelEvaluator(
            material_cache=cache,
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=xi_values,
            temperature_K=float(args.temperature_K),
            eta_eV=float(args.eta_eV),
            process_workers=workers,
            canonical_reduction_block_size=int(args.canonical_block),
            runtime_chunk_size=int(args.runtime_chunk),
        )
        try:
            evaluated = evaluator.evaluate(tasks)
        finally:
            evaluator.close()
        execution = evaluator.metadata()
        for label, evaluated_task in zip(labels, evaluated, strict=True):
            for frequency_index, (n, xi_eV) in enumerate(
                zip(indices, xi_values, strict=True)
            ):
                point_states[label][str(n)] = _two_plate_state(
                    evaluated_task.result,
                    frequency_index=frequency_index,
                    n=int(n),
                    xi_eV=float(xi_eV),
                    args=args,
                )
        group_records.append(
            {
                "matsubara_indices": list(indices),
                "q_labels": list(labels),
                "workers": workers,
                "execution": execution,
            }
        )

    return {
        "pairing": pairing_name,
        "N": int(n_grid),
        "point_count": int(grid.point_count),
        "shift": list(shift),
        "grid_build_seconds": float(grid_seconds),
        "material_build_seconds": float(material_seconds),
        "material_cache_array_bytes": int(cache_array_bytes),
        "context_wall_seconds": float(perf_counter() - context_started),
        "process_pid": int(os.getpid()),
        "process_peak_rss_bytes": int(_peak_rss_bytes()),
        "process_current_pss_bytes": int(_current_pss_bytes()),
        "material_cache": cache.metadata(),
        "groups": group_records,
        "points": point_states,
    }


def _physics_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "temperature_K",
        "delta0_eV",
        "eta_eV",
        "degeneracy",
        "separation_nm",
        "ward_tolerance",
        "ward_absolute_tolerance",
        "condition_max",
        "static_energy_scale_eV",
        "static_reality_tolerance",
        "static_longitudinal_tolerance",
        "static_mixing_tolerance",
        "static_passivity_tolerance",
        "canonical_block",
        "runtime_chunk",
    )
    payload = {name: getattr(args, name) for name in names}
    payload["plate_angles_rad"] = tuple(float(value) for value in args.plate_angles_rad)
    return payload


def _context_cost(job: dict[str, Any]) -> float:
    active_evaluations = sum(
        len(indices) for indices in job["active_by_label"].values()
    )
    pairing_factor = 1.25 if job["pairing_name"] == "dwave" else 1.0
    return float(pairing_factor * int(job["n_grid"]) ** 2 * active_evaluations)


def _evaluate_context_job(job: dict[str, Any]) -> dict[str, Any]:
    threadpools, threadpool_passed = actual_threadpool_record()
    if bool(job["require_single_thread_runtime"]) and not threadpool_passed:
        raise RuntimeError(
            "context-parallel worker has a non-single-thread BLAS runtime: "
            f"{threadpools}"
        )
    record = _evaluate_shift(
        pairing_name=str(job["pairing_name"]),
        n_grid=int(job["n_grid"]),
        shift=tuple(float(value) for value in job["shift"]),
        active_by_label={
            str(label): tuple(int(value) for value in indices)
            for label, indices in job["active_by_label"].items()
        },
        q_by_label={
            str(label): np.asarray(value, dtype=float)
            for label, value in job["q_by_label"].items()
        },
        args=SimpleNamespace(**dict(job["args_payload"])),
        q_workers=1,
    )
    record["context_id"] = str(job["context_id"])
    record["shift_index"] = int(job["shift_index"])
    record["context_worker_actual_threadpools"] = [
        dict(item) for item in threadpools
    ]
    record["context_worker_actual_threadpool_passed"] = bool(threadpool_passed)
    return record


def _build_context_jobs(
    *,
    n_grid: int,
    args: argparse.Namespace,
    active: dict[tuple[str, str], set[int]],
    q_by_label: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    args_payload = _physics_args_payload(args)
    for pairing_name in args.pairings:
        active_by_label = {
            label: tuple(sorted(active[(pairing_name, label)]))
            for label in q_by_label
        }
        if not any(active_by_label.values()):
            continue
        for shift_index, shift in enumerate(args.shifts):
            jobs.append(
                {
                    "context_id": f"{pairing_name}:shift_{shift_index}",
                    "pairing_name": pairing_name,
                    "shift_index": int(shift_index),
                    "shift": tuple(float(value) for value in shift),
                    "n_grid": int(n_grid),
                    "active_by_label": active_by_label,
                    "q_by_label": {
                        label: q.tolist() for label, q in q_by_label.items()
                    },
                    "args_payload": args_payload,
                    "require_single_thread_runtime": True,
                }
            )
    return jobs


def _max_q_tasks_per_context(jobs: Sequence[dict[str, Any]]) -> int:
    maximum = 0
    for job in jobs:
        grouped: dict[tuple[int, ...], int] = defaultdict(int)
        for indices in job["active_by_label"].values():
            if indices:
                grouped[tuple(indices)] += 1
        maximum = max(maximum, max(grouped.values(), default=0))
    return max(maximum, 1)


def _execute_level(
    *,
    jobs: Sequence[dict[str, Any]],
    plan: CPUParallelPlan,
) -> dict[tuple[str, int], dict[str, Any]]:
    ordered = sorted(jobs, key=_context_cost, reverse=True)
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if plan.strategy == "context":
        validate_single_thread_blas_environment()
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(plan.context_workers),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_evaluate_context_job, job): job
                for job in ordered
            }
            for future in as_completed(futures):
                job = futures[future]
                record = future.result()
                key = (str(job["pairing_name"]), int(job["shift_index"]))
                records[key] = record
    else:
        q_workers = int(plan.q_workers) if plan.strategy == "q" else 1
        if q_workers > 1:
            validate_single_thread_blas_environment()
        for job in ordered:
            record = _evaluate_shift(
                pairing_name=str(job["pairing_name"]),
                n_grid=int(job["n_grid"]),
                shift=tuple(float(value) for value in job["shift"]),
                active_by_label={
                    str(label): tuple(int(value) for value in indices)
                    for label, indices in job["active_by_label"].items()
                },
                q_by_label={
                    str(label): np.asarray(value, dtype=float)
                    for label, value in job["q_by_label"].items()
                },
                args=SimpleNamespace(**dict(job["args_payload"])),
                q_workers=q_workers,
            )
            record["context_id"] = str(job["context_id"])
            record["shift_index"] = int(job["shift_index"])
            key = (str(job["pairing_name"]), int(job["shift_index"]))
            records[key] = record
    if len(records) != len(jobs):
        raise RuntimeError("parallel level execution did not return every material context")
    return records


def assess_frequency_level(
    *,
    current_by_shift: dict[str, dict[str, Any]],
    previous_by_shift: dict[str, dict[str, Any]] | None,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    ordered_labels = tuple(current_by_shift)
    current_logdets = [
        float(current_by_shift[label]["two_plate_logdet"])
        for label in ordered_labels
    ]
    hard_closure = all(
        bool(current_by_shift[label]["hard_physical_passed"])
        for label in ordered_labels
    )
    cross_shift = _scalar_spread(current_logdets, rtol=rtol, atol=atol)
    adjacent: dict[str, dict[str, Any]] | None = None
    adjacent_passed = False
    if previous_by_shift is not None:
        adjacent = {
            label: _mixed_scalar(
                previous_by_shift[label]["two_plate_logdet"],
                current_by_shift[label]["two_plate_logdet"],
                rtol=rtol,
                atol=atol,
            )
            for label in ordered_labels
        }
        adjacent_passed = all(bool(row["passed"]) for row in adjacent.values())
    accepted = bool(
        previous_by_shift is not None
        and hard_closure
        and cross_shift["passed"]
        and adjacent_passed
    )
    return {
        "hard_physical_closure_across_shifts": hard_closure,
        "two_plate_logdet_cross_shift": cross_shift,
        "adjacent_N_by_shift": adjacent,
        "adjacent_N_all_shifts_passed": adjacent_passed,
        "accepted_transition": accepted,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_payload(
    *,
    args: argparse.Namespace,
    q_by_label: dict[str, np.ndarray],
    result_records: dict[tuple[str, str, int], dict[str, Any]],
    execution_levels: list[dict[str, Any]],
    observed_cache_bytes_per_point: float | None,
    run_complete: bool,
) -> dict[str, Any]:
    point_results = list(result_records.values())
    all_established = all(
        row["sweet_spot"]["status"] == "established"
        for row in point_results
    )
    return {
        "schema": "transverse-point-sweet-spot-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "point_specific_transverse_BZ_working_and_audit_N_selection",
        "integration_family": "fixed_even_N_full_periodic_BZ_only",
        "local_refinement_present": False,
        "single_public_point_convergence_script": True,
        "q_points": [
            {"label": label, "q_lab": q.tolist()}
            for label, q in q_by_label.items()
        ],
        "pairings": list(args.pairings),
        "matsubara_indices": list(args.matsubara_indices),
        "plate_angles_deg": list(args.plate_angles_deg),
        "N_candidates": list(args.N_candidates),
        "shifts": [list(shift) for shift in args.shifts],
        "required_consecutive_passes": int(args.required_consecutive_passes),
        "logdet_rtol": float(args.logdet_rtol),
        "logdet_atol": float(args.logdet_atol),
        "cpu_parallel_policy": {
            "mode": str(args.parallel_mode),
            "requested_workers": int(args.workers),
            "memory_budget_gb": float(args.memory_budget_gb),
            "max_context_workers": int(args.max_context_workers),
            "memory_safety_factor": float(args.memory_safety_factor),
            "fallback_context_bytes_per_point": float(
                args.fallback_context_bytes_per_point
            ),
            "observed_material_cache_bytes_per_point_max": (
                None
                if observed_cache_bytes_per_point is None
                else float(observed_cache_bytes_per_point)
            ),
            "one_process_parallel_layer_only": True,
            "nested_process_pools": False,
            "blas_openmp_threads_per_process_required": 1,
            "q_axis_uses_fork_shared_readonly_material_cache": True,
            "context_axis_uses_spawn_and_builds_one_cache_per_worker": True,
        },
        "hard_gate_policy": {
            "operator_ward": True,
            "effective_ward": True,
            "sheet_finite_reality_mixing_passivity": True,
            "reflection_constructed": True,
            "two_plate_logdet_finite": True,
            "static_longitudinal": False,
            "historical_strict_static_aggregate": False,
        },
        "point_results": point_results,
        "execution_levels": execution_levels,
        "all_requested_sweet_spots_established": all_established,
        "point_specific_early_stop": True,
        "checkpoint_written_after_each_completed_N": True,
        "resume_from_checkpoint_implemented": False,
        "run_complete": bool(run_complete),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    q_by_label = {
        str(point["label"]): np.asarray(point["q_lab"], dtype=float)
        for point in args.q_points
    }
    result_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    active: dict[tuple[str, str], set[int]] = {}
    consecutive: dict[tuple[str, str, int], int] = {}
    previous_states: dict[
        tuple[str, str, int],
        dict[str, dict[str, Any]],
    ] = {}

    for pairing_name in args.pairings:
        for label in q_by_label:
            active[(pairing_name, label)] = set(args.matsubara_indices)
            for n in args.matsubara_indices:
                key = (pairing_name, label, int(n))
                consecutive[key] = 0
                result_records[key] = {
                    "pairing": pairing_name,
                    "q_label": label,
                    "q_lab": q_by_label[label].tolist(),
                    "n": int(n),
                    "history": [],
                    "sweet_spot": {
                        "status": "not_established",
                        "working_N": None,
                        "audit_N": None,
                    },
                }

    execution_levels: list[dict[str, Any]] = []
    observed_cache_bytes_per_point: float | None = None
    q_parallel_supported = "fork" in get_all_start_methods()

    for n_grid in args.N_candidates:
        jobs = _build_context_jobs(
            n_grid=int(n_grid),
            args=args,
            active=active,
            q_by_label=q_by_label,
        )
        if not jobs:
            break
        estimated_bytes = estimate_context_bytes(
            point_count=int(n_grid) ** 2,
            observed_bytes_per_point=observed_cache_bytes_per_point,
            safety_factor=float(args.memory_safety_factor),
            fallback_bytes_per_point=float(args.fallback_context_bytes_per_point),
        )
        plan = choose_cpu_parallel_plan(
            mode=str(args.parallel_mode),
            requested_workers=int(args.workers),
            context_count=len(jobs),
            max_q_tasks_per_context=_max_q_tasks_per_context(jobs),
            estimated_context_bytes=estimated_bytes,
            memory_budget_gb=float(args.memory_budget_gb),
            max_context_workers=int(args.max_context_workers),
            q_parallel_supported=q_parallel_supported,
        )
        level_started = perf_counter()
        records = _execute_level(jobs=jobs, plan=plan)

        for record in records.values():
            point_count = max(int(record["point_count"]), 1)
            measured = float(record["material_cache_array_bytes"]) / point_count
            observed_cache_bytes_per_point = (
                measured
                if observed_cache_bytes_per_point is None
                else max(observed_cache_bytes_per_point, measured)
            )

        level_record: dict[str, Any] = {
            "N": int(n_grid),
            "parallel_plan": plan.as_dict(),
            "level_wall_seconds": float(perf_counter() - level_started),
            "pairings": {},
        }
        for pairing_name in args.pairings:
            shift_records = [
                records[(pairing_name, shift_index)]
                for shift_index in range(len(args.shifts))
                if (pairing_name, shift_index) in records
            ]
            if not shift_records:
                continue
            level_record["pairings"][pairing_name] = shift_records
            active_by_label = {
                label: tuple(sorted(active[(pairing_name, label)]))
                for label in q_by_label
            }
            resolved_now: list[tuple[str, int]] = []
            shift_labels = tuple(
                f"shift_{index}:{tuple(record['shift'])}"
                for index, record in enumerate(shift_records)
            )
            for label, indices in active_by_label.items():
                for n in indices:
                    key = (pairing_name, label, int(n))
                    current_by_shift = {
                        shift_label: record["points"][label][str(n)]
                        for shift_label, record in zip(
                            shift_labels,
                            shift_records,
                            strict=True,
                        )
                    }
                    assessment = assess_frequency_level(
                        current_by_shift=current_by_shift,
                        previous_by_shift=previous_states.get(key),
                        rtol=float(args.logdet_rtol),
                        atol=float(args.logdet_atol),
                    )
                    if assessment["accepted_transition"]:
                        consecutive[key] += 1
                    else:
                        consecutive[key] = 0
                    history_row = {
                        "N": int(n_grid),
                        "shifts": current_by_shift,
                        **assessment,
                        "consecutive_accepted_transitions": int(consecutive[key]),
                    }
                    result_records[key]["history"].append(history_row)
                    previous_states[key] = current_by_shift
                    if consecutive[key] >= int(args.required_consecutive_passes):
                        history = result_records[key]["history"]
                        working_index = -int(args.required_consecutive_passes)
                        working_N = int(history[working_index]["N"])
                        audit_N = int(n_grid)
                        result_records[key]["sweet_spot"] = {
                            "status": "established",
                            "working_N": working_N,
                            "audit_N": audit_N,
                            "required_consecutive_passes": int(
                                args.required_consecutive_passes
                            ),
                            "criterion": (
                                "hard physical closure plus two-plate logdet "
                                "adjacent-N and cross-shift convergence"
                            ),
                        }
                        resolved_now.append((label, int(n)))
            for label, n in resolved_now:
                active[(pairing_name, label)].discard(n)

        execution_levels.append(level_record)
        checkpoint = _build_payload(
            args=args,
            q_by_label=q_by_label,
            result_records=result_records,
            execution_levels=execution_levels,
            observed_cache_bytes_per_point=observed_cache_bytes_per_point,
            run_complete=False,
        )
        _atomic_write(args.output, checkpoint)
        if not any(active.values()):
            break

    payload = _build_payload(
        args=args,
        q_by_label=q_by_label,
        result_records=result_records,
        execution_levels=execution_levels,
        observed_cache_bytes_per_point=observed_cache_bytes_per_point,
        run_complete=True,
    )
    _atomic_write(args.output, payload)

    summary = {
        "output": str(args.output),
        "all_requested_sweet_spots_established": payload[
            "all_requested_sweet_spots_established"
        ],
        "parallel_plans": [
            {
                "N": level["N"],
                "strategy": level["parallel_plan"]["strategy"],
                "total_worker_budget": level["parallel_plan"][
                    "total_worker_budget"
                ],
                "context_workers": level["parallel_plan"]["context_workers"],
                "q_workers": level["parallel_plan"]["q_workers"],
                "reason": level["parallel_plan"]["reason"],
            }
            for level in execution_levels
        ],
        "points": [
            {
                "pairing": row["pairing"],
                "q_label": row["q_label"],
                "n": row["n"],
                **row["sweet_spot"],
                "evaluated_N": [item["N"] for item in row["history"]],
            }
            for row in payload["point_results"]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
