"""Blocking correctness and performance preflight for the total Matsubara scan.

The preflight exercises the exact production-facing orchestration path rather than a
mock evaluator. It requires one combined ``[0, positive...]`` batch, verifies exact
zero-frequency physical postprocessing, compares serial and POSIX-fork primitive
integrals, checks batched workspace metadata and callback counts, measures real wall
speedup, and cross-checks d-wave ``n=0`` against the independent legacy exact-static
primitive implementation on a small deterministic reference grid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.dwave_adaptive_bond_metric import (
    AdaptiveStaticValidationConfig,
    postprocess_adaptive_bond_metric_static,
)
from validation.lib.dwave_commensurate_orbit_gauss import (
    integrate_commensurate_orbit_gauss_vector,
)
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.dwave_static_primitives import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_matsubara_orbit_gauss

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/orbit_gauss_preflight/preflight.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairings",
        nargs="+",
        choices=("spm", "dwave"),
        default=["spm", "dwave"],
    )
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=1)
    parser.add_argument("--my", type=int, default=1)
    parser.add_argument(
        "--matsubara-indices", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--transverse-order", type=int, default=32)
    parser.add_argument("--panel-count", type=int, default=16)
    parser.add_argument("--transverse-workers", type=int, default=8)
    parser.add_argument("--transverse-task-size", type=int, default=4)
    parser.add_argument("--legacy-static-nk", type=int, default=32)
    parser.add_argument("--legacy-static-order", type=int, default=16)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument(
        "--minimum-parallel-cpu-wall-ratio", type=float, default=1.5
    )
    parser.add_argument("--comparison-rtol", type=float, default=2e-11)
    parser.add_argument("--comparison-atol", type=float, default=2e-12)
    parser.add_argument(
        "--require-physical",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument(
        "--subgrid-average", choices=("auto", "none"), default="auto"
    )
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.nk <= 0 or args.legacy_static_nk <= 0:
        parser.error("nk values must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if args.transverse_order <= 0 or args.legacy_static_order <= 0:
        parser.error("transverse orders must be positive")
    if args.panel_count <= 0 or args.transverse_order % args.panel_count != 0:
        parser.error("transverse order must be divisible by positive panel count")
    if args.transverse_workers <= 1:
        parser.error("preflight requires at least two transverse workers")
    if args.transverse_task_size <= 0:
        parser.error("transverse task size must be positive")
    indices = tuple(sorted(set(int(value) for value in args.matsubara_indices)))
    if (
        any(index < 0 for index in indices)
        or 0 not in indices
        or not any(index > 0 for index in indices)
    ):
        parser.error(
            "preflight requires index 0 and at least one positive Matsubara index"
        )
    args.matsubara_indices = indices
    args.pairings = tuple(dict.fromkeys(args.pairings))
    for name in (
        "minimum_speedup",
        "minimum_parallel_cpu_wall_ratio",
        "comparison_rtol",
        "comparison_atol",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    return args


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _xi_values(indices: Sequence[int], temperature_K: float) -> np.ndarray:
    return np.asarray(
        [
            0.0
            if index == 0
            else matsubara_energy_eV(index, temperature_K)
            for index in indices
        ],
        dtype=float,
    )


def _budget(
    nk: int,
    mx: int,
    my: int,
    subgrid_average: str,
    order: int,
) -> int:
    common = int(np.gcd(abs(int(mx)), abs(int(my))))
    origins = 2 if subgrid_average == "auto" and common % 2 == 1 else 1
    return int(nk) * origins * int(order)


def _run_combined(
    *,
    args: argparse.Namespace,
    pairing_name: str,
    workers: int,
    nk: int | None = None,
    order: int | None = None,
    panel_count: int | None = None,
    indices: Sequence[int] | None = None,
):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        pairing_name,
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(args.delta0_eV)
    selected_nk = int(args.nk if nk is None else nk)
    selected_order = int(args.transverse_order if order is None else order)
    selected_panels = int(args.panel_count if panel_count is None else panel_count)
    selected_indices = args.matsubara_indices if indices is None else tuple(indices)
    xi_values = _xi_values(selected_indices, args.temperature_K)
    started = time.perf_counter()
    integrated = integrate_matsubara_orbit_gauss(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_values,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        nk=selected_nk,
        mx=args.mx,
        my=args.my,
        transverse_order=selected_order,
        panel_count=selected_panels,
        shift_s=args.shift_s,
        subgrid_average=args.subgrid_average,
        max_point_evaluations=_budget(
            selected_nk,
            args.mx,
            args.my,
            args.subgrid_average,
            selected_order,
        ),
        transverse_workers=workers,
        transverse_task_size=args.transverse_task_size,
    )
    wall = float(time.perf_counter() - started)
    return integrated, wall


def _physical_rows(
    args: argparse.Namespace,
    integrated,
) -> list[dict[str, Any]]:
    config = OrbitAcceptancePhysicsConfig(
        degeneracy=args.degeneracy,
        separation_nm=args.separation_nm,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
    )
    rows: list[dict[str, Any]] = []
    q = np.asarray(integrated.quadrature.q_model, dtype=float)
    for index, xi, components, rhs in zip(
        args.matsubara_indices,
        integrated.xi_eV_values,
        integrated.components,
        integrated.rhs,
        strict=True,
    ):
        physical = evaluate_matsubara_pipeline(
            components=components,
            rhs=rhs,
            q_model=q,
            xi_eV=float(xi),
            config=config,
        )
        rows.append(
            {
                "matsubara_index": int(index),
                "response_sector": str(physical["response_sector"]),
                "physical_passed": bool(physical["physical_passed"]),
                "ward_passed": bool(physical["ward_passed"]),
                "strict_static_ward_passed": bool(
                    physical["strict_static_ward_passed"]
                ),
                "sheet_validation_passed": bool(
                    physical["sheet_validation_passed"]
                ),
                "reflection_constructed": bool(
                    physical["reflection_constructed"]
                ),
                "logdet_passed": bool(physical["logdet_passed"]),
                "error": str(physical["error"]),
            }
        )
    return rows


def _comparison_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    relative = absolute / max(scale, np.finfo(float).tiny)
    threshold = float(atol) + float(rtol) * scale
    mixed_ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "absolute": absolute,
        "scale": scale,
        "relative": relative,
        "threshold": threshold,
        "mixed_ratio": mixed_ratio,
        "passed": bool(np.isfinite(mixed_ratio) and mixed_ratio <= 1.0),
        "denominator_collapsed": bool(float(rtol) * scale <= float(atol)),
    }


def _serial_parallel_record(
    args: argparse.Namespace,
    pairing_name: str,
) -> dict[str, Any]:
    serial, serial_wall = _run_combined(
        args=args,
        pairing_name=pairing_name,
        workers=1,
    )
    parallel, parallel_wall = _run_combined(
        args=args,
        pairing_name=pairing_name,
        workers=args.transverse_workers,
    )
    primitive_comparison = _comparison_metrics(
        serial.quadrature.value,
        parallel.quadrature.value,
        rtol=args.comparison_rtol,
        atol=args.comparison_atol,
    )
    profile = parallel.evaluator_profile
    speedup = serial_wall / max(parallel_wall, np.finfo(float).tiny)
    cpu_wall_ratio = profile.total_seconds / max(
        parallel.quadrature.wall_seconds,
        np.finfo(float).tiny,
    )
    execution_expected = "fork_process_transverse_nodes_ordered_parent_reduction"
    metadata = parallel.components[0].metadata
    physical_rows = _physical_rows(args, parallel)
    correctness = bool(
        primitive_comparison["passed"]
        and (
            not args.require_physical
            or all(row["physical_passed"] for row in physical_rows)
        )
    )
    optimization = bool(
        profile.material_workspace_implementation == "batched_model_capability"
        and profile.q_workspace_implementation == "batched_model_capability"
        and parallel.quadrature.execution_strategy == execution_expected
        and profile.callbacks == parallel.quadrature.transverse_evaluations
        and profile.callbacks == args.transverse_order
        and profile.complete_orbit_points == parallel.quadrature.point_evaluations
        and bool(metadata.get("zero_and_positive_frequencies_share_eigensystems"))
        and bool(metadata.get("exact_zero_uses_divided_difference"))
        and not bool(metadata.get("symmetry_reduction_applied"))
        and bool(metadata.get("full_transverse_period_integrated"))
        and speedup >= args.minimum_speedup
        and cpu_wall_ratio >= args.minimum_parallel_cpu_wall_ratio
    )
    return {
        "pairing": pairing_name,
        "serial_wall_seconds": serial_wall,
        "parallel_wall_seconds": parallel_wall,
        "speedup": speedup,
        "parallel_cpu_wall_ratio": cpu_wall_ratio,
        "serial_seconds_per_node": serial_wall / args.transverse_order,
        "parallel_seconds_per_node": parallel_wall / args.transverse_order,
        "primitive_serial_parallel_absolute": primitive_comparison["absolute"],
        "primitive_serial_parallel_scale": primitive_comparison["scale"],
        "primitive_serial_parallel_relative": primitive_comparison["relative"],
        "primitive_serial_parallel_mixed_ratio": primitive_comparison["mixed_ratio"],
        "primitive_serial_parallel_denominator_collapsed": primitive_comparison[
            "denominator_collapsed"
        ],
        "material_workspace_implementation": profile.material_workspace_implementation,
        "q_workspace_implementation": profile.q_workspace_implementation,
        "execution_strategy": parallel.quadrature.execution_strategy,
        "evaluator_callbacks": int(profile.callbacks),
        "transverse_evaluations": int(parallel.quadrature.transverse_evaluations),
        "point_evaluations": int(parallel.quadrature.point_evaluations),
        "frequency_count": len(args.matsubara_indices),
        "callbacks_not_multiplied_by_frequency_count": bool(
            profile.callbacks == args.transverse_order
        ),
        "full_transverse_period_integrated": bool(
            parallel.quadrature.full_transverse_period_integrated
        ),
        "symmetry_reduction_applied": bool(
            parallel.quadrature.symmetry_reduction_applied
        ),
        "physical_rows": physical_rows,
        "correctness_passed": correctness,
        "optimization_passed": optimization,
    }


def _component_fields(component: object) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(getattr(component, name), dtype=complex)
        for name in (
            "bare_bubble",
            "direct",
            "bare_total",
            "collective_bubble",
            "collective_counterterm",
            "em_collective_left",
            "collective_em_right",
            "amplitude_phase_schur",
            "gauge_restored",
        )
    )


def _legacy_static_record(args: argparse.Namespace) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    nk = int(args.legacy_static_nk)
    order = int(args.legacy_static_order)
    q = (2.0 * np.pi / float(nk)) * np.asarray(
        [args.mx, args.my], dtype=float
    )
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )
    legacy_quadrature = integrate_commensurate_orbit_gauss_vector(
        context.evaluate_complex,
        nk=nk,
        mx=args.mx,
        my=args.my,
        transverse_order=order,
        shift_s=args.shift_s,
        subgrid_average=args.subgrid_average,
        chunk_size=1024,
        max_point_evaluations=_budget(
            nk, args.mx, args.my, args.subgrid_average, order
        ),
    )
    legacy_components, legacy_rhs, _ = assemble_dwave_static_primitives(
        context,
        legacy_quadrature.value,
        metadata={"preflight_reference": True},
    )
    processed = postprocess_adaptive_bond_metric_static(
        legacy_components,
        legacy_rhs,
        ansatz=ansatz,
        q_model=q,
        config=AdaptiveStaticValidationConfig(
            mixed_ward_tolerance=args.ward_tolerance,
            mixed_ward_absolute_tolerance=args.ward_absolute_tolerance,
            condition_max=args.condition_max,
            degeneracy=args.degeneracy,
        ),
    )
    generic, _ = _run_combined(
        args=args,
        pairing_name="dwave",
        workers=1,
        nk=nk,
        order=order,
        panel_count=1,
        indices=(0,),
    )
    component_comparisons = [
        _comparison_metrics(
            left,
            right,
            rtol=args.comparison_rtol,
            atol=args.comparison_atol,
        )
        for left, right in zip(
            _component_fields(processed.components),
            _component_fields(generic.components[0]),
            strict=True,
        )
    ]
    rhs_left = _comparison_metrics(
        legacy_rhs.left,
        generic.rhs[0].left,
        rtol=args.comparison_rtol,
        atol=args.comparison_atol,
    )
    rhs_right = _comparison_metrics(
        legacy_rhs.right,
        generic.rhs[0].right,
        rtol=args.comparison_rtol,
        atol=args.comparison_atol,
    )
    rhs_mixed_ratio = max(
        float(rhs_left["mixed_ratio"]),
        float(rhs_right["mixed_ratio"]),
    )
    maximum_component_relative = max(
        float(value["relative"]) for value in component_comparisons
    )
    maximum_component_mixed_ratio = max(
        float(value["mixed_ratio"]) for value in component_comparisons
    )
    passed = bool(
        all(bool(value["passed"]) for value in component_comparisons)
        and bool(rhs_left["passed"])
        and bool(rhs_right["passed"])
    )
    return {
        "nk": nk,
        "order": order,
        "maximum_component_relative": maximum_component_relative,
        "maximum_component_mixed_ratio": maximum_component_mixed_ratio,
        "rhs_absolute_max": max(
            float(rhs_left["absolute"]),
            float(rhs_right["absolute"]),
        ),
        "rhs_scale_max": max(
            float(rhs_left["scale"]),
            float(rhs_right["scale"]),
        ),
        "rhs_relative": max(
            float(rhs_left["relative"]),
            float(rhs_right["relative"]),
        ),
        "rhs_mixed_ratio": rhs_mixed_ratio,
        "rhs_denominator_collapsed": bool(
            rhs_left["denominator_collapsed"]
            or rhs_right["denominator_collapsed"]
        ),
        "legacy_strict_static_passed": bool(processed.strict.passed),
        "legacy_sheet_validation_passed": bool(processed.sheet.validation.passed),
        "passed": passed,
    }


def main() -> None:
    args = _parse_args()
    print("total Matsubara orbit-Gauss preflight", flush=True)
    print(
        f"speed grid: nk={args.nk}, m=({args.mx},{args.my}), "
        f"order={args.transverse_order}, panels={args.panel_count}, "
        f"workers={args.transverse_workers}, n={args.matsubara_indices}",
        flush=True,
    )
    records = [_serial_parallel_record(args, pairing) for pairing in args.pairings]
    legacy = _legacy_static_record(args)
    correctness_passed = bool(
        all(record["correctness_passed"] for record in records)
        and legacy["passed"]
    )
    optimization_passed = bool(
        all(record["optimization_passed"] for record in records)
    )
    passed = bool(correctness_passed and optimization_passed)

    for record in records:
        print(
            f"{record['pairing']}: serial={record['serial_wall_seconds']:.3f}s, "
            f"parallel={record['parallel_wall_seconds']:.3f}s, "
            f"speedup={record['speedup']:.2f}x, "
            f"CPU/wall={record['parallel_cpu_wall_ratio']:.2f}, "
            f"mixed={record['primitive_serial_parallel_mixed_ratio']:.3e}, "
            f"correct={record['correctness_passed']}, "
            f"optimized={record['optimization_passed']}",
            flush=True,
        )
    print(
        "legacy exact-static d-wave agreement: component mixed="
        f"{legacy['maximum_component_mixed_ratio']:.3e}; "
        f"RHS mixed={legacy['rhs_mixed_ratio']:.3e}; "
        f"RHS relative={legacy['rhs_relative']:.3e}; "
        f"denominator_collapsed={legacy['rhs_denominator_collapsed']}; "
        f"passed={legacy['passed']}",
        flush=True,
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "total_matsubara_orbit_gauss_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "pairing_records": records,
        "legacy_exact_static_reference": legacy,
        "status": {
            "correctness_passed": correctness_passed,
            "optimization_passed": optimization_passed,
            "passed": passed,
            "zero_matsubara_included": True,
            "zero_uses_exact_static_divided_difference": True,
            "zero_conductivity_division_used": False,
            "combined_zero_positive_batch_verified": True,
            "formal_scan_allowed": passed,
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"preflight manifest: {output}")
    if not passed:
        raise SystemExit("preflight failed; formal total scan is blocked")


if __name__ == "__main__":
    main()
