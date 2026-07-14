"""Detailed timing profile for the total zero/positive Matsubara orbit backend.

The profiler exercises the same full-period composite Gauss, batched material/q
workspaces, exact zero-frequency divided difference, positive-frequency batch, and
POSIX-fork path used by the blocking preflight and staged scan.  Stage times are
reported as summed worker-seconds; wall times are reported separately so CPU work and
parallel scaling cannot be confused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Sequence

import numpy as np

from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_matsubara_orbit_gauss

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/orbit_gauss_timing_profile/timing_profile.json"
)
_STAGE_FIELDS = (
    ("material_workspace", "material_workspace_seconds"),
    ("q_workspace", "q_workspace_seconds"),
    ("kubo_factors", "kubo_factor_seconds"),
    ("kubo_contraction", "kubo_contraction_seconds"),
    ("primitive_packing", "primitive_packing_seconds"),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairings", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"]
    )
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=1)
    parser.add_argument("--my", type=int, default=1)
    parser.add_argument(
        "--matsubara-indices", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--transverse-order", type=int, default=64)
    parser.add_argument("--panel-count", type=int, default=16)
    parser.add_argument("--transverse-workers", type=int, default=8)
    parser.add_argument("--transverse-task-size", type=int, default=4)
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument("--minimum-cpu-wall-ratio", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.nk <= 0 or args.transverse_order <= 0 or args.panel_count <= 0:
        parser.error("nk, transverse order, and panel count must be positive")
    if args.transverse_order % args.panel_count != 0:
        parser.error("transverse order must be divisible by panel count")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if args.transverse_workers <= 1 or args.transverse_task_size <= 0:
        parser.error("timing profile requires at least two workers and positive task size")
    indices = tuple(sorted(set(int(value) for value in args.matsubara_indices)))
    if any(index < 0 for index in indices):
        parser.error("Matsubara indices must be non-negative")
    if 0 not in indices or not any(index > 0 for index in indices):
        parser.error("timing profile requires exact n=0 and at least one positive index")
    args.matsubara_indices = indices
    args.pairings = tuple(dict.fromkeys(args.pairings))
    for name in ("minimum_speedup", "minimum_cpu_wall_ratio"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def _origin_count(mx: int, my: int, subgrid_average: str) -> int:
    common = math.gcd(abs(int(mx)), abs(int(my)))
    return 2 if subgrid_average == "auto" and common % 2 == 1 else 1


def _budget(args: argparse.Namespace) -> int:
    return int(
        args.nk
        * _origin_count(args.mx, args.my, args.subgrid_average)
        * args.transverse_order
    )


def _xi_values(args: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        [
            0.0
            if index == 0
            else matsubara_energy_eV(index, args.temperature_K)
            for index in args.matsubara_indices
        ],
        dtype=float,
    )


def _run(
    *,
    args: argparse.Namespace,
    pairing_name: str,
    workers: int,
) -> tuple[object, dict[str, Any]]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    started = time.perf_counter()
    integrated = integrate_matsubara_orbit_gauss(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=_xi_values(args),
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        transverse_order=args.transverse_order,
        panel_count=args.panel_count,
        shift_s=args.shift_s,
        subgrid_average=args.subgrid_average,
        max_point_evaluations=_budget(args),
        transverse_workers=workers,
        transverse_task_size=args.transverse_task_size,
    )
    total_wall = float(time.perf_counter() - started)
    quadrature = integrated.quadrature
    profile = integrated.evaluator_profile
    worker_total = float(profile.total_seconds)
    quadrature_wall = float(quadrature.wall_seconds)
    stage_worker_seconds = {
        label: float(getattr(profile, field)) for label, field in _STAGE_FIELDS
    }
    stage_worker_fractions = {
        label: value / max(worker_total, np.finfo(float).tiny)
        for label, value in stage_worker_seconds.items()
    }
    record: dict[str, Any] = {
        "workers": int(workers),
        "execution_strategy": str(quadrature.execution_strategy),
        "total_call_wall_seconds": total_wall,
        "quadrature_wall_seconds": quadrature_wall,
        "pool_startup_and_postprocess_wall_seconds": max(total_wall - quadrature_wall, 0.0),
        "worker_seconds_total": worker_total,
        "worker_seconds_per_callback": float(profile.seconds_per_callback),
        "worker_cpu_wall_ratio": worker_total / max(quadrature_wall, np.finfo(float).tiny),
        "wall_seconds_per_node": total_wall / max(int(args.transverse_order), 1),
        "quadrature_seconds_per_node": quadrature_wall / max(int(args.transverse_order), 1),
        "stage_worker_seconds": stage_worker_seconds,
        "stage_worker_fractions": stage_worker_fractions,
        "callbacks": int(profile.callbacks),
        "complete_orbit_points": int(profile.complete_orbit_points),
        "point_evaluations": int(quadrature.point_evaluations),
        "frequency_count": len(args.matsubara_indices),
        "callbacks_not_multiplied_by_frequency_count": bool(
            profile.callbacks == args.transverse_order
        ),
        "material_workspace_implementation": str(
            profile.material_workspace_implementation
        ),
        "q_workspace_implementation": str(profile.q_workspace_implementation),
        "full_transverse_period_integrated": bool(
            quadrature.full_transverse_period_integrated
        ),
        "symmetry_reduction_applied": bool(quadrature.symmetry_reduction_applied),
        "transverse_task_size": int(quadrature.transverse_task_size),
        "transverse_task_count": int(quadrature.transverse_task_count),
    }
    return integrated, record


def _pairing_record(args: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    serial, serial_record = _run(args=args, pairing_name=pairing_name, workers=1)
    parallel, parallel_record = _run(
        args=args,
        pairing_name=pairing_name,
        workers=args.transverse_workers,
    )
    serial_value = np.asarray(serial.quadrature.value, dtype=complex)
    parallel_value = np.asarray(parallel.quadrature.value, dtype=complex)
    difference = float(np.linalg.norm(serial_value - parallel_value))
    scale = max(
        float(np.linalg.norm(serial_value)),
        float(np.linalg.norm(parallel_value)),
        np.finfo(float).tiny,
    )
    speedup = float(
        serial_record["total_call_wall_seconds"]
        / max(parallel_record["total_call_wall_seconds"], np.finfo(float).tiny)
    )
    optimized = bool(
        parallel_record["material_workspace_implementation"]
        == "batched_model_capability"
        and parallel_record["q_workspace_implementation"]
        == "batched_model_capability"
        and parallel_record["execution_strategy"]
        == "fork_process_transverse_nodes_ordered_parent_reduction"
        and parallel_record["callbacks_not_multiplied_by_frequency_count"]
        and not parallel_record["symmetry_reduction_applied"]
        and speedup >= args.minimum_speedup
        and parallel_record["worker_cpu_wall_ratio"] >= args.minimum_cpu_wall_ratio
    )
    return {
        "pairing": pairing_name,
        "serial": serial_record,
        "parallel": parallel_record,
        "serial_parallel_absolute": difference,
        "serial_parallel_relative": difference / scale,
        "serial_parallel_exact_equal": bool(np.array_equal(serial_value, parallel_value)),
        "speedup": speedup,
        "optimization_sufficient": optimized,
    }


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "total Matsubara complete-orbit timing profile",
        "=" * 88,
        (
            f"nk={payload['arguments']['nk']}; "
            f"m=({payload['arguments']['mx']},{payload['arguments']['my']}); "
            f"order={payload['arguments']['transverse_order']}; "
            f"panels={payload['arguments']['panel_count']}; "
            f"n={tuple(payload['arguments']['matsubara_indices'])}"
        ),
        "stage percentages are fractions of summed evaluator worker-seconds",
        "quadrature wall includes orbit geometry, dispatch/wait, and ordered Kahan reduction",
        "",
    ]
    for record in payload["pairing_records"]:
        parallel = record["parallel"]
        serial = record["serial"]
        lines.extend(
            [
                f"[{record['pairing']}]",
                (
                    f"serial wall={serial['total_call_wall_seconds']:.6f}s; "
                    f"parallel wall={parallel['total_call_wall_seconds']:.6f}s; "
                    f"speedup={record['speedup']:.3f}x; "
                    f"CPU/wall={parallel['worker_cpu_wall_ratio']:.3f}"
                ),
                (
                    f"quadrature wall={parallel['quadrature_wall_seconds']:.6f}s; "
                    "pool startup + postprocess="
                    f"{parallel['pool_startup_and_postprocess_wall_seconds']:.6f}s"
                ),
            ]
        )
        for label, _ in _STAGE_FIELDS:
            seconds = parallel["stage_worker_seconds"][label]
            fraction = parallel["stage_worker_fractions"][label]
            lines.append(f"  {label:<24s} {seconds:12.6f} worker-s  {fraction:8.2%}")
        lines.extend(
            [
                (
                    f"  callbacks={parallel['callbacks']}; frequencies="
                    f"{parallel['frequency_count']}; shared="
                    f"{parallel['callbacks_not_multiplied_by_frequency_count']}"
                ),
                (
                    f"  exact serial/process equality="
                    f"{record['serial_parallel_exact_equal']}; "
                    f"relative={record['serial_parallel_relative']:.3e}"
                ),
                f"  optimization sufficient={record['optimization_sufficient']}",
                "",
            ]
        )
    status = payload["status"]
    lines.extend(
        [
            f"all optimization checks passed = {status['all_optimization_checks_passed']}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    records = [_pairing_record(args, pairing) for pairing in args.pairings]
    payload: dict[str, Any] = {
        "schema": "total_matsubara_orbit_timing_profile_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "pairing_records": records,
        "status": {
            "zero_matsubara_included": True,
            "zero_and_positive_share_eigensystems": True,
            "all_optimization_checks_passed": all(
                bool(record["optimization_sufficient"]) for record in records
            ),
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = _summary(payload)
    output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"JSON:    {output}")
    print(f"Summary: {output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
