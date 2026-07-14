"""Architecture and real-hardware performance preflight for arbitrary-q periodic BZ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.workflows.arbitrary_q_parallel import (
    ArbitraryQParallelEvaluator,
    QLabAngleTask,
    thread_environment,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_performance_preflight/preflight.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"])
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--q-tasks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--canonical-block-size", type=int, default=4096)
    parser.add_argument("--runtime-chunk-sizes", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--minimum-speedup", type=float, default=4.0)
    parser.add_argument("--minimum-cpu-wall-ratio", type=float, default=4.0)
    parser.add_argument("--maximum-pool-overhead-fraction", type=float, default=0.05)
    parser.add_argument("--comparison-rtol", type=float, default=2e-11)
    parser.add_argument("--comparison-atol", type=float, default=2e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.N <= 0 or args.N % 2:
        parser.error("--N must be positive and even")
    if args.q_tasks < args.workers or args.workers <= 1:
        parser.error("preflight requires q_tasks >= workers > 1")
    if args.canonical_block_size <= 0 or args.canonical_block_size % 2:
        parser.error("canonical block size must be positive and even")
    for value in args.runtime_chunk_sizes:
        if value < args.canonical_block_size or value % args.canonical_block_size:
            parser.error("runtime chunks must be multiples of canonical block size")
    indices = tuple(sorted(set(int(value) for value in args.matsubara_indices)))
    if 0 not in indices or not any(value > 0 for value in indices):
        parser.error("preflight requires index 0 and at least one positive index")
    args.matsubara_indices = indices
    args.pairings = tuple(dict.fromkeys(args.pairings))
    return args


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _xi_values(indices: Sequence[int], temperature_K: float) -> np.ndarray:
    return np.asarray(
        [
            0.0 if index == 0 else matsubara_energy_eV(index, temperature_K)
            for index in indices
        ],
        dtype=float,
    )


def _tasks(count: int) -> tuple[QLabAngleTask, ...]:
    base = (2.0 * np.pi / 1256.0) * np.asarray([6.0, 4.0])
    tasks: list[QLabAngleTask] = []
    for index in range(int(count)):
        angle = np.deg2rad(3.0 + 11.0 * index)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        q_lab = (0.85 + 0.03 * index) * (rotation @ base)
        tasks.append(
            QLabAngleTask(
                index=index,
                q_lab=q_lab,
                theta_1_rad=0.0,
                theta_2_rad_values=np.asarray([0.0, np.deg2rad(17.0)]),
            )
        )
    return tuple(tasks)


def _signature(task_result) -> np.ndarray:
    values: list[np.ndarray] = []
    for response in (task_result.result.plate_1, *task_result.result.plate_2):
        for component, rhs in zip(response.components, response.rhs, strict=True):
            values.append(np.asarray(component.gauge_restored, dtype=complex).reshape(-1))
            values.append(np.asarray(rhs.left, dtype=complex).reshape(-1))
    return np.concatenate(values)


def _mixed_comparison(left: np.ndarray, right: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "absolute": absolute,
        "scale": scale,
        "threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _architecture_rows(results) -> dict[str, Any]:
    profiles = []
    operator_passed = True
    cache_hits = 0
    for task in results:
        for response in (task.result.plate_1, *task.result.plate_2):
            profiles.append(response.profile)
            operator_passed = operator_passed and response.operator_ward.passed
        cache_hits += int(task.result.response_cache_metadata["hits"])
    frequency_counts = {profile.frequency_count for profile in profiles}
    counterterm_counts = {profile.counterterm_add_count for profile in profiles}
    shifted_per_block = {
        profile.shifted_eigensystem_build_count // max(profile.canonical_block_count, 1)
        for profile in profiles
    }
    return {
        "response_count": len(profiles),
        "operator_ward_all_passed": bool(operator_passed),
        "frequency_counts": sorted(frequency_counts),
        "counterterm_add_counts": sorted(counterterm_counts),
        "shifted_eigensystem_builds_per_canonical_block": sorted(shifted_per_block),
        "exact_response_cache_hits": cache_hits,
        "passed": bool(
            operator_passed
            and counterterm_counts == {1}
            and shifted_per_block.issubset({0, 2})
            and cache_hits >= len(results)
        ),
    }


def _run_pairing(args: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    xi_values = _xi_values(args.matsubara_indices, args.temperature_K)
    grid = build_periodic_bz_grid(args.N, (0.5, 0.5))
    config = KuboConfig.from_kelvin(
        omega_eV=float(xi_values[0]),
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    material = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        options=options,
        grid=grid,
    )
    readonly = all(
        not np.asarray(getattr(material.workspace, name)).flags.writeable
        for name in (
            "k_points",
            "k_weights",
            "midpoint_energies",
            "midpoint_states",
            "midpoint_occupations",
            "collective_counterterm_matrix",
        )
    )
    tasks = _tasks(args.q_tasks)
    records: list[dict[str, Any]] = []
    baseline_signatures = None
    all_passed = readonly and material.build_count == 1

    for runtime_chunk in args.runtime_chunk_sizes:
        common = dict(
            material_cache=material,
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=xi_values,
            temperature_K=args.temperature_K,
            eta_eV=args.eta_eV,
            canonical_reduction_block_size=args.canonical_block_size,
            runtime_chunk_size=runtime_chunk,
        )
        started = perf_counter()
        with ArbitraryQParallelEvaluator(process_workers=1, **common) as serial:
            serial_results = serial.evaluate(tasks)
        serial_wall = float(perf_counter() - started)

        started = perf_counter()
        with ArbitraryQParallelEvaluator(process_workers=args.workers, **common) as parallel:
            parallel_results = parallel.evaluate(tasks)
        parallel_wall = float(perf_counter() - started)
        parallel_metadata = parallel.metadata()

        comparisons = [
            _mixed_comparison(
                _signature(left),
                _signature(right),
                atol=args.comparison_atol,
                rtol=args.comparison_rtol,
            )
            for left, right in zip(serial_results, parallel_results, strict=True)
        ]
        signatures = tuple(_signature(item) for item in serial_results)
        chunk_comparisons: list[dict[str, Any]] = []
        if baseline_signatures is None:
            baseline_signatures = signatures
        else:
            chunk_comparisons = [
                _mixed_comparison(
                    left,
                    right,
                    atol=args.comparison_atol,
                    rtol=args.comparison_rtol,
                )
                for left, right in zip(baseline_signatures, signatures, strict=True)
            ]

        speedup = serial_wall / max(parallel_wall, np.finfo(float).tiny)
        worker_seconds = sum(item.worker_seconds for item in parallel_results)
        cpu_wall = worker_seconds / max(parallel_wall, np.finfo(float).tiny)
        pool_overhead = (
            float(parallel_metadata["pool_startup_seconds"])
            + float(parallel_metadata["pool_shutdown_seconds"])
        ) / max(parallel_wall, np.finfo(float).tiny)
        architecture = _architecture_rows(parallel_results)
        passed = bool(
            architecture["passed"]
            and all(item["passed"] for item in comparisons)
            and all(item["passed"] for item in chunk_comparisons)
            and speedup >= args.minimum_speedup
            and cpu_wall >= args.minimum_cpu_wall_ratio
            and pool_overhead <= args.maximum_pool_overhead_fraction
        )
        all_passed = all_passed and passed
        records.append(
            {
                "runtime_chunk_size": int(runtime_chunk),
                "serial_wall_seconds": serial_wall,
                "parallel_wall_seconds": parallel_wall,
                "speedup": speedup,
                "summed_worker_seconds": worker_seconds,
                "cpu_wall_ratio": cpu_wall,
                "pool_overhead_fraction": pool_overhead,
                "serial_process_comparisons": comparisons,
                "chunk_size_comparisons": chunk_comparisons,
                "architecture": architecture,
                "parallel_metadata": parallel_metadata,
                "passed": passed,
            }
        )

    return {
        "pairing": pairing_name,
        "material_cache": material.metadata(),
        "material_arrays_readonly": readonly,
        "records": records,
        "passed": bool(all_passed),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    pairing_rows = [_run_pairing(args, name) for name in args.pairings]
    passed = all(row["passed"] for row in pairing_rows)
    payload = {
        "schema": "arbitrary-q-performance-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "platform": platform.platform(),
        "thread_environment": thread_environment(),
        "config": {
            "pairings": list(args.pairings),
            "N": args.N,
            "q_tasks": args.q_tasks,
            "workers": args.workers,
            "matsubara_indices": list(args.matsubara_indices),
            "canonical_block_size": args.canonical_block_size,
            "runtime_chunk_sizes": list(args.runtime_chunk_sizes),
            "minimum_speedup": args.minimum_speedup,
            "minimum_cpu_wall_ratio": args.minimum_cpu_wall_ratio,
            "maximum_pool_overhead_fraction": args.maximum_pool_overhead_fraction,
        },
        "pairings": pairing_rows,
        "arbitrary_q_performance_contract": "preflight_passed" if passed else "preflight_failed",
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": bool(passed),
    }
    _atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit("arbitrary-q performance preflight failed")


if __name__ == "__main__":
    main()
