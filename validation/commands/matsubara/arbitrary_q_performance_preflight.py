"""Formal architecture and real-hardware preflight for arbitrary-q periodic BZ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_formal_policy import (
    EXECUTION_STRATEGY,
    FORMAL_POLICY_ID,
    THREAD_POLICY_ID,
    config_fingerprint,
    validate_performance_formal_config,
)
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.workflows.arbitrary_q_matsubara import (
    integrate_arbitrary_q_periodic_bz,
)
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


def _formal_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pairings": list(args.pairings),
        "N": int(args.N),
        "q_tasks": int(args.q_tasks),
        "workers": int(args.workers),
        "matsubara_indices": list(args.matsubara_indices),
        "canonical_block_size": int(args.canonical_block_size),
        "runtime_chunk_sizes": list(args.runtime_chunk_sizes),
        "minimum_speedup": float(args.minimum_speedup),
        "minimum_cpu_wall_ratio": float(args.minimum_cpu_wall_ratio),
        "maximum_pool_overhead_fraction": float(
            args.maximum_pool_overhead_fraction
        ),
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairings",
        nargs="+",
        choices=("spm", "dwave"),
        default=["spm", "dwave"],
    )
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--q-tasks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--matsubara-indices",
        nargs="+",
        type=int,
        default=[0, 1, 2, 4, 8],
    )
    parser.add_argument("--canonical-block-size", type=int, default=4096)
    parser.add_argument(
        "--runtime-chunk-sizes",
        nargs="+",
        type=int,
        default=[4096, 16384],
    )
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--minimum-speedup", type=float, default=4.0)
    parser.add_argument("--minimum-cpu-wall-ratio", type=float, default=4.0)
    parser.add_argument("--maximum-pool-overhead-fraction", type=float, default=0.05)
    parser.add_argument("--comparison-rtol", type=float, default=2e-11)
    parser.add_argument("--comparison-atol", type=float, default=2e-12)
    parser.add_argument(
        "--diagnostic-nonformal",
        action="store_true",
        help="allow a reduced diagnostic run that can never establish formal passed",
    )
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
    formal = validate_performance_formal_config(_formal_config(args))
    if not formal.passed and not args.diagnostic_nonformal:
        parser.error(
            "configuration is looser than ArbitraryQFormalPolicyV1: "
            + "; ".join(formal.violations)
        )
    args.formal_policy = formal
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


def _response_signature(response: object) -> np.ndarray:
    values: list[np.ndarray] = [
        np.asarray(response.packed_primitives, dtype=complex).reshape(-1)
    ]
    for component, rhs in zip(response.components, response.rhs, strict=True):
        for field in (
            "bare_bubble",
            "direct",
            "bare_total",
            "collective_bubble",
            "collective_counterterm",
            "em_collective_left",
            "collective_em_right",
            "gauge_restored",
        ):
            values.append(
                np.asarray(getattr(component, field), dtype=complex).reshape(-1)
            )
        values.append(np.asarray(rhs.left, dtype=complex).reshape(-1))
        values.append(np.asarray(rhs.right, dtype=complex).reshape(-1))
    values.append(
        np.asarray(
            [
                response.operator_ward.max_absolute_error,
                response.operator_ward.max_relative_error,
                response.operator_ward.max_mixed_ratio,
            ],
            dtype=complex,
        )
    )
    return np.concatenate(values)


def _signature(task_result: object) -> np.ndarray:
    values = [_response_signature(task_result.result.plate_1)]
    values.extend(_response_signature(item) for item in task_result.result.plate_2)
    return np.concatenate(values)


def _mixed_comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    relative = absolute / max(scale, np.finfo(float).tiny)
    return {
        "absolute": absolute,
        "relative": relative,
        "scale": scale,
        "threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _threadpool_runtime() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info  # type: ignore
    except ImportError:
        return [{"error": "threadpoolctl not installed", "num_threads": -1}]
    return [dict(item) for item in threadpool_info()]


def _threadpool_passed(rows: Sequence[dict[str, Any]]) -> bool:
    return bool(rows) and all(int(row.get("num_threads", -1)) == 1 for row in rows)


def _hardware_record(threadpools: Sequence[dict[str, Any]]) -> dict[str, Any]:
    record = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "threadpools": list(threadpools),
    }
    encoded = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    record["hardware_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return record


def _architecture_rows(results: Sequence[object]) -> dict[str, Any]:
    profiles = []
    operator_passed = True
    cache_hits = 0
    payload_bytes = 0
    rss = 0
    pss = 0
    for task in results:
        for response in (task.result.plate_1, *task.result.plate_2):
            profiles.append(response.profile)
            operator_passed = operator_passed and response.operator_ward.passed
        cache_hits += int(task.result.response_cache_metadata["hits"])
        payload_bytes += int(task.payload_bytes)
        rss = max(rss, int(task.worker_rss_bytes))
        pss = max(pss, int(task.worker_pss_bytes))
    frequency_counts = {profile.frequency_count for profile in profiles}
    counterterm_counts = {profile.counterterm_add_count for profile in profiles}
    shifted_per_block = {
        profile.shifted_eigensystem_build_count
        // max(profile.canonical_block_count, 1)
        for profile in profiles
    }
    return {
        "response_count": len(profiles),
        "operator_ward_all_passed": bool(operator_passed),
        "frequency_counts": sorted(frequency_counts),
        "counterterm_add_counts": sorted(counterterm_counts),
        "actual_eigh_calls_per_canonical_block": sorted(shifted_per_block),
        "exact_response_cache_hits": cache_hits,
        "serialized_worker_payload_bytes": payload_bytes,
        "max_worker_rss_bytes": rss,
        "max_worker_pss_bytes": pss,
        "passed": bool(
            operator_passed
            and counterterm_counts == {1}
            and shifted_per_block.issubset({0, 2})
            and cache_hits >= len(results)
            and payload_bytes > 0
            and rss > 0
        ),
    }


def _frequency_count_audit(
    *,
    args: argparse.Namespace,
    material: object,
    model: object,
    ansatz: object,
    pairing: object,
    q: np.ndarray,
    runtime_chunk: int,
) -> dict[str, Any]:
    full_xi = _xi_values(args.matsubara_indices, args.temperature_K)
    short_indices = (0, min(v for v in args.matsubara_indices if v > 0))
    short_xi = _xi_values(short_indices, args.temperature_K)
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        q_model=q,
        n=material.grid.n,
        shift=material.grid.shift,
        canonical_reduction_block_size=args.canonical_block_size,
        runtime_chunk_size=runtime_chunk,
        material_cache=material,
        response_cache=None,
    )
    short = integrate_arbitrary_q_periodic_bz(
        xi_eV_values=short_xi,
        **common,
    )
    full = integrate_arbitrary_q_periodic_bz(
        xi_eV_values=full_xi,
        **common,
    )
    return {
        "short_frequency_count": int(short_xi.size),
        "full_frequency_count": int(full_xi.size),
        "short_actual_eigh_calls": int(
            short.profile.shifted_eigensystem_build_count
        ),
        "full_actual_eigh_calls": int(
            full.profile.shifted_eigensystem_build_count
        ),
        "passed": bool(
            short.profile.shifted_eigensystem_build_count
            == full.profile.shifted_eigensystem_build_count
        ),
    }


def _cache_audit(
    *,
    args: argparse.Namespace,
    material: object,
    model: object,
    ansatz: object,
    pairing: object,
    q: np.ndarray,
    runtime_chunk: int,
) -> dict[str, Any]:
    xi = _xi_values(args.matsubara_indices, args.temperature_K)
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        q_model=q,
        n=material.grid.n,
        shift=material.grid.shift,
        canonical_reduction_block_size=args.canonical_block_size,
        runtime_chunk_size=runtime_chunk,
        response_cache=None,
    )
    started = perf_counter()
    cached = integrate_arbitrary_q_periodic_bz(
        material_cache=material,
        **common,
    )
    cache_on_wall = float(perf_counter() - started)
    started = perf_counter()
    uncached = integrate_arbitrary_q_periodic_bz(
        material_cache=None,
        **common,
    )
    cache_off_wall = float(perf_counter() - started)
    comparison = _mixed_comparison(
        cached.packed_primitives,
        uncached.packed_primitives,
        atol=args.comparison_atol,
        rtol=args.comparison_rtol,
    )
    return {
        "cache_on_wall_seconds": cache_on_wall,
        "cache_off_wall_seconds": cache_off_wall,
        "cache_speedup": cache_off_wall
        / max(cache_on_wall, np.finfo(float).tiny),
        "packed_comparison": comparison,
        "passed": bool(comparison["passed"] and cache_on_wall < cache_off_wall),
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
        with ArbitraryQParallelEvaluator(
            process_workers=args.workers,
            **common,
        ) as parallel:
            parallel_results = parallel.evaluate(tasks)
            parallel_metadata = parallel.metadata()
        parallel_wall = float(perf_counter() - started)

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
                for left, right in zip(
                    baseline_signatures,
                    signatures,
                    strict=True,
                )
            ]

        speedup = serial_wall / max(parallel_wall, np.finfo(float).tiny)
        worker_seconds = sum(item.worker_seconds for item in parallel_results)
        cpu_wall = worker_seconds / max(parallel_wall, np.finfo(float).tiny)
        pool_overhead = (
            float(parallel_metadata["pool_startup_seconds"])
            + float(parallel_metadata["pool_shutdown_seconds"])
        ) / max(parallel_wall, np.finfo(float).tiny)
        architecture = _architecture_rows(parallel_results)
        frequency_audit = _frequency_count_audit(
            args=args,
            material=material,
            model=model,
            ansatz=ansatz,
            pairing=pairing,
            q=tasks[0].q_lab,
            runtime_chunk=runtime_chunk,
        )
        cache_audit = _cache_audit(
            args=args,
            material=material,
            model=model,
            ansatz=ansatz,
            pairing=pairing,
            q=tasks[0].q_lab,
            runtime_chunk=runtime_chunk,
        )
        passed = bool(
            architecture["passed"]
            and frequency_audit["passed"]
            and cache_audit["passed"]
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
                "frequency_count_audit": frequency_audit,
                "cache_on_off_audit": cache_audit,
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    threadpools = _threadpool_runtime()
    threadpool_passed = _threadpool_passed(threadpools)
    pairing_rows = [_run_pairing(args, name) for name in args.pairings]
    metric_passed = all(row["passed"] for row in pairing_rows) and threadpool_passed
    formal_passed = bool(args.formal_policy.passed and metric_passed)
    formal_config = _formal_config(args)
    command_values = list(sys.argv if argv is None else argv)
    payload = {
        "schema": "arbitrary-q-performance-preflight-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "exact_command": shlex.join(str(value) for value in command_values),
        "hardware": _hardware_record(threadpools),
        "thread_environment": thread_environment(),
        "thread_policy_id": THREAD_POLICY_ID,
        "actual_threadpools": threadpools,
        "actual_threadpool_passed": threadpool_passed,
        "execution_strategy": EXECUTION_STRATEGY,
        "config": formal_config,
        "config_fingerprint": config_fingerprint(formal_config),
        **args.formal_policy.as_dict(),
        "pairings": pairing_rows,
        "metric_passed": bool(metric_passed),
        "arbitrary_q_performance_contract": (
            "formal_preflight_passed"
            if formal_passed
            else (
                "diagnostic_preflight_passed_not_formal"
                if metric_passed
                else "preflight_failed"
            )
        ),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": bool(formal_passed),
    }
    _atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": formal_passed}, indent=2))
    if not formal_passed:
        raise SystemExit(
            f"arbitrary-q performance preflight did not establish {FORMAL_POLICY_ID}"
        )


if __name__ == "__main__":
    main()
