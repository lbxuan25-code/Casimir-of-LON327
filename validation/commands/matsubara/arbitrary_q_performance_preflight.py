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
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_formal_policy import (
    EXECUTION_STRATEGY,
    FORMAL_POLICY_ID,
    MODEL_WORKLOAD_ID,
    OUTER_Q_BATCH_WORKLOAD_ID,
    PERFORMANCE_WORKLOAD_ID,
    QUALIFICATION_AUDIT_WORKLOAD_ID,
    QUALIFICATION_PRIMARY_WORKLOAD_ID,
    THREAD_POLICY_ID,
    config_fingerprint,
    validate_performance_formal_config,
)
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.workflows.arbitrary_q_matsubara import integrate_arbitrary_q_periodic_bz
from lno327.workflows.arbitrary_q_parallel import (
    ArbitraryQParallelEvaluator,
    QLabAngleTask,
    thread_environment,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.source_tree_provenance import source_tree_provenance

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_performance_preflight/preflight.json"
)


def _formal_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "performance_workload_id": PERFORMANCE_WORKLOAD_ID,
        "model_workload_id": MODEL_WORKLOAD_ID,
        "workload_classes": [
            OUTER_Q_BATCH_WORKLOAD_ID,
            QUALIFICATION_PRIMARY_WORKLOAD_ID,
            QUALIFICATION_AUDIT_WORKLOAD_ID,
        ],
        "pairings": list(args.pairings),
        "N": int(args.N),
        "q_tasks": int(args.q_tasks),
        "workers": int(args.workers),
        "qualification_primary_tasks": 4,
        "qualification_primary_workers": 4,
        "qualification_audit_tasks": 1,
        "qualification_audit_workers": 1,
        "matsubara_indices": list(args.matsubara_indices),
        "canonical_block_size": int(args.canonical_block_size),
        "runtime_chunk_sizes": list(args.runtime_chunk_sizes),
        "minimum_speedup": float(args.minimum_speedup),
        "minimum_cpu_wall_ratio": float(args.minimum_cpu_wall_ratio),
        "maximum_pool_overhead_fraction": float(args.maximum_pool_overhead_fraction),
        "comparison_rtol": float(args.comparison_rtol),
        "comparison_atol": float(args.comparison_atol),
        "temperature_K": float(args.temperature_K),
        "delta0_eV": float(args.delta0_eV),
        "eta_eV": float(args.eta_eV),
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


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
    parser.add_argument("--diagnostic-nonformal", action="store_true")
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
    args.matsubara_indices = tuple(sorted(set(int(v) for v in args.matsubara_indices)))
    if 0 not in args.matsubara_indices or not any(v > 0 for v in args.matsubara_indices):
        parser.error("preflight requires exact zero and at least one positive index")
    args.pairings = tuple(dict.fromkeys(args.pairings))
    formal = validate_performance_formal_config(_formal_config(args))
    if not formal.passed and not args.diagnostic_nonformal:
        parser.error("configuration is looser than the frozen formal policy: " + "; ".join(formal.violations))
    args.formal_policy = formal
    return args


def _xi_values(indices: Sequence[int], temperature_K: float) -> np.ndarray:
    return np.asarray(
        [0.0 if index == 0 else matsubara_energy_eV(index, temperature_K) for index in indices],
        dtype=float,
    )


def _base_q() -> np.ndarray:
    return (2.0 * np.pi / 1256.0) * np.asarray([6.0, 4.0])


def _outer_tasks(count: int) -> tuple[QLabAngleTask, ...]:
    base = _base_q()
    tasks = []
    for index in range(int(count)):
        angle = np.deg2rad(3.0 + 11.0 * index)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        tasks.append(
            QLabAngleTask(
                index=index,
                q_lab=(0.85 + 0.03 * index) * (rotation @ base),
                theta_1_rad=0.0,
                theta_2_rad_values=np.asarray([0.0, np.deg2rad(17.0)]),
            )
        )
    return tuple(tasks)


def _qualification_primary_tasks() -> tuple[QLabAngleTask, ...]:
    factor = 2.0 * np.pi / 1256.0
    return (
        QLabAngleTask(0, factor * np.asarray([1.0, 0.0]), 0.0, np.asarray([0.0])),
        QLabAngleTask(1, factor * np.asarray([6.0, 4.0]), 0.0, np.asarray([0.0, np.deg2rad(17.0)])),
        QLabAngleTask(2, factor * np.asarray([25.0, 24.0]), 0.0, np.asarray([0.0])),
        QLabAngleTask(3, factor * np.asarray([6.0, 6.0]), 0.0, np.asarray([0.0])),
    )


def _qualification_audit_tasks() -> tuple[QLabAngleTask, ...]:
    return (
        QLabAngleTask(0, _base_q(), 0.0, np.asarray([np.deg2rad(17.0)])),
    )


def _response_signature(response: object) -> np.ndarray:
    values = [np.asarray(response.packed_primitives, dtype=complex).reshape(-1)]
    for component, rhs in zip(response.components, response.rhs, strict=True):
        for field in (
            "bare_bubble", "direct", "bare_total", "collective_bubble",
            "collective_counterterm", "em_collective_left",
            "collective_em_right", "gauge_restored",
        ):
            values.append(np.asarray(getattr(component, field), dtype=complex).reshape(-1))
        values.append(np.asarray(rhs.left, dtype=complex).reshape(-1))
        values.append(np.asarray(rhs.right, dtype=complex).reshape(-1))
    return np.concatenate(values)


def _signature(task_result: object) -> np.ndarray:
    return np.concatenate(
        [_response_signature(task_result.result.plate_1)]
        + [_response_signature(item) for item in task_result.result.plate_2]
    )


def _mixed(left: np.ndarray, right: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    a, b = np.asarray(left, dtype=complex), np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
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
    record["hardware_fingerprint"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return record


def _architecture(results: Sequence[object]) -> dict[str, Any]:
    profiles = [response.profile for task in results for response in (task.result.plate_1, *task.result.plate_2)]
    operator_passed = all(
        response.operator_ward.passed
        for task in results
        for response in (task.result.plate_1, *task.result.plate_2)
    )
    cache_hits = sum(int(task.result.response_cache_metadata["hits"]) for task in results)
    payload_bytes = sum(int(task.payload_bytes) for task in results)
    rss = max((int(task.worker_rss_bytes) for task in results), default=0)
    pss = max((int(task.worker_pss_bytes) for task in results), default=0)
    runtime_build_pairs = {
        (int(profile.runtime_chunk_count), int(profile.q_workspace_build_count))
        for profile in profiles
    }
    shifted_per_runtime = {
        int(profile.shifted_eigensystem_build_count) // max(int(profile.runtime_chunk_count), 1)
        for profile in profiles
    }
    return {
        "response_count": len(profiles),
        "operator_ward_all_passed": bool(operator_passed),
        "counterterm_add_counts": sorted({int(p.counterterm_add_count) for p in profiles}),
        "runtime_chunk_q_workspace_counts": [list(v) for v in sorted(runtime_build_pairs)],
        "actual_eigh_calls_per_runtime_chunk": sorted(shifted_per_runtime),
        "exact_response_cache_hits": cache_hits,
        "serialized_worker_payload_bytes": payload_bytes,
        "max_worker_rss_bytes": rss,
        "max_worker_pss_bytes": pss,
        "passed": bool(
            operator_passed
            and all(a == b for a, b in runtime_build_pairs)
            and shifted_per_runtime.issubset({0, 2})
            and {int(p.counterterm_add_count) for p in profiles} == {1}
            and cache_hits >= len(results)
            and payload_bytes > 0
            and rss > 0
        ),
    }


def _run_evaluator(
    *, tasks: Sequence[QLabAngleTask], workers: int, common: dict[str, Any]
) -> tuple[tuple[object, ...], float, dict[str, Any]]:
    evaluator = ArbitraryQParallelEvaluator(process_workers=int(workers), **common)
    started = perf_counter()
    try:
        results = evaluator.evaluate(tasks)
    finally:
        evaluator.close()
    wall = float(perf_counter() - started)
    metadata = evaluator.metadata()
    return results, wall, metadata


def _workload_record(
    *,
    workload_id: str,
    tasks: Sequence[QLabAngleTask],
    workers: int,
    common: dict[str, Any],
    args: argparse.Namespace,
    require_outer_thresholds: bool,
) -> dict[str, Any]:
    serial, serial_wall, serial_meta = _run_evaluator(tasks=tasks, workers=1, common=common)
    if workers == 1:
        parallel, parallel_wall, parallel_meta = serial, serial_wall, serial_meta
    else:
        parallel, parallel_wall, parallel_meta = _run_evaluator(tasks=tasks, workers=workers, common=common)
    comparisons = [
        _mixed(_signature(a), _signature(b), atol=args.comparison_atol, rtol=args.comparison_rtol)
        for a, b in zip(serial, parallel, strict=True)
    ]
    worker_seconds = sum(float(item.worker_seconds) for item in parallel)
    speedup = serial_wall / max(parallel_wall, np.finfo(float).tiny)
    cpu_wall = worker_seconds / max(parallel_wall, np.finfo(float).tiny)
    pool_overhead = (
        float(parallel_meta["pool_startup_seconds"])
        + float(parallel_meta["pool_shutdown_seconds"])
    ) / max(parallel_wall, np.finfo(float).tiny)
    architecture = _architecture(parallel)
    thresholds = True
    if require_outer_thresholds:
        thresholds = bool(
            speedup >= args.minimum_speedup
            and cpu_wall >= args.minimum_cpu_wall_ratio
            and pool_overhead <= args.maximum_pool_overhead_fraction
        )
    elif workers > 1:
        thresholds = bool(speedup >= 1.25 and cpu_wall >= 1.5 and pool_overhead <= 0.10)
    passed = bool(architecture["passed"] and all(row["passed"] for row in comparisons) and thresholds)
    return {
        "workload_id": workload_id,
        "task_count": len(tasks),
        "workers": int(workers),
        "serial_wall_seconds": serial_wall,
        "parallel_wall_seconds": parallel_wall,
        "speedup": speedup,
        "summed_worker_seconds": worker_seconds,
        "cpu_wall_ratio": cpu_wall,
        "pool_overhead_fraction": pool_overhead,
        "pool_shutdown_measured_after_close": bool(float(parallel_meta["pool_shutdown_seconds"]) >= 0.0),
        "serial_process_comparisons": comparisons,
        "architecture": architecture,
        "serial_metadata": serial_meta,
        "parallel_metadata": parallel_meta,
        "passed": passed,
    }


def _frequency_audit(args: argparse.Namespace, material: object, model: object, ansatz: object, pairing: object, q: np.ndarray, runtime_chunk: int) -> dict[str, Any]:
    positive = min(v for v in args.matsubara_indices if v > 0)
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
    short = integrate_arbitrary_q_periodic_bz(xi_eV_values=_xi_values((0, positive), args.temperature_K), **common)
    full = integrate_arbitrary_q_periodic_bz(xi_eV_values=_xi_values(args.matsubara_indices, args.temperature_K), **common)
    return {
        "short_actual_eigh_calls": int(short.profile.shifted_eigensystem_build_count),
        "full_actual_eigh_calls": int(full.profile.shifted_eigensystem_build_count),
        "short_q_workspace_builds": int(short.profile.q_workspace_build_count),
        "full_q_workspace_builds": int(full.profile.q_workspace_build_count),
        "passed": bool(
            short.profile.shifted_eigensystem_build_count == full.profile.shifted_eigensystem_build_count
            and short.profile.q_workspace_build_count == full.profile.q_workspace_build_count
        ),
    }


def _cache_audit(args: argparse.Namespace, material: object, model: object, ansatz: object, pairing: object, q: np.ndarray, runtime_chunk: int) -> dict[str, Any]:
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=_xi_values(args.matsubara_indices, args.temperature_K),
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
    cached = integrate_arbitrary_q_periodic_bz(material_cache=material, **common)
    cache_on = float(perf_counter() - started)
    started = perf_counter()
    uncached = integrate_arbitrary_q_periodic_bz(material_cache=None, **common)
    cache_off = float(perf_counter() - started)
    comparison = _mixed(cached.packed_primitives, uncached.packed_primitives, atol=args.comparison_atol, rtol=args.comparison_rtol)
    return {
        "cache_on_wall_seconds": cache_on,
        "cache_off_wall_seconds": cache_off,
        "cache_speedup": cache_off / max(cache_on, np.finfo(float).tiny),
        "packed_comparison": comparison,
        "passed": bool(comparison["passed"] and cache_on < cache_off),
    }


def _run_pairing(args: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    xi = _xi_values(args.matsubara_indices, args.temperature_K)
    grid = build_periodic_bz_grid(args.N, (0.5, 0.5))
    material = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=KuboConfig.from_kelvin(omega_eV=float(xi[0]), temperature_K=args.temperature_K, eta_eV=args.eta_eV, output_si=False),
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
        grid=grid,
    )
    common_base = dict(
        material_cache=material,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        canonical_reduction_block_size=args.canonical_block_size,
    )
    records = []
    baseline = None
    all_passed = True
    for runtime_chunk in args.runtime_chunk_sizes:
        common = {**common_base, "runtime_chunk_size": int(runtime_chunk)}
        workloads = [
            _workload_record(
                workload_id=OUTER_Q_BATCH_WORKLOAD_ID,
                tasks=_outer_tasks(args.q_tasks),
                workers=args.workers,
                common=common,
                args=args,
                require_outer_thresholds=True,
            ),
            _workload_record(
                workload_id=QUALIFICATION_PRIMARY_WORKLOAD_ID,
                tasks=_qualification_primary_tasks(),
                workers=4,
                common=common,
                args=args,
                require_outer_thresholds=False,
            ),
            _workload_record(
                workload_id=QUALIFICATION_AUDIT_WORKLOAD_ID,
                tasks=_qualification_audit_tasks(),
                workers=1,
                common=common,
                args=args,
                require_outer_thresholds=False,
            ),
        ]
        signatures = tuple(_signature(item) for item in _run_evaluator(tasks=_qualification_primary_tasks(), workers=1, common=common)[0])
        chunk_comparisons = [] if baseline is None else [
            _mixed(a, b, atol=args.comparison_atol, rtol=args.comparison_rtol)
            for a, b in zip(baseline, signatures, strict=True)
        ]
        if baseline is None:
            baseline = signatures
        frequency = _frequency_audit(args, material, model, ansatz, pairing, _base_q(), runtime_chunk)
        cache = _cache_audit(args, material, model, ansatz, pairing, _base_q(), runtime_chunk)
        passed = bool(
            all(row["passed"] for row in workloads)
            and all(row["passed"] for row in chunk_comparisons)
            and frequency["passed"]
            and cache["passed"]
        )
        all_passed = all_passed and passed
        records.append({
            "runtime_chunk_size": int(runtime_chunk),
            "workloads": workloads,
            "chunk_size_comparisons": chunk_comparisons,
            "frequency_count_audit": frequency,
            "cache_on_off_audit": cache,
            "passed": passed,
        })
    return {
        "pairing": pairing_name,
        "material_cache": material.metadata(),
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
    provenance = source_tree_provenance()
    if not provenance.worktree_clean and not args.diagnostic_nonformal:
        provenance.require_clean()
    threadpools = _threadpool_runtime()
    threadpool_passed = _threadpool_passed(threadpools)
    pairing_rows = [_run_pairing(args, name) for name in args.pairings]
    metric_passed = bool(all(row["passed"] for row in pairing_rows) and threadpool_passed)
    formal_passed = bool(
        args.formal_policy.passed
        and metric_passed
        and provenance.worktree_clean
        and not args.diagnostic_nonformal
    )
    formal_config = _formal_config(args)
    command_values = list(sys.argv if argv is None else argv)
    payload = {
        "schema": "arbitrary-q-performance-preflight-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **provenance.as_dict(),
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
        "metric_passed": metric_passed,
        "diagnostic_nonformal_requested": bool(args.diagnostic_nonformal),
        "arbitrary_q_performance_contract": (
            "formal_preflight_passed"
            if formal_passed
            else ("diagnostic_preflight_passed_not_formal" if metric_passed else "preflight_failed")
        ),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": formal_passed,
    }
    _atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": formal_passed}, indent=2))
    if not formal_passed:
        raise SystemExit(f"arbitrary-q performance preflight did not establish {FORMAL_POLICY_ID}")


if __name__ == "__main__":
    main()
