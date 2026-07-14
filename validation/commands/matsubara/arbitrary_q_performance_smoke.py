"""Small nonformal arbitrary-q performance-structure and timing-profile smoke test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.workflows.arbitrary_q_matsubara import integrate_arbitrary_q_periodic_bz
from lno327.workflows.arbitrary_q_parallel import ArbitraryQParallelEvaluator, QLabAngleTask
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.source_tree_provenance import source_tree_provenance

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_staged_flow/stage1_performance_smoke.json"
)
_TIMING_FIELDS = (
    "q_workspace_seconds",
    "kubo_factor_seconds",
    "kubo_contraction_seconds",
    "primitive_pack_seconds",
    "operator_ward_seconds",
    "accumulation_seconds",
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=["spm"])
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--q-tasks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--canonical-block-size", type=int, default=4096)
    parser.add_argument("--runtime-chunk-sizes", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--comparison-atol", type=float, default=2e-12)
    parser.add_argument("--comparison-rtol", type=float, default=2e-11)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.pairings = tuple(dict.fromkeys(args.pairings))
    args.matsubara_indices = tuple(sorted(set(int(v) for v in args.matsubara_indices)))
    args.runtime_chunk_sizes = tuple(dict.fromkeys(int(v) for v in args.runtime_chunk_sizes))
    if args.N <= 0 or args.N % 2:
        parser.error("--N must be positive and even")
    if args.N * args.N < max(args.runtime_chunk_sizes):
        parser.error("N^2 must be at least the largest runtime chunk for a meaningful smoke test")
    if args.workers <= 0 or args.q_tasks < args.workers:
        parser.error("require q_tasks >= workers >= 1")
    if 0 not in args.matsubara_indices or not any(v > 0 for v in args.matsubara_indices):
        parser.error("exact zero and at least one positive Matsubara index are required")
    if args.canonical_block_size <= 0 or args.canonical_block_size % 2:
        parser.error("canonical block size must be positive and even")
    for size in args.runtime_chunk_sizes:
        if size < args.canonical_block_size or size % args.canonical_block_size:
            parser.error("runtime chunks must be integer multiples of canonical block size")
    return args


def _xi(indices: Sequence[int], temperature_K: float) -> np.ndarray:
    return np.asarray(
        [0.0 if n == 0 else matsubara_energy_eV(int(n), temperature_K) for n in indices],
        dtype=float,
    )


def _base_q() -> np.ndarray:
    return (2.0 * np.pi / 1256.0) * np.asarray([6.0, 4.0])


def _tasks(count: int) -> tuple[QLabAngleTask, ...]:
    base = _base_q()
    rows: list[QLabAngleTask] = []
    for index in range(int(count)):
        angle = np.deg2rad(5.0 + 13.0 * index)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        rows.append(
            QLabAngleTask(
                index=index,
                q_lab=(0.9 + 0.025 * index) * (rotation @ base),
                theta_1_rad=0.0,
                theta_2_rad_values=np.asarray([0.0, np.deg2rad(17.0)]),
            )
        )
    return tuple(rows)


def _mixed(left: np.ndarray, right: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
        "threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _timing_breakdown(profiles: Iterable[object]) -> dict[str, Any]:
    values = tuple(profiles)
    seconds = {
        field: float(sum(float(getattr(profile, field)) for profile in values))
        for field in _TIMING_FIELDS
    }
    reported_total = float(sum(float(profile.total_seconds) for profile in values))
    attributed = float(sum(seconds.values()))
    unattributed = max(reported_total - attributed, 0.0)
    denominator = max(reported_total, np.finfo(float).tiny)
    fractions = {field: value / denominator for field, value in seconds.items()}
    fractions["unattributed_seconds"] = unattributed / denominator
    return {
        "response_profile_count": len(values),
        "reported_total_seconds": reported_total,
        "attributed_seconds": attributed,
        "unattributed_seconds": unattributed,
        "stage_seconds": seconds,
        "stage_fraction_of_reported_total": fractions,
    }


def _all_responses(results: Sequence[object]) -> tuple[object, ...]:
    values: list[object] = []
    seen: set[int] = set()
    for item in results:
        for response in (item.result.plate_1, *item.result.plate_2):
            identity = id(response)
            if identity not in seen:
                seen.add(identity)
                values.append(response)
    return tuple(values)


def _run_parallel(*, tasks: Sequence[QLabAngleTask], workers: int, common: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    evaluator = ArbitraryQParallelEvaluator(process_workers=int(workers), **common)
    try:
        results = evaluator.evaluate(tasks)
    finally:
        evaluator.close()
    end_to_end = float(perf_counter() - started)
    metadata = evaluator.metadata()
    responses = _all_responses(results)
    return {
        "end_to_end_seconds": end_to_end,
        "summed_worker_seconds": float(sum(float(item.worker_seconds) for item in results)),
        "metadata": metadata,
        "timing_breakdown": _timing_breakdown(response.profile for response in responses),
        "operator_identity_all_passed": all(response.operator_ward.passed for response in responses),
        "counterterm_add_counts": sorted({int(response.profile.counterterm_add_count) for response in responses}),
        "q_workspace_build_count": int(sum(response.profile.q_workspace_build_count for response in responses)),
        "shifted_eigh_call_count": int(sum(response.profile.shifted_eigensystem_build_count for response in responses)),
    }


def _run_pairing(args: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    xi_full = _xi(args.matsubara_indices, args.temperature_K)
    first_positive = min(v for v in args.matsubara_indices if v > 0)
    xi_short = _xi((0, first_positive), args.temperature_K)
    grid = build_periodic_bz_grid(args.N, (0.5, 0.5))
    cache = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=KuboConfig.from_kelvin(
            omega_eV=0.0,
            temperature_K=args.temperature_K,
            eta_eV=args.eta_eV,
            output_si=False,
        ),
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
        grid=grid,
    )
    base_common = dict(
        material_cache=cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        canonical_reduction_block_size=args.canonical_block_size,
    )
    baseline: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    all_structure = True
    for runtime in args.runtime_chunk_sizes:
        response = integrate_arbitrary_q_periodic_bz(
            xi_eV_values=xi_full,
            q_model=_base_q(),
            n=args.N,
            shift=(0.5, 0.5),
            runtime_chunk_size=runtime,
            **base_common,
        )
        short = integrate_arbitrary_q_periodic_bz(
            xi_eV_values=xi_short,
            q_model=_base_q(),
            n=args.N,
            shift=(0.5, 0.5),
            runtime_chunk_size=runtime,
            **base_common,
        )
        expected_chunks = int(np.ceil(grid.point_count / runtime))
        count_passed = bool(
            response.profile.runtime_chunk_count == expected_chunks
            and response.profile.q_workspace_build_count == expected_chunks
            and response.profile.shifted_eigensystem_build_count == 2 * expected_chunks
        )
        frequency_reuse = bool(
            short.profile.q_workspace_build_count == response.profile.q_workspace_build_count
            and short.profile.shifted_eigensystem_build_count
            == response.profile.shifted_eigensystem_build_count
        )
        comparison = None if baseline is None else _mixed(
            baseline,
            response.packed_primitives,
            atol=args.comparison_atol,
            rtol=args.comparison_rtol,
        )
        if baseline is None:
            baseline = np.asarray(response.packed_primitives, dtype=complex)
        parallel = _run_parallel(
            tasks=_tasks(args.q_tasks),
            workers=args.workers,
            common={
                **base_common,
                "xi_eV_values": xi_full,
                "runtime_chunk_size": runtime,
            },
        )
        structure = bool(
            count_passed
            and frequency_reuse
            and response.operator_ward.passed
            and response.profile.counterterm_add_count == 1
            and (comparison is None or comparison["passed"])
            and parallel["operator_identity_all_passed"]
            and parallel["counterterm_add_counts"] == [1]
        )
        all_structure = all_structure and structure
        rows.append(
            {
                "runtime_chunk_size": int(runtime),
                "expected_runtime_chunks": expected_chunks,
                "serial_profile": response.profile.as_dict(),
                "serial_timing_breakdown": _timing_breakdown((response.profile,)),
                "short_frequency_profile": short.profile.as_dict(),
                "frequency_batch_reuses_shifted_eigensystems": frequency_reuse,
                "runtime_count_structure_passed": count_passed,
                "packed_comparison_to_first_runtime": comparison,
                "parallel_batch": parallel,
                "structure_passed": structure,
            }
        )
    return {
        "pairing": pairing_name,
        "material_cache": cache.metadata(),
        "records": rows,
        "optimization_structure_passed": bool(all_structure),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _args(argv)
    rows = [_run_pairing(args, pairing) for pairing in args.pairings]
    passed = all(row["optimization_structure_passed"] for row in rows)
    payload = {
        "schema": "arbitrary-q-performance-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **source_tree_provenance().as_dict(),
        "config": {
            "pairings": list(args.pairings),
            "N": int(args.N),
            "q_tasks": int(args.q_tasks),
            "workers": int(args.workers),
            "matsubara_indices": list(args.matsubara_indices),
            "canonical_block_size": int(args.canonical_block_size),
            "runtime_chunk_sizes": list(args.runtime_chunk_sizes),
            "temperature_K": float(args.temperature_K),
            "delta0_eV": float(args.delta0_eV),
            "eta_eV": float(args.eta_eV),
        },
        "pairings": rows,
        "optimization_structure_passed": bool(passed),
        "formal_performance_evidence": False,
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": bool(passed),
    }
    _write(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit("arbitrary-q performance smoke found a structural optimization failure")


if __name__ == "__main__":
    main()
