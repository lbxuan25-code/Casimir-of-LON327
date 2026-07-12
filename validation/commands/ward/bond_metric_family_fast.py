"""Parallel resumable commensurate d-wave bond-metric Ward family.

Each point still delegates to the canonical ``bond-metric-full-kernel`` command,
so the complete 48-component primitive contract and output schema stay defined in
one place.  Independent points may run concurrently while each worker keeps BLAS
single-threaded to avoid nested oversubscription.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from validation.commands.ward.bond_metric_family import (
    DEFAULT_OUTPUT,
    DEFAULT_POINTS,
    _load_point_payload,
    _point_output,
    _summary,
    _write_csv,
    c4_comparisons,
    parse_integer_point,
    strict_gate_from_point_payload,
)


def _point_command(
    args: argparse.Namespace,
    point_output: Path,
    mx: int,
    my: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "validation",
        "ward",
        "bond-metric-full-kernel",
        "--nk",
        str(args.nk),
        "--mx",
        str(mx),
        "--my",
        str(my),
        "--shift-x",
        str(args.shift_x),
        "--shift-y",
        str(args.shift_y),
        "--subgrid-average",
        "auto",
        "--chunk-size",
        str(args.chunk_size),
        "--max-points",
        str(args.max_points),
        "--temperature-K",
        str(args.temperature_K),
        "--delta0-eV",
        str(args.delta0_eV),
        "--eta-eV",
        str(args.eta_eV),
        "--condition-max",
        str(args.condition_max),
        "--output",
        str(point_output),
    ]


def _run_point(command: list[str]) -> str:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(name, "1")
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return completed.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=628)
    parser.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--primitive-tolerance", type=float, default=1e-9)
    parser.add_argument("--amplitude-tolerance", type=float, default=1e-9)
    parser.add_argument("--phase-tolerance", type=float, default=1e-9)
    parser.add_argument("--effective-direct-tolerance", type=float, default=1e-9)
    parser.add_argument("--effective-residual-tolerance", type=float, default=1e-9)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-9)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--c4-tolerance", type=float, default=1e-8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.nk <= 0 or args.chunk_size <= 0 or args.max_points <= 0:
        parser.error("--nk, --chunk-size, and --max-points must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    points: list[tuple[int, int]] = []
    try:
        for item in args.points:
            point = parse_integer_point(item)
            if point not in points:
                points.append(point)
    except ValueError as exc:
        parser.error(str(exc))
    args.parsed_points = tuple(points)

    for name in (
        "primitive_tolerance",
        "amplitude_tolerance",
        "phase_tolerance",
        "effective_direct_tolerance",
        "effective_residual_tolerance",
        "longitudinal_tolerance",
        "c4_tolerance",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not np.isfinite(args.condition_max) or args.condition_max <= 0.0:
        parser.error("--condition-max must be finite and positive")
    return args


def _row_and_gate_payloads(
    payload: dict[str, Any],
    json_path: Path,
    mx: int,
    my: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_gate = strict_gate_from_point_payload(
        payload,
        "baseline",
        primitive_tolerance=args.primitive_tolerance,
        amplitude_tolerance=args.amplitude_tolerance,
        phase_tolerance=args.phase_tolerance,
        effective_direct_tolerance=args.effective_direct_tolerance,
        effective_residual_tolerance=args.effective_residual_tolerance,
        longitudinal_tolerance=args.longitudinal_tolerance,
        condition_max=args.condition_max,
    )
    corrected_gate = strict_gate_from_point_payload(
        payload,
        "corrected",
        primitive_tolerance=args.primitive_tolerance,
        amplitude_tolerance=args.amplitude_tolerance,
        phase_tolerance=args.phase_tolerance,
        effective_direct_tolerance=args.effective_direct_tolerance,
        effective_residual_tolerance=args.effective_residual_tolerance,
        longitudinal_tolerance=args.longitudinal_tolerance,
        condition_max=args.condition_max,
    )
    row = dict(payload["row"])
    row.update(
        {
            "baseline_effective_residual_over_q": baseline_gate[
                "effective_residual_over_q"
            ],
            "corrected_effective_residual_over_q": corrected_gate[
                "effective_residual_over_q"
            ],
            "baseline_strict_gate_passed": bool(baseline_gate["passed"]),
            "corrected_strict_gate_passed": bool(corrected_gate["passed"]),
            "strict_gate_criterion": "strict_static_q_normalized_v1",
            "point_json": str(json_path),
        }
    )
    gate_payload = {
        "point": (mx, my),
        "source": str(json_path),
        "baseline_strict_gate": baseline_gate,
        "corrected_strict_gate": corrected_gate,
    }
    return row, gate_payload


def main() -> None:
    args = _parse_args()
    points = list(args.parsed_points)
    point_root = args.output.parent / (args.output.stem + "_points")
    point_root.mkdir(parents=True, exist_ok=True)

    pending: dict[tuple[int, int], tuple[Path, list[str]]] = {}
    for index, (mx, my) in enumerate(points, start=1):
        point_output = _point_output(point_root, args.nk, mx, my)
        json_path = point_output.with_suffix(".json")
        if args.resume and point_output.is_file() and json_path.is_file():
            print(f"[{index}/{len(points)}] reusing {json_path}", flush=True)
        else:
            command = _point_command(args, point_output, mx, my)
            pending[(mx, my)] = (point_output, command)

    if pending:
        worker_count = min(int(args.workers), len(pending))
        if worker_count == 1:
            for index, ((mx, my), (_, command)) in enumerate(
                pending.items(), start=1
            ):
                print(
                    f"[{index}/{len(pending)}] running m=({mx},{my})",
                    flush=True,
                )
                output_text = _run_point(command)
                if output_text:
                    print(output_text, end="")
        else:
            print(
                f"running {len(pending)} independent point(s) with "
                f"{worker_count} workers",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_run_point, command): (mx, my)
                    for (mx, my), (_, command) in pending.items()
                }
                completed_count = 0
                for future in as_completed(futures):
                    mx, my = futures[future]
                    try:
                        output_text = future.result()
                    except subprocess.CalledProcessError as exc:
                        if exc.stdout:
                            print(exc.stdout, end="", flush=True)
                        raise
                    completed_count += 1
                    print(
                        f"[{completed_count}/{len(pending)}] completed "
                        f"m=({mx},{my})",
                        flush=True,
                    )
                    if output_text:
                        print(output_text, end="")

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for mx, my in points:
        point_output = _point_output(point_root, args.nk, mx, my)
        json_path = point_output.with_suffix(".json")
        payload = _load_point_payload(point_output, args.nk, mx, my)
        row, gate_payload = _row_and_gate_payloads(
            payload, json_path, mx, my, args
        )
        rows.append(row)
        payloads.append(gate_payload)
        _write_csv(args.output, rows)

    comparisons = c4_comparisons(rows, args.c4_tolerance)
    family_passed = bool(
        all(bool(row["corrected_strict_gate_passed"]) for row in rows)
        and all(bool(item["passed"]) for item in comparisons)
    )
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "parsed_points"
    }
    metadata = {
        "schema": "dwave_bond_metric_static_family_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "execution": {
            "requested_workers": int(args.workers),
            "effective_workers": min(int(args.workers), max(len(pending), 1)),
            "point_subprocesses": True,
            "blas_threads_per_worker": 1,
            "resume_enabled": bool(args.resume),
        },
        "rows": rows,
        "point_gates": payloads,
        "c4_comparisons": comparisons,
        "family_gate": {
            "all_corrected_strict_points_passed": all(
                bool(row["corrected_strict_gate_passed"]) for row in rows
            ),
            "all_available_c4_comparisons_passed": all(
                bool(item["passed"]) for item in comparisons
            ),
            "passed": family_passed,
        },
        "status": {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary = _summary(rows, comparisons)
    args.output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"CSV:     {args.output}")
    print(f"JSON:    {args.output.with_suffix('.json')}")
    print(f"summary: {args.output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
