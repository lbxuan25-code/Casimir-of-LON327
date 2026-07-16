"""Microscopic staged preflight for the outer-q Casimir integral.

The command builds one deduplicated union of all requested cutoff, radial-order,
angular-order, and angular-cut grids.  It then invokes the sole public universal
transverse-point sweet-spot command once, requires every microscopic node to be
physically and numerically certified, and reduces the canonical primary-shift
logdet values into finite Matsubara partial free energies.

Passing this command qualifies the requested finite pilot ladder only.  It does not
qualify an infinite Matsubara tail, torque differentiation, or a production result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from validation.lib.microscopic_outer_q_preflight import (
    aggregate_certified_outer_q,
    build_staged_grid_plan,
    build_union_node_manifest,
    compare_ladders,
)


DEFAULT_OUTPUT_ROOT = Path(
    "validation/outputs/casimir/microscopic_outer_q_preflight"
)
DEFAULT_SHIFTS = ((0.5, 0.5), (0.25, 0.75), (0.75, 0.25))
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-preflight", type=Path, required=True)
    parser.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=["spm"])
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--u-max-values", nargs="+", type=float, default=[6.0, 10.0])
    parser.add_argument("--radial-orders", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--angular-orders", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--angular-offsets", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument(
        "--N-candidates",
        nargs="+",
        type=int,
        default=[128, 192, 256],
    )
    parser.add_argument("--shift", action="append", nargs=2, type=float)
    parser.add_argument("--plate-angles-deg", nargs=2, type=float, default=[0.0, 17.0])
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "q", "context", "wave"),
        default="auto",
    )
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--max-context-workers", type=int, default=0)
    parser.add_argument("--memory-safety-factor", type=float, default=1.5)
    parser.add_argument("--fallback-context-bytes-per-point", type=float, default=16384.0)
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
    parser.add_argument("--logdet-atol", type=float, default=1e-6)
    parser.add_argument("--outer-rtol", type=float, default=5e-2)
    parser.add_argument("--outer-atol-J-m2", type=float, default=1e-10)
    parser.add_argument(
        "--require-ladder-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    args.pairings = tuple(dict.fromkeys(str(value) for value in args.pairings))
    args.matsubara_indices = tuple(sorted(set(int(value) for value in args.matsubara_indices)))
    args.N_candidates = tuple(int(value) for value in args.N_candidates)
    args.shifts = tuple(
        tuple(float(component) for component in value)
        for value in (args.shift or DEFAULT_SHIFTS)
    )
    if not args.matsubara_indices or any(value < 0 for value in args.matsubara_indices):
        parser.error("--matsubara-indices must be nonempty and non-negative")
    if len(args.N_candidates) < 3:
        parser.error("--N-candidates requires at least three levels")
    if (
        tuple(sorted(set(args.N_candidates))) != args.N_candidates
        or any(value <= 0 or value % 2 for value in args.N_candidates)
    ):
        parser.error("--N-candidates must be strictly increasing unique positive even values")
    if len(args.shifts) < 2 or len(set(args.shifts)) != len(args.shifts):
        parser.error("at least two unique shifts are required")
    if len(args.u_max_values) < 2 or len(args.radial_orders) < 2 or len(args.angular_orders) < 2:
        parser.error("cutoff, radial, and angular ladders each require at least two values")
    if len(args.angular_offsets) < 2:
        parser.error("angular cut audit requires at least two offsets")
    for name in (
        "temperature_K",
        "delta0_eV",
        "separation_nm",
        "outer_rtol",
        "outer_atol_J_m2",
        "logdet_rtol",
        "logdet_atol",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0 or (
            name in {"temperature_K", "delta0_eV", "separation_nm"} and value == 0.0
        ):
            parser.error(f"--{name.replace('_', '-')} has an invalid value")
    return args


def _validate_measure_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"outer-q measure preflight does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read outer-q measure preflight {path}: {exc}") from exc
    status = payload.get("status", {})
    if payload.get("schema") != "outer-q-quadrature-preflight-v1":
        raise SystemExit("outer-q measure preflight has the wrong schema")
    if not bool(status.get("passed")) or not bool(
        status.get("microscopic_outer_q_preflight_allowed")
    ):
        raise SystemExit("outer-q measure preflight did not authorize microscopic preflight")
    return payload


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _thread_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in _THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _sweet_spot_command(args: argparse.Namespace, manifest, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "validation",
        "diagnostic",
        "transverse-point-sweet-spot",
    ]
    for label, q in zip(manifest.labels, manifest.q_model, strict=True):
        command.extend(["--q-point", label, repr(float(q[0])), repr(float(q[1]))])
    command.extend(["--pairings", *args.pairings])
    command.extend(["--matsubara-indices", *[str(value) for value in args.matsubara_indices]])
    command.extend(["--N-candidates", *[str(value) for value in args.N_candidates]])
    for shift in args.shifts:
        command.extend(["--shift", repr(float(shift[0])), repr(float(shift[1]))])
    command.extend(
        [
            "--plate-angles-deg",
            repr(float(args.plate_angles_deg[0])),
            repr(float(args.plate_angles_deg[1])),
            "--required-consecutive-passes",
            str(int(args.required_consecutive_passes)),
            "--workers",
            str(int(args.workers)),
            "--parallel-mode",
            str(args.parallel_mode),
            "--memory-budget-gb",
            repr(float(args.memory_budget_gb)),
            "--max-context-workers",
            str(int(args.max_context_workers)),
            "--memory-safety-factor",
            repr(float(args.memory_safety_factor)),
            "--fallback-context-bytes-per-point",
            repr(float(args.fallback_context_bytes_per_point)),
            "--canonical-block",
            str(int(args.canonical_block)),
            "--runtime-chunk",
            str(int(args.runtime_chunk)),
            "--temperature-K",
            repr(float(args.temperature_K)),
            "--delta0-eV",
            repr(float(args.delta0_eV)),
            "--eta-eV",
            repr(float(args.eta_eV)),
            "--degeneracy",
            repr(float(args.degeneracy)),
            "--separation-nm",
            repr(float(args.separation_nm)),
            "--ward-tolerance",
            repr(float(args.ward_tolerance)),
            "--ward-absolute-tolerance",
            repr(float(args.ward_absolute_tolerance)),
            "--condition-max",
            repr(float(args.condition_max)),
            "--static-energy-scale-eV",
            repr(float(args.static_energy_scale_eV)),
            "--static-reality-tolerance",
            repr(float(args.static_reality_tolerance)),
            "--static-longitudinal-tolerance",
            repr(float(args.static_longitudinal_tolerance)),
            "--static-mixing-tolerance",
            repr(float(args.static_mixing_tolerance)),
            "--static-passivity-tolerance",
            repr(float(args.static_passivity_tolerance)),
            "--logdet-rtol",
            repr(float(args.logdet_rtol)),
            "--logdet-atol",
            repr(float(args.logdet_atol)),
            "--output",
            str(output),
        ]
    )
    return command


def _plan_payload(plan, manifest) -> dict[str, Any]:
    return {
        "reference_spec_id": plan.reference_spec_id,
        "reference_offset_fraction": plan.reference_offset_fraction,
        "ladders": {name: list(values) for name, values in plan.ladders.items()},
        "specs": [
            {
                "spec_id": spec.spec_id,
                "u_max": spec.u_max,
                "radial_order": spec.radial_order,
                "angular_order": spec.angular_order,
                "angular_offset_fraction": spec.angular_offset_fraction,
                "node_count": manifest.grids[spec.spec_id].node_count,
            }
            for spec in plan.specs
        ],
        "unique_microscopic_q_node_count": len(manifest.labels),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    measure_payload = _validate_measure_preflight(args.measure_preflight)
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    plan = build_staged_grid_plan(
        u_max_values=args.u_max_values,
        radial_orders=args.radial_orders,
        angular_orders=args.angular_orders,
        angular_offsets=args.angular_offsets,
    )
    manifest = build_union_node_manifest(
        plan,
        separation_m=float(args.separation_nm) * 1e-9,
        lattice_a_x_m=material.lattice_a_x_m,
        lattice_a_y_m=material.lattice_a_y_m,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    sweet_path = args.output_root / "transverse_sweet_spot.json"
    sweet_stdout = args.output_root / "transverse_sweet_spot.stdout.json"
    sweet_stderr = args.output_root / "transverse_sweet_spot.stderr.log"
    command = _sweet_spot_command(args, manifest, sweet_path)

    base = {
        "schema": "microscopic-outer-q-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "measure_preflight": {
            "path": str(args.measure_preflight),
            "schema": measure_payload.get("schema"),
            "passed": bool(measure_payload.get("status", {}).get("passed")),
        },
        "arguments": {
            "pairings": list(args.pairings),
            "matsubara_indices": list(args.matsubara_indices),
            "u_max_values": [float(value) for value in args.u_max_values],
            "radial_orders": [int(value) for value in args.radial_orders],
            "angular_orders": [int(value) for value in args.angular_orders],
            "angular_offsets": [float(value) for value in args.angular_offsets],
            "N_candidates": list(args.N_candidates),
            "temperature_K": float(args.temperature_K),
            "separation_nm": float(args.separation_nm),
            "outer_rtol": float(args.outer_rtol),
            "outer_atol_J_m2": float(args.outer_atol_J_m2),
        },
        "grid_plan": _plan_payload(plan, manifest),
        "sweet_spot_command": command,
        "status": {
            "dry_run": bool(args.dry_run),
            "all_microscopic_nodes_certified": False,
            "finite_partial_outer_q_integrals_available": False,
            "candidate_outer_q_budget_established": False,
            "production_casimir_allowed": False,
        },
        "diagnostic_only": True,
    }
    if args.dry_run:
        return base

    completed = subprocess.run(
        command,
        env=_thread_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    sweet_stdout.write_text(completed.stdout, encoding="utf-8")
    sweet_stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        base["status"]["sweet_spot_command_returncode"] = int(completed.returncode)
        base["status"]["failure"] = "transverse sweet-spot command failed"
        return base
    try:
        sweet_payload = json.loads(sweet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["status"]["failure"] = f"cannot read transverse sweet-spot payload: {exc}"
        return base

    config_results, unresolved = aggregate_certified_outer_q(
        sweet_spot_payload=sweet_payload,
        plan=plan,
        manifest=manifest,
        pairings=args.pairings,
        matsubara_indices=args.matsubara_indices,
        temperature_K=float(args.temperature_K),
    )
    all_certified = bool(
        sweet_payload.get("run_complete")
        and sweet_payload.get("all_requested_sweet_spots_established")
        and not unresolved
    )
    comparisons = compare_ladders(
        plan=plan,
        config_results=config_results,
        pairings=args.pairings,
        absolute_tolerance_J_m2=float(args.outer_atol_J_m2),
        relative_tolerance=float(args.outer_rtol),
    )
    candidate = bool(
        all_certified
        and all(
            comparisons[ladder][pairing][
                "all_passed" if ladder == "offset" else "final_transition_passed"
            ]
            for ladder in comparisons
            for pairing in args.pairings
        )
    )
    base.update(
        {
            "transverse_sweet_spot": {
                "path": str(sweet_path),
                "schema": sweet_payload.get("schema"),
                "run_complete": sweet_payload.get("run_complete"),
                "all_requested_sweet_spots_established": sweet_payload.get(
                    "all_requested_sweet_spots_established"
                ),
            },
            "config_results": config_results,
            "unresolved_microscopic_points": unresolved,
            "ladder_comparisons": comparisons,
        }
    )
    base["status"] = {
        "dry_run": False,
        "sweet_spot_command_returncode": 0,
        "all_microscopic_nodes_certified": all_certified,
        "finite_partial_outer_q_integrals_available": all_certified,
        "candidate_outer_q_budget_established": candidate,
        "production_casimir_allowed": False,
    }
    return base


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "preflight.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "unique_q_nodes": payload["grid_plan"]["unique_microscopic_q_node_count"],
                **payload["status"],
            },
            indent=2,
        )
    )
    status = payload["status"]
    if not args.dry_run and not status["all_microscopic_nodes_certified"]:
        raise SystemExit("microscopic outer-q preflight has unresolved microscopic nodes")
    if args.require_ladder_convergence and not status["candidate_outer_q_budget_established"]:
        raise SystemExit("microscopic outer-q ladder did not meet the requested convergence gate")


if __name__ == "__main__":
    main()
