"""Focused validation of the exact-static longitudinal projection policy.

This runner is intentionally smaller than ``run_static_nk_scan``.  It reports
only the raw static gates, projection eligibility/correction, projected gates,
and the retained density/transverse observables.  It is intended to establish a
moderate ``nk`` budget after the raw quadrature diagnostics have already
localized the remaining leakage to the analytically pure-gauge longitudinal
row and column.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.electrodynamics.static_gauge_projection import (
    DEFAULT_PROJECTION_RAW_LONGITUDINAL_CEILING,
    PROJECT_AFTER_VALIDATED_WARD,
    static_matsubara_kernel_to_sheet_response_with_policy,
)
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/static_gauge_projection/raw/"
    "static_projection_scan.csv"
)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    nk = int(task["nk"])
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        task["pairing"],
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(task["delta0_eV"])
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )

    total_start = time.perf_counter()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        FiniteQEngineOptions(),
    )
    q_workspace = precompute_finite_q_q_workspace(material, q)
    components = finite_q_bdg_response_from_q_workspace(q_workspace, 0.0)
    kernel = effective_em_kernel_from_components(
        components,
        q_model=q,
        xi_eV=0.0,
    )
    rhs = primitive_ward_rhs_from_q_workspace(q_workspace, 0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=task["ward_tolerance"],
    )
    raw = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=task["longitudinal_tolerance"],
        mixing_tolerance=task["mixing_tolerance"],
        reality_tolerance=task["reality_tolerance"],
        passivity_tolerance=task["passivity_tolerance"],
    )

    projection_eligible = False
    projection_applied = False
    projection_error = ""
    projected_longitudinal = float("nan")
    relative_projection_correction = float("nan")
    projected_static_passed = False
    chi_delta = float("nan")
    dbar_t_delta = float("nan")

    try:
        projected = static_matsubara_kernel_to_sheet_response_with_policy(
            kernel,
            ward,
            longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
            projection_raw_longitudinal_ceiling=task["raw_longitudinal_ceiling"],
            longitudinal_tolerance=task["longitudinal_tolerance"],
            mixing_tolerance=task["mixing_tolerance"],
            reality_tolerance=task["reality_tolerance"],
            passivity_tolerance=task["passivity_tolerance"],
        )
    except (ValueError, RuntimeError) as exc:
        projection_error = str(exc)
    else:
        projection_eligible = True
        projection_applied = bool(projected.metadata["gauge_projection_applied"])
        projected_longitudinal = float(
            projected.validation.relative_longitudinal_gauge_residual
        )
        relative_projection_correction = float(
            projected.metadata["relative_projection_correction_norm"]
        )
        projected_static_passed = bool(projected.validation.passed)
        chi_delta = float(projected.chi_bar - raw.chi_bar)
        dbar_t_delta = float(projected.dbar_t - raw.dbar_t)

    total_seconds = time.perf_counter() - total_start
    return {
        "nk": nk,
        "num_k_points": int(points.shape[0]),
        "pairing": task["pairing"],
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_abs": float(np.linalg.norm(q)),
        "temperature_K": float(task["temperature_K"]),
        "delta0_eV": float(task["delta0_eV"]),
        "ward_tolerance": float(task["ward_tolerance"]),
        "raw_longitudinal_ceiling": float(task["raw_longitudinal_ceiling"]),
        "target_longitudinal_tolerance": float(task["longitudinal_tolerance"]),
        "ward_passed": bool(ward.passed),
        "ward_condition_ok": bool(ward.condition_ok),
        "ward_denominator_collapse_detected": bool(
            ward.denominator_collapse_detected
        ),
        "ward_effective_relative_max": max(
            ward.left.effective_relative_residual,
            ward.right.effective_relative_residual,
        ),
        "ward_effective_mixed_ratio_max": max(
            ward.left.effective_mixed_ratio,
            ward.right.effective_mixed_ratio,
        ),
        "schur_condition_number": float(ward.schur_condition_number),
        "schur_inverse_method": ward.schur_inverse_method,
        "raw_relative_imaginary_norm": raw.validation.relative_imaginary_norm,
        "raw_relative_longitudinal_gauge_residual": (
            raw.validation.relative_longitudinal_gauge_residual
        ),
        "raw_relative_density_transverse_mixing": (
            raw.validation.relative_density_transverse_mixing
        ),
        "raw_static_validation_passed_at_target": bool(raw.validation.passed),
        "projection_eligible": projection_eligible,
        "projection_applied": projection_applied,
        "relative_projection_correction_norm": relative_projection_correction,
        "projected_relative_longitudinal_gauge_residual": projected_longitudinal,
        "projected_static_validation_passed": projected_static_passed,
        "chi_bar": raw.chi_bar,
        "dbar_t": raw.dbar_t,
        "chi_bar_projection_delta": chi_delta,
        "dbar_t_projection_delta": dbar_t_delta,
        "projection_error": projection_error,
        "total_wall_seconds": total_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "pid": os.getpid(),
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    output: Path,
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "workers": args.workers,
        "policy": PROJECT_AFTER_VALIDATED_WARD,
        "projection_formula": "K_projected = diag(1,0,1) K_raw diag(1,0,1)",
        "rows": rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        " nk      raw-long    raw-pass  eligible  correction   projected-long  "
        "proj-pass      chi_bar      Dbar_T   Ward-mixed"
    )
    print("-" * 126)
    for row in rows:
        correction = row["relative_projection_correction_norm"]
        projected = row["projected_relative_longitudinal_gauge_residual"]
        print(
            f"{row['nk']:3d}  "
            f"{row['raw_relative_longitudinal_gauge_residual']:12.3e}  "
            f"{str(row['raw_static_validation_passed_at_target']):>8s}  "
            f"{str(row['projection_eligible']):>8s}  "
            f"{correction:11.3e}  "
            f"{projected:14.3e}  "
            f"{str(row['projected_static_validation_passed']):>9s}  "
            f"{row['chi_bar']:11.5e}  "
            f"{row['dbar_t']:11.5e}  "
            f"{row['ward_effective_mixed_ratio_max']:11.3e}"
        )
        if row["projection_error"]:
            print(f"     projection_error: {row['projection_error']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--qx", type=float, required=True)
    parser.add_argument("--qy", type=float, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--raw-longitudinal-ceiling",
        type=float,
        default=DEFAULT_PROJECTION_RAW_LONGITUDINAL_CEILING,
    )
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-7)
    parser.add_argument("--reality-tolerance", type=float, default=1e-9)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(nk <= 0 for nk in args.nks):
        parser.error("all --nks values must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if float(np.hypot(args.qx, args.qy)) == 0.0:
        parser.error("q must be nonzero")
    return args


def main() -> None:
    args = _parse_args()
    tasks = [
        {
            "nk": nk,
            "pairing": args.pairing,
            "qx": args.qx,
            "qy": args.qy,
            "temperature_K": args.temperature_K,
            "delta0_eV": args.delta0_eV,
            "eta_eV": args.eta_eV,
            "ward_tolerance": args.ward_tolerance,
            "raw_longitudinal_ceiling": args.raw_longitudinal_ceiling,
            "longitudinal_tolerance": args.longitudinal_tolerance,
            "mixing_tolerance": args.mixing_tolerance,
            "reality_tolerance": args.reality_tolerance,
            "passivity_tolerance": args.passivity_tolerance,
        }
        for nk in sorted(set(args.nks))
    ]

    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            row = _run_task(task)
            rows.append(row)
            print(
                f"completed nk={row['nk']} in {row['total_wall_seconds']:.3f} s "
                f"(peak RSS {row['peak_rss_mb']:.1f} MiB)"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(_run_task, task): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                row = future.result()
                rows.append(row)
                print(
                    f"completed nk={row['nk']} in {row['total_wall_seconds']:.3f} s "
                    f"(peak RSS {row['peak_rss_mb']:.1f} MiB)"
                )

    rows.sort(key=lambda row: int(row["nk"]))
    _write_outputs(rows, args.output, args)
    _print_summary(rows)
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
