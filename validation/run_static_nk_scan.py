"""Timed zero-Matsubara k-grid convergence scan using optimized workspaces.

The scan parallelizes independent ``nk`` values only.  Each worker keeps BLAS
threading external to this module so callers can avoid process/thread
oversubscription with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, and related
environment variables.
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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
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
    "validation/outputs/zero_matsubara/static_nk_convergence/raw/static_nk_scan.csv"
)


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.  This validation CLI targets the project's
    # Linux/WSL environment.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    nk = int(task["nk"])
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    cpu_start = time.process_time()
    total_start = time.perf_counter()

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(task["pairing"], phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )
    options = FiniteQEngineOptions()

    start = time.perf_counter()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )
    material_seconds = time.perf_counter() - start

    start = time.perf_counter()
    q_workspace = precompute_finite_q_q_workspace(material, q)
    q_workspace_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        components = finite_q_bdg_response_from_q_workspace(q_workspace, 0.0)
    response_seconds = time.perf_counter() - start

    start = time.perf_counter()
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    rhs = primitive_ward_rhs_from_q_workspace(q_workspace, 0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=task["ward_tolerance"],
    )
    static = static_matsubara_kernel_to_sheet_response(kernel, ward)
    postprocess_seconds = time.perf_counter() - start

    total_seconds = time.perf_counter() - total_start
    cpu_seconds = time.process_time() - cpu_start
    warning_messages = [str(item.message) for item in caught]

    return {
        "nk": nk,
        "num_k_points": int(points.shape[0]),
        "pairing": task["pairing"],
        "qx": float(q[0]),
        "qy": float(q[1]),
        "temperature_K": float(task["temperature_K"]),
        "delta0_eV": float(task["delta0_eV"]),
        "eta_eV": float(task["eta_eV"]),
        "ward_tolerance": float(task["ward_tolerance"]),
        "material_seconds": material_seconds,
        "q_workspace_seconds": q_workspace_seconds,
        "response_seconds": response_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_wall_seconds": total_seconds,
        "process_cpu_seconds": cpu_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "midpoint_eigensystems": int(material.metadata["midpoint_eigensystem_count"]),
        "shifted_eigensystems": int(q_workspace.metadata["shifted_eigensystem_count"]),
        "ward_left_primitive": ward.left.primitive_relative_residual,
        "ward_right_primitive": ward.right.primitive_relative_residual,
        "ward_left_effective": ward.left.effective_relative_residual,
        "ward_right_effective": ward.right.effective_relative_residual,
        "schur_condition_number": ward.schur_condition_number,
        "ward_passed": bool(ward.passed),
        "relative_imaginary_norm": static.validation.relative_imaginary_norm,
        "relative_longitudinal_gauge_residual": (
            static.validation.relative_longitudinal_gauge_residual
        ),
        "relative_density_transverse_mixing": (
            static.validation.relative_density_transverse_mixing
        ),
        "chi_bar": static.chi_bar,
        "dbar_t": static.dbar_t,
        "static_validation_passed": bool(static.validation.passed),
        "warning_count": len(warning_messages),
        "warning_first": warning_messages[0] if warning_messages else "",
        "pid": os.getpid(),
    }


def _write_outputs(rows: list[dict[str, Any]], output: Path, args: argparse.Namespace) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = output.with_suffix(".json")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "workers": args.workers,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "rows": rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        " nk    Nk      total[s]  material[s]  q-cache[s]  response[s]  "
        "Ward-eff(max)  longitudinal  mixing       chi_bar      Dbar_T"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ward_eff = max(row["ward_left_effective"], row["ward_right_effective"])
        print(
            f"{row['nk']:3d} {row['num_k_points']:6d} "
            f"{row['total_wall_seconds']:11.4f} "
            f"{row['material_seconds']:12.4f} "
            f"{row['q_workspace_seconds']:10.4f} "
            f"{row['response_seconds']:11.4f} "
            f"{ward_eff:13.3e} "
            f"{row['relative_longitudinal_gauge_residual']:12.3e} "
            f"{row['relative_density_transverse_mixing']:10.3e} "
            f"{row['chi_bar']:12.5e} "
            f"{row['dbar_t']:12.5e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    if any(nk <= 0 for nk in args.nks):
        parser.error("all --nks values must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not np.isfinite([args.qx, args.qy]).all() or np.hypot(args.qx, args.qy) == 0.0:
        parser.error("(qx, qy) must be finite and nonzero")

    common = {
        "pairing": args.pairing,
        "qx": args.qx,
        "qy": args.qy,
        "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV,
        "eta_eV": args.eta_eV,
        "ward_tolerance": args.ward_tolerance,
    }
    tasks = [{**common, "nk": nk} for nk in sorted(set(args.nks))]

    sweep_start = time.perf_counter()
    if args.workers == 1:
        rows = [_run_one(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {executor.submit(_run_one, task): task["nk"] for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"completed nk={row['nk']} in {row['total_wall_seconds']:.3f} s "
                    f"(peak RSS {row['peak_rss_mb']:.1f} MiB)",
                    flush=True,
                )
        rows.sort(key=lambda row: row["nk"])
    sweep_seconds = time.perf_counter() - sweep_start

    _write_outputs(rows, args.output, args)
    _print_summary(rows)
    print(f"\nSweep wall time: {sweep_seconds:.4f} s")
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
