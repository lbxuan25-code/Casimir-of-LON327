"""Validate exact-static d-wave response with Ward-compatible periodic shifts.

Each task uses a weighted union of complete periodic tensor lattices.  All
primitive electromagnetic and collective blocks are merged over the common
quadrature before one amplitude/phase Schur complement.  Unlike local cell
refinement, this preserves the discrete periodic-lattice structure that the
primitive finite-q Ward cancellation relies on.
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

from lno327 import KuboConfig
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.static_gauge_projection import (
    PROJECT_AFTER_VALIDATED_WARD,
    static_matsubara_kernel_to_sheet_response_with_policy,
)
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.dwave_periodic_multishift_quadrature import (
    DWavePeriodicMultishiftOptions,
    build_dwave_periodic_multishift_quadrature,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_periodic_multishift/raw/"
    "dwave_static_periodic_multishift_scan.csv"
)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (2, 2):
        raise ValueError(f"{prefix} must have shape (2, 2)")
    fields = {f"{prefix}_frobenius_norm": float(np.linalg.norm(value))}
    for label, i, j in (("ll", 0, 0), ("lt", 0, 1), ("tl", 1, 0), ("tt", 1, 1)):
        scalar = complex(value[i, j])
        fields[f"{prefix}_{label}_real"] = float(scalar.real)
        fields[f"{prefix}_{label}_imag"] = float(scalar.imag)
    return fields


def _matrix_from_row(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(row[f"{prefix}_ll_real"], row[f"{prefix}_ll_imag"]),
                complex(row[f"{prefix}_lt_real"], row[f"{prefix}_lt_imag"]),
            ],
            [
                complex(row[f"{prefix}_tl_real"], row[f"{prefix}_tl_imag"]),
                complex(row[f"{prefix}_tt_real"], row[f"{prefix}_tt_imag"]),
            ],
        ],
        dtype=complex,
    )


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )

    total_start = time.perf_counter()
    start = time.perf_counter()
    points, weights, quadrature = build_dwave_periodic_multishift_quadrature(
        q,
        DWavePeriodicMultishiftOptions(
            base_nk=task["base_nk"],
            shift_order=task["shift_order"],
            max_quadrature_points=task["max_quadrature_points"],
        ),
    )
    quadrature_seconds = time.perf_counter() - start

    start = time.perf_counter()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        FiniteQEngineOptions(),
    )
    material_seconds = time.perf_counter() - start

    start = time.perf_counter()
    q_workspace = precompute_finite_q_q_workspace(material, q)
    q_workspace_seconds = time.perf_counter() - start

    start = time.perf_counter()
    components = finite_q_bdg_response_from_q_workspace(q_workspace, 0.0)
    response_seconds = time.perf_counter() - start

    start = time.perf_counter()
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    rhs = primitive_ward_rhs_from_q_workspace(q_workspace, 0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=task["ward_tolerance"],
        absolute_residual_tolerance=task["ward_absolute_tolerance"],
        condition_max=task["condition_max"],
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
    projection_error = ""
    projection_correction = float("nan")
    projected_longitudinal = float("nan")
    projected_static_passed = False
    reflection_constructed = False
    reflection_error = ""
    logdet_passed = False
    logdet_error = ""
    reflection_matrix = np.full((2, 2), np.nan + 1j * np.nan, dtype=complex)
    logdet = float("nan")

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
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        projection_error = str(exc)
    else:
        projection_eligible = True
        projection_correction = float(
            projected.metadata["relative_projection_correction_norm"]
        )
        projected_longitudinal = float(
            projected.validation.relative_longitudinal_gauge_residual
        )
        projected_static_passed = bool(projected.validation.passed)
        try:
            reflection = static_sheet_response_to_reflection(
                projected,
                q_lab_model=q,
                theta_rad=0.0,
                require_physical=True,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            reflection_error = str(exc)
        else:
            reflection_constructed = True
            reflection_matrix = np.asarray(reflection.matrix_lt, dtype=complex)
            try:
                point = passive_sheet_logdet(
                    reflection,
                    reflection,
                    separation_m=float(task["separation_nm"]) * 1e-9,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                logdet_error = str(exc)
            else:
                logdet_passed = True
                logdet = float(point.logdet)

    postprocess_seconds = time.perf_counter() - start
    row: dict[str, Any] = {
        "base_nk": int(task["base_nk"]),
        "shift_order": int(task["shift_order"]),
        "num_grid_shifts": int(quadrature["num_grid_shifts"]),
        "num_points_per_shift": int(quadrature["num_points_per_shift"]),
        "num_quadrature_points": int(quadrature["num_quadrature_points"]),
        "effective_points_per_axis": int(quadrature["effective_points_per_axis"]),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_abs": float(np.linalg.norm(q)),
        "temperature_K": float(task["temperature_K"]),
        "delta0_eV": float(task["delta0_eV"]),
        "eta_eV": float(task["eta_eV"]),
        "ward_passed": bool(ward.passed),
        "ward_condition_ok": bool(ward.condition_ok),
        "ward_primitive_mixed_ratio_max": max(
            ward.left.primitive_mixed_ratio,
            ward.right.primitive_mixed_ratio,
        ),
        "ward_effective_mixed_ratio_max": max(
            ward.left.effective_mixed_ratio,
            ward.right.effective_mixed_ratio,
        ),
        "schur_condition_number": float(ward.schur_condition_number),
        "schur_inverse_method": ward.schur_inverse_method,
        "raw_relative_imaginary_norm": float(raw.validation.relative_imaginary_norm),
        "raw_relative_longitudinal_gauge_residual": float(
            raw.validation.relative_longitudinal_gauge_residual
        ),
        "raw_relative_density_transverse_mixing": float(
            raw.validation.relative_density_transverse_mixing
        ),
        "raw_static_validation_passed": bool(raw.validation.passed),
        "chi_bar": float(raw.chi_bar),
        "dbar_t": float(raw.dbar_t),
        "projection_eligible": projection_eligible,
        "projection_correction": projection_correction,
        "projected_longitudinal": projected_longitudinal,
        "projected_static_passed": projected_static_passed,
        "projection_error": projection_error,
        "reflection_constructed": reflection_constructed,
        "reflection_error": reflection_error,
        "logdet_passed": logdet_passed,
        "logdet": logdet,
        "logdet_error": logdet_error,
        "separation_nm": float(task["separation_nm"]),
        "quadrature_seconds": quadrature_seconds,
        "material_seconds": material_seconds,
        "q_workspace_seconds": q_workspace_seconds,
        "response_seconds": response_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_wall_seconds": time.perf_counter() - total_start,
        "peak_rss_mb": _peak_rss_mb(),
        "pid": os.getpid(),
    }
    row.update(_matrix_fields("reflection", reflection_matrix))
    return row


def _annotate_convergence(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    reference = max(
        rows,
        key=lambda row: (
            int(row["num_quadrature_points"]),
            int(row["shift_order"]),
            int(row["base_nk"]),
        ),
    )
    chi_ref = float(reference["chi_bar"])
    dbar_ref = float(reference["dbar_t"])
    reflection_ref = _matrix_from_row(reference, "reflection")
    reflection_scale = max(float(np.linalg.norm(reflection_ref)), 1e-30)
    logdet_ref = float(reference["logdet"])
    for row in rows:
        row["reference_num_quadrature_points"] = int(
            reference["num_quadrature_points"]
        )
        row["relative_chi_bar_to_reference"] = abs(float(row["chi_bar"]) - chi_ref) / max(
            abs(chi_ref), 1e-30
        )
        row["relative_dbar_t_to_reference"] = abs(float(row["dbar_t"]) - dbar_ref) / max(
            abs(dbar_ref), 1e-30
        )
        matrix = _matrix_from_row(row, "reflection")
        if np.isfinite(matrix.real).all() and np.isfinite(reflection_ref.real).all():
            row["relative_reflection_to_reference"] = float(
                np.linalg.norm(matrix - reflection_ref) / reflection_scale
            )
        else:
            row["relative_reflection_to_reference"] = float("nan")
        value = float(row["logdet"])
        if np.isfinite(value) and np.isfinite(logdet_ref):
            row["relative_logdet_to_reference"] = abs(value - logdet_ref) / max(
                abs(logdet_ref), 1e-30
            )
        else:
            row["relative_logdet_to_reference"] = float("nan")


def _write_outputs(rows: list[dict[str, Any]], output: Path, args: argparse.Namespace) -> None:
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
        "integration_contract": (
            "weighted complete periodic lattices; primitive blocks merged before one Schur; "
            "no local adaptive cell replacement"
        ),
        "rows": rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        " base order shifts        Nk  raw-long   chi_bar    Dbar_T   rel-chi    rel-D   "
        "Ward-prim Ward-eff   proj    rel-R rel-logdet"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['base_nk']:5d} "
            f"{row['shift_order']:5d} "
            f"{row['num_grid_shifts']:6d} "
            f"{row['num_quadrature_points']:9d} "
            f"{row['raw_relative_longitudinal_gauge_residual']:10.3e} "
            f"{row['chi_bar']:9.5f} "
            f"{row['dbar_t']:9.5f} "
            f"{row['relative_chi_bar_to_reference']:9.2e} "
            f"{row['relative_dbar_t_to_reference']:9.2e} "
            f"{row['ward_primitive_mixed_ratio_max']:9.2e} "
            f"{row['ward_effective_mixed_ratio_max']:9.2e} "
            f"{str(row['projection_eligible']):>6s} "
            f"{row['relative_reflection_to_reference']:8.2e} "
            f"{row['relative_logdet_to_reference']:10.2e}"
        )
        if row["projection_error"]:
            print(f"    projection_error: {row['projection_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nks", type=int, nargs="+", required=True)
    parser.add_argument("--shift-orders", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--max-quadrature-points", type=int, default=400_000)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--raw-longitudinal-ceiling", type=float, default=1e-3)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-7)
    parser.add_argument("--reality-tolerance", type=float, default=1e-9)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tasks = []
    for base_nk in args.base_nks:
        for shift_order in args.shift_orders:
            tasks.append(
                {
                    "base_nk": base_nk,
                    "shift_order": shift_order,
                    "qx": args.qx,
                    "qy": args.qy,
                    "temperature_K": args.temperature_K,
                    "delta0_eV": args.delta0_eV,
                    "eta_eV": args.eta_eV,
                    "max_quadrature_points": args.max_quadrature_points,
                    "ward_tolerance": args.ward_tolerance,
                    "ward_absolute_tolerance": args.ward_absolute_tolerance,
                    "condition_max": args.condition_max,
                    "raw_longitudinal_ceiling": args.raw_longitudinal_ceiling,
                    "longitudinal_tolerance": args.longitudinal_tolerance,
                    "mixing_tolerance": args.mixing_tolerance,
                    "reality_tolerance": args.reality_tolerance,
                    "passivity_tolerance": args.passivity_tolerance,
                    "separation_nm": args.separation_nm,
                }
            )

    rows: list[dict[str, Any]] = []
    sweep_start = time.perf_counter()
    if int(args.workers) == 1:
        for task in tasks:
            row = _run_task(task)
            rows.append(row)
            print(
                f"completed base={row['base_nk']} order={row['shift_order']} "
                f"with {row['num_quadrature_points']} points"
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"completed base={row['base_nk']} order={row['shift_order']} "
                    f"with {row['num_quadrature_points']} points"
                )

    rows.sort(
        key=lambda row: (
            int(row["num_quadrature_points"]),
            int(row["shift_order"]),
            int(row["base_nk"]),
        )
    )
    _annotate_convergence(rows)
    _write_outputs(rows, args.output, args)
    _print_summary(rows)
    print(f"Sweep wall time: {time.perf_counter() - sweep_start:.4f} s")
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
