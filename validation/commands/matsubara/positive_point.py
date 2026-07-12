"""Positive-Matsubara single-point convergence scan for the two-band pipeline.

For each requested k-grid this runner builds one material workspace and one q
workspace, evaluates all requested positive Matsubara indices in a single
vectorized contraction, and validates the complete single-point chain

    BdG response -> effective kernel -> RHS-aware Ward -> sheet response
    -> lab-LT reflection -> passive signed Lifshitz logdet.

The runner does not perform a q integral or Matsubara sum.  Its purpose is to
select a trustworthy microscopic k-grid and expose failures before the full
Casimir quadrature driver is introduced.
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
from typing import Any, Iterable

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.constants import KB_EV_PER_K
from lno327.electrodynamics.conventions import (
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import (
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_responses_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/single_point/raw/"
    "positive_matsubara_point_scan.csv"
)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def matsubara_energy_eV(index: int, temperature_K: float) -> float:
    """Return hbar*xi_n in eV, namely 2*pi*n*k_B*T."""

    n = int(index)
    temperature = float(temperature_K)
    if n <= 0:
        raise ValueError("positive Matsubara index must be a positive integer")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    return float(2.0 * np.pi * n * KB_EV_PER_K * temperature)


def _matrix_scalar_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (2, 2):
        raise ValueError(f"{prefix} matrix must have shape (2, 2)")
    fields: dict[str, float] = {
        f"{prefix}_frobenius_norm": float(np.linalg.norm(value)),
    }
    for label, row, col in (
        ("xx", 0, 0),
        ("xy", 0, 1),
        ("yx", 1, 0),
        ("yy", 1, 1),
    ):
        scalar = complex(value[row, col])
        fields[f"{prefix}_{label}_real"] = float(scalar.real)
        fields[f"{prefix}_{label}_imag"] = float(scalar.imag)
        fields[f"{prefix}_{label}_abs"] = float(abs(scalar))
    return fields


def _row_matrix(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(row[f"{prefix}_xx_real"], row[f"{prefix}_xx_imag"]),
                complex(row[f"{prefix}_xy_real"], row[f"{prefix}_xy_imag"]),
            ],
            [
                complex(row[f"{prefix}_yx_real"], row[f"{prefix}_yx_imag"]),
                complex(row[f"{prefix}_yy_real"], row[f"{prefix}_yy_imag"]),
            ],
        ],
        dtype=complex,
    )


def _run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    nk = int(task["nk"])
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    indices = tuple(int(value) for value in task["matsubara_indices"])
    xi_values = np.asarray(
        [matsubara_energy_eV(index, task["temperature_K"]) for index in indices],
        dtype=float,
    )

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        task["pairing"],
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(task["delta0_eV"])
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=float(xi_values[0]),
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )

    total_start = time.perf_counter()
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
    components_values = finite_q_bdg_responses_from_q_workspace(
        q_workspace,
        xi_values,
    )
    response_batch_seconds = time.perf_counter() - start

    rows: list[dict[str, Any]] = []
    lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    separation_m = float(task["separation_nm"]) * 1e-9

    for index, xi_eV, components in zip(
        indices,
        xi_values,
        components_values,
        strict=True,
    ):
        post_start = time.perf_counter()
        kernel = effective_em_kernel_from_components(
            components,
            q_model=q,
            xi_eV=float(xi_eV),
        )
        rhs = primitive_ward_rhs_from_q_workspace(q_workspace, float(xi_eV))
        ward = validate_effective_ward_xy(
            kernel,
            rhs,
            residual_tolerance=task["ward_tolerance"],
            absolute_residual_tolerance=task["ward_absolute_tolerance"],
            condition_max=task["condition_max"],
        )
        sheet = positive_matsubara_kernel_to_sheet_response(
            kernel,
            degeneracy=task["degeneracy"],
        )
        sheet_validation = validate_positive_matsubara_sheet_response(sheet)

        reflection_error = ""
        logdet_error = ""
        reflection_constructed = False
        logdet_passed = False
        reflection_matrix = np.full((2, 2), np.nan + 1j * np.nan, dtype=complex)
        reflection_spectral_radius = float("nan")
        logdet = float("nan")
        propagation_factor = float("nan")
        product_eigenvalue_max = float("nan")
        round_trip_eigenvalue_max = float("nan")
        kappa_m_inv = float("nan")

        try:
            reflection = positive_matsubara_sheet_response_to_reflection(
                sheet,
                q_lab_model=q,
                theta_rad=0.0,
                lattice_constant_m=lattice,
                require_physical=True,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            reflection_error = str(exc)
        else:
            reflection_constructed = True
            reflection_matrix = np.asarray(reflection.matrix_lt, dtype=complex)
            reflection_spectral_radius = float(
                np.max(np.abs(np.linalg.eigvals(reflection_matrix)))
            )
            kappa_m_inv = float(reflection.kappa_m_inv)
            try:
                point = passive_sheet_logdet(
                    reflection,
                    reflection,
                    separation_m=separation_m,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                logdet_error = str(exc)
            else:
                logdet_passed = True
                logdet = float(point.logdet)
                propagation_factor = float(point.propagation_factor)
                product_eigenvalue_max = float(np.max(point.product_eigenvalues))
                round_trip_eigenvalue_max = float(
                    np.max(point.round_trip_eigenvalues)
                )

        postprocess_seconds = time.perf_counter() - post_start
        sigma_tilde = np.asarray(sheet.matrix_tilde, dtype=complex)
        row: dict[str, Any] = {
            "nk": nk,
            "num_k_points": int(points.shape[0]),
            "pairing": task["pairing"],
            "qx": float(q[0]),
            "qy": float(q[1]),
            "q_abs": float(np.linalg.norm(q)),
            "temperature_K": float(task["temperature_K"]),
            "matsubara_index": int(index),
            "xi_eV": float(xi_eV),
            "delta0_eV": float(task["delta0_eV"]),
            "eta_eV": float(task["eta_eV"]),
            "degeneracy": float(task["degeneracy"]),
            "separation_nm": float(task["separation_nm"]),
            "ward_relative_tolerance": float(task["ward_tolerance"]),
            "ward_absolute_tolerance": float(task["ward_absolute_tolerance"]),
            "ward_passed": bool(ward.passed),
            "ward_condition_ok": bool(ward.condition_ok),
            "ward_denominator_collapse_detected": bool(
                ward.denominator_collapse_detected
            ),
            "ward_primitive_relative_max": max(
                ward.left.primitive_relative_residual,
                ward.right.primitive_relative_residual,
            ),
            "ward_effective_relative_max": max(
                ward.left.effective_relative_residual,
                ward.right.effective_relative_residual,
            ),
            "ward_primitive_absolute_max": max(
                ward.left.primitive_absolute_residual,
                ward.right.primitive_absolute_residual,
            ),
            "ward_effective_absolute_max": max(
                ward.left.effective_absolute_residual,
                ward.right.effective_absolute_residual,
            ),
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
            "sheet_finite": bool(sheet_validation.finite),
            "sheet_relative_imaginary_norm": float(
                sheet_validation.relative_imaginary_norm
            ),
            "sheet_relative_symmetry_residual": float(
                sheet_validation.relative_symmetry_residual
            ),
            "sheet_minimum_symmetric_eigenvalue": float(
                sheet_validation.minimum_symmetric_eigenvalue
            ),
            "sheet_validation_passed": bool(sheet_validation.passed),
            "reflection_constructed": reflection_constructed,
            "reflection_spectral_radius": reflection_spectral_radius,
            "kappa_m_inv": kappa_m_inv,
            "logdet_passed": logdet_passed,
            "logdet": logdet,
            "propagation_factor": propagation_factor,
            "product_eigenvalue_max": product_eigenvalue_max,
            "round_trip_eigenvalue_max": round_trip_eigenvalue_max,
            "single_point_pipeline_passed": bool(
                ward.passed
                and sheet_validation.passed
                and reflection_constructed
                and logdet_passed
            ),
            "reflection_error": reflection_error,
            "logdet_error": logdet_error,
            "material_seconds": material_seconds,
            "q_workspace_seconds": q_workspace_seconds,
            "response_batch_seconds": response_batch_seconds,
            "postprocess_seconds": postprocess_seconds,
            "total_wall_seconds": time.perf_counter() - total_start,
            "peak_rss_mb": _peak_rss_mb(),
            "midpoint_eigensystems": int(
                material.metadata["midpoint_eigensystem_count"]
            ),
            "shifted_eigensystems": int(
                q_workspace.metadata["shifted_eigensystem_count"]
            ),
            "pid": os.getpid(),
        }
        row.update(_matrix_scalar_fields("sigma_tilde", sigma_tilde))
        row.update(_matrix_scalar_fields("reflection", reflection_matrix))
        rows.append(row)

    return rows


def _finite_rows(rows: Iterable[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [row for row in rows if np.isfinite(float(row[field]))]


def _annotate_convergence(rows: list[dict[str, Any]]) -> None:
    indices = sorted({int(row["matsubara_index"]) for row in rows})
    for index in indices:
        group = [row for row in rows if int(row["matsubara_index"]) == index]
        valid = _finite_rows(group, "sigma_tilde_frobenius_norm")
        if not valid:
            continue
        reference = max(valid, key=lambda row: int(row["nk"]))
        reference_nk = int(reference["nk"])
        sigma_reference = _row_matrix(reference, "sigma_tilde")
        sigma_scale = max(float(np.linalg.norm(sigma_reference)), 1e-30)

        reflection_reference = None
        reflection_scale = float("nan")
        if bool(reference["reflection_constructed"]):
            reflection_reference = _row_matrix(reference, "reflection")
            reflection_scale = max(
                float(np.linalg.norm(reflection_reference)),
                1e-30,
            )

        logdet_reference = float(reference["logdet"])
        for row in group:
            sigma = _row_matrix(row, "sigma_tilde")
            row["convergence_reference_nk"] = reference_nk
            row["relative_sigma_tilde_to_reference"] = float(
                np.linalg.norm(sigma - sigma_reference) / sigma_scale
            )

            if reflection_reference is not None and bool(
                row["reflection_constructed"]
            ):
                reflection = _row_matrix(row, "reflection")
                row["relative_reflection_to_reference"] = float(
                    np.linalg.norm(reflection - reflection_reference)
                    / reflection_scale
                )
            else:
                row["relative_reflection_to_reference"] = float("nan")

            value = float(row["logdet"])
            if np.isfinite(value) and np.isfinite(logdet_reference):
                row["absolute_logdet_to_reference"] = abs(
                    value - logdet_reference
                )
                row["relative_logdet_to_reference"] = abs(
                    value - logdet_reference
                ) / max(abs(logdet_reference), 1e-30)
            else:
                row["absolute_logdet_to_reference"] = float("nan")
                row["relative_logdet_to_reference"] = float("nan")


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
        "frequency_formula": "xi_n_eV = 2*pi*n*k_B_eV_per_K*T_K",
        "single_point_chain": [
            "two_band_bdg_response",
            "amplitude_phase_schur",
            "primitive_xy_rhs_aware_ward",
            "positive_matsubara_sheet_response",
            "lab_LT_tangential_E_reflection",
            "passive_signed_logdet",
        ],
        "not_a_full_casimir_integral": True,
        "rows": rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        " n   nk      xi[eV]   Ward-mixed  cond(Keta)  sheet  min-eig(sigma)  "
        "rel-sigma   rel-R       logdet     rel-logdet  point"
    )
    print("-" * 146)
    for row in sorted(rows, key=lambda item: (item["matsubara_index"], item["nk"])):
        print(
            f"{row['matsubara_index']:2d} "
            f"{row['nk']:4d}  "
            f"{row['xi_eV']:11.4e}  "
            f"{row['ward_effective_mixed_ratio_max']:10.3e}  "
            f"{row['schur_condition_number']:10.3e}  "
            f"{str(row['sheet_validation_passed']):>5s}  "
            f"{row['sheet_minimum_symmetric_eigenvalue']:14.3e}  "
            f"{row.get('relative_sigma_tilde_to_reference', float('nan')):9.3e}  "
            f"{row.get('relative_reflection_to_reference', float('nan')):9.3e}  "
            f"{row['logdet']:11.4e}  "
            f"{row.get('relative_logdet_to_reference', float('nan')):10.3e}  "
            f"{str(row['single_point_pipeline_passed']):>5s}"
        )
        if row["reflection_error"]:
            print(f"      reflection_error: {row['reflection_error']}")
        if row["logdet_error"]:
            print(f"      logdet_error: {row['logdet_error']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", nargs="+", type=int, required=True)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--qx", type=float, required=True)
    parser.add_argument("--qy", type=float, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if any(value <= 0 for value in args.nks):
        parser.error("all --nks values must be positive")
    if any(value <= 0 for value in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not np.isfinite(args.temperature_K) or args.temperature_K <= 0.0:
        parser.error("--temperature-K must be finite and positive")
    if not np.isfinite(args.degeneracy) or args.degeneracy <= 0.0:
        parser.error("--degeneracy must be finite and positive")
    if not np.isfinite(args.separation_nm) or args.separation_nm <= 0.0:
        parser.error("--separation-nm must be finite and positive")
    if float(np.hypot(args.qx, args.qy)) == 0.0:
        parser.error("positive-Matsubara point scan requires nonzero q")
    return args


def main() -> None:
    args = _parse_args()
    base_task = {
        "pairing": args.pairing,
        "qx": args.qx,
        "qy": args.qy,
        "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV,
        "eta_eV": args.eta_eV,
        "degeneracy": args.degeneracy,
        "separation_nm": args.separation_nm,
        "ward_tolerance": args.ward_tolerance,
        "ward_absolute_tolerance": args.ward_absolute_tolerance,
        "condition_max": args.condition_max,
        "matsubara_indices": tuple(sorted(set(args.matsubara_indices))),
    }
    tasks = [{**base_task, "nk": nk} for nk in sorted(set(args.nks))]

    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            task_rows = _run_task(task)
            rows.extend(task_rows)
            print(
                f"completed nk={task['nk']} with "
                f"{len(task_rows)} Matsubara point(s)"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                task_rows = future.result()
                rows.extend(task_rows)
                print(
                    f"completed nk={task['nk']} with "
                    f"{len(task_rows)} Matsubara point(s)"
                )

    rows.sort(key=lambda row: (int(row["matsubara_index"]), int(row["nk"])))
    _annotate_convergence(rows)
    _write_outputs(rows, args.output, args)
    _print_summary(rows)
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
