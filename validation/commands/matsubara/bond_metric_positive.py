"""Positive-Matsubara d-wave bond-metric fixed-q validation and nk convergence.

For every nk this command builds one optimized material/q workspace, evaluates
all requested positive Matsubara indices as one batch, and validates

    response -> effective Ward -> sheet -> reflection -> passive logdet.

The nearest-neighbour bond phase-Hessian policy is explicitly enabled.  This is
still a single-q diagnostic; it performs no q integration or Matsubara sum and
never marks a result as Casimir-ready.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any

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
    "dwave_bond_metric_positive_nk_convergence.csv"
)


def matsubara_energy_eV(index: int, temperature_K: float) -> float:
    n = int(index)
    temperature = float(temperature_K)
    if n <= 0:
        raise ValueError("positive Matsubara index must be positive")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    return float(2.0 * np.pi * n * KB_EV_PER_K * temperature)


def _matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (2, 2):
        raise ValueError(f"{prefix} must have shape (2, 2)")
    result = {f"{prefix}_frobenius_norm": float(np.linalg.norm(value))}
    for label, row, col in (
        ("xx", 0, 0),
        ("xy", 0, 1),
        ("yx", 1, 0),
        ("yy", 1, 1),
    ):
        scalar = complex(value[row, col])
        result[f"{prefix}_{label}_real"] = float(scalar.real)
        result[f"{prefix}_{label}_imag"] = float(scalar.imag)
    return result


def _matrix_from_row(row: dict[str, Any], prefix: str) -> np.ndarray:
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


def _relative_matrix_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(np.linalg.norm(left), np.linalg.norm(right), 1e-30)
    )


def _relative_scalar_difference(left: float, right: float) -> float:
    return float(abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-30))


def _run_one(task: dict[str, Any]) -> list[dict[str, Any]]:
    nk = int(task["nk"])
    indices = tuple(int(value) for value in task["matsubara_indices"])
    xi_values = np.asarray(
        [matsubara_energy_eV(index, task["temperature_K"]) for index in indices],
        dtype=float,
    )
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    started = time.perf_counter()

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=float(xi_values[0]),
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )
    options = FiniteQEngineOptions(
        phase_hessian_policy="nearest_neighbor_bond_metric"
    )
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )
    workspace = precompute_finite_q_q_workspace(material, q)
    components_values = finite_q_bdg_responses_from_q_workspace(workspace, xi_values)

    lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    separation_m = float(task["separation_nm"]) * 1e-9
    rows: list[dict[str, Any]] = []
    for index, xi_eV, components in zip(indices, xi_values, components_values, strict=True):
        kernel = effective_em_kernel_from_components(
            components, q_model=q, xi_eV=float(xi_eV)
        )
        rhs = primitive_ward_rhs_from_q_workspace(workspace, float(xi_eV))
        ward = validate_effective_ward_xy(
            kernel,
            rhs,
            residual_tolerance=task["ward_tolerance"],
            absolute_residual_tolerance=task["ward_absolute_tolerance"],
            condition_max=task["condition_max"],
        )
        sheet = positive_matsubara_kernel_to_sheet_response(
            kernel, degeneracy=task["degeneracy"]
        )
        sheet_validation = validate_positive_matsubara_sheet_response(sheet)

        reflection_constructed = False
        reflection_error = ""
        logdet_passed = False
        logdet_error = ""
        reflection_matrix = np.full((2, 2), np.nan + 1j * np.nan, dtype=complex)
        reflection_spectral_radius = float("nan")
        logdet = float("nan")
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

        sigma = np.asarray(sheet.matrix_tilde, dtype=complex)
        row: dict[str, Any] = {
            "nk": nk,
            "num_k_points": int(points.shape[0]),
            "qx": float(q[0]),
            "qy": float(q[1]),
            "q_norm": float(np.linalg.norm(q)),
            "temperature_K": float(task["temperature_K"]),
            "matsubara_index": int(index),
            "xi_eV": float(xi_eV),
            "phase_hessian_policy": str(components.metadata["phase_hessian_policy"]),
            "phase_hessian_multiplier": float(
                components.metadata["phase_hessian_multiplier"]
            ),
            "ward_passed": bool(ward.passed),
            "ward_condition_ok": bool(ward.condition_ok),
            "ward_primitive_mixed_ratio_max": max(
                ward.left.primitive_mixed_ratio, ward.right.primitive_mixed_ratio
            ),
            "ward_effective_mixed_ratio_max": max(
                ward.left.effective_mixed_ratio, ward.right.effective_mixed_ratio
            ),
            "ward_primitive_absolute_max": max(
                ward.left.primitive_absolute_residual,
                ward.right.primitive_absolute_residual,
            ),
            "ward_effective_absolute_max": max(
                ward.left.effective_absolute_residual,
                ward.right.effective_absolute_residual,
            ),
            "schur_condition_number": float(ward.schur_condition_number),
            "schur_inverse_method": str(ward.schur_inverse_method),
            "sheet_validation_passed": bool(sheet_validation.passed),
            "sheet_relative_imaginary_norm": float(
                sheet_validation.relative_imaginary_norm
            ),
            "sheet_relative_symmetry_residual": float(
                sheet_validation.relative_symmetry_residual
            ),
            "sheet_minimum_symmetric_eigenvalue": float(
                sheet_validation.minimum_symmetric_eigenvalue
            ),
            "reflection_constructed": reflection_constructed,
            "reflection_spectral_radius": reflection_spectral_radius,
            "logdet_passed": logdet_passed,
            "logdet": logdet,
            "point_pipeline_passed": bool(
                ward.passed
                and sheet_validation.passed
                and reflection_constructed
                and logdet_passed
            ),
            "reflection_error": reflection_error,
            "logdet_error": logdet_error,
            "wall_seconds": float(time.perf_counter() - started),
            "pid": os.getpid(),
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }
        row.update(_matrix_fields("sigma_tilde", sigma))
        row.update(_matrix_fields("reflection", reflection_matrix))
        rows.append(row)
    return rows


def _annotate_convergence(rows: list[dict[str, Any]]) -> None:
    for index in sorted({int(row["matsubara_index"]) for row in rows}):
        group = sorted(
            [row for row in rows if int(row["matsubara_index"]) == index],
            key=lambda row: int(row["nk"]),
        )
        reference = group[-1]
        sigma_reference = _matrix_from_row(reference, "sigma_tilde")
        reflection_reference = _matrix_from_row(reference, "reflection")
        previous: dict[str, Any] | None = None
        for row in group:
            row["sigma_relative_to_finest"] = _relative_matrix_difference(
                _matrix_from_row(row, "sigma_tilde"), sigma_reference
            )
            row["reflection_relative_to_finest"] = _relative_matrix_difference(
                _matrix_from_row(row, "reflection"), reflection_reference
            )
            row["logdet_relative_to_finest"] = _relative_scalar_difference(
                row["logdet"], reference["logdet"]
            )
            if previous is None:
                row["sigma_relative_to_previous"] = float("nan")
                row["reflection_relative_to_previous"] = float("nan")
                row["logdet_relative_to_previous"] = float("nan")
            else:
                row["sigma_relative_to_previous"] = _relative_matrix_difference(
                    _matrix_from_row(row, "sigma_tilde"),
                    _matrix_from_row(previous, "sigma_tilde"),
                )
                row["reflection_relative_to_previous"] = _relative_matrix_difference(
                    _matrix_from_row(row, "reflection"),
                    _matrix_from_row(previous, "reflection"),
                )
                row["logdet_relative_to_previous"] = _relative_scalar_difference(
                    row["logdet"], previous["logdet"]
                )
            previous = row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], convergence_tolerance: float) -> str:
    lines = [
        "positive-Matsubara d-wave bond-metric validation",
        "=" * 52,
        " n   nk       xi[eV]    Ward-ratio  condition   sheet  reflection logdet  sigma-prev",
        "-" * 91,
    ]
    for row in sorted(rows, key=lambda item: (item["matsubara_index"], item["nk"])):
        lines.append(
            f"{int(row['matsubara_index']):2d} {int(row['nk']):4d} "
            f"{float(row['xi_eV']):11.4e} "
            f"{float(row['ward_effective_mixed_ratio_max']):11.3e} "
            f"{float(row['schur_condition_number']):10.3e} "
            f"{str(bool(row['sheet_validation_passed'])):>6s} "
            f"{str(bool(row['reflection_constructed'])):>10s} "
            f"{str(bool(row['logdet_passed'])):>6s} "
            f"{float(row['sigma_relative_to_previous']):10.3e}"
        )
    lines.extend(
        [
            "",
            f"all point pipelines passed = {all(bool(row['point_pipeline_passed']) for row in rows)}",
            f"convergence tolerance = {convergence_tolerance:.3e}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", nargs="+", type=int, required=True)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qx", type=float, required=True)
    parser.add_argument("--qy", type=float, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-9)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if any(value <= 0 for value in args.nks):
        parser.error("all --nks values must be positive")
    if any(value <= 0 for value in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not np.isfinite([args.qx, args.qy]).all() or np.hypot(args.qx, args.qy) == 0.0:
        parser.error("(qx, qy) must be finite and nonzero")
    if not np.isfinite(args.condition_max) or args.condition_max <= 0.0:
        parser.error("--condition-max must be finite and positive")
    if not np.isfinite(args.convergence_tolerance) or args.convergence_tolerance < 0.0:
        parser.error("--convergence-tolerance must be finite and non-negative")

    common = {
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
    tasks = [{**common, "nk": nk} for nk in sorted(set(args.nks))]
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            rows.extend(_run_one(task))
            print(f"completed nk={task['nk']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {executor.submit(_run_one, task): task["nk"] for task in tasks}
            for future in as_completed(futures):
                rows.extend(future.result())
                print(f"completed nk={futures[future]}", flush=True)

    rows.sort(key=lambda row: (int(row["matsubara_index"]), int(row["nk"])))
    _annotate_convergence(rows)
    _write_csv(args.output, rows)

    finest_passed = True
    for index in sorted({int(row["matsubara_index"]) for row in rows}):
        group = [row for row in rows if int(row["matsubara_index"]) == index]
        if len(group) < 2:
            finest_passed = False
            continue
        finest = max(group, key=lambda row: int(row["nk"]))
        finest_passed = bool(
            finest_passed
            and float(finest["sigma_relative_to_previous"]) <= args.convergence_tolerance
            and float(finest["reflection_relative_to_previous"])
            <= args.convergence_tolerance
            and float(finest["logdet_relative_to_previous"]) <= args.convergence_tolerance
        )
    payload = {
        "schema": "dwave_bond_metric_positive_nk_convergence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "validation_gate": {
            "all_point_pipelines_passed": all(
                bool(row["point_pipeline_passed"]) for row in rows
            ),
            "finest_pair_converged_for_all_indices": finest_passed,
            "convergence_tolerance": args.convergence_tolerance,
            "passed": bool(
                all(bool(row["point_pipeline_passed"]) for row in rows)
                and finest_passed
            ),
        },
        "status": {
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    summary = _summary(rows, args.convergence_tolerance)
    args.output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"CSV:     {args.output}")
    print(f"JSON:    {args.output.with_suffix('.json')}")
    print(f"summary: {args.output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
