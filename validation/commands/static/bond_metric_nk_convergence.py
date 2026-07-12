"""Fixed-physical-q d-wave bond-metric zero-Matsubara nk convergence.

Unlike commensurate integer-shift Ward audits, this command keeps ``q_model``
fixed while changing the uniform Brillouin-zone mesh.  It is therefore the
appropriate runner for convergence of ``chi_bar`` and ``Dbar_T``.  The command
also records how far each fixed q lies from an integer grid translation so Ward
closure and observable convergence are not conflated.

The nearest-neighbour bond phase-Hessian policy is explicitly enabled.  All
results remain diagnostic-only and invalid for Casimir input.
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
import warnings

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
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
    "validation/outputs/zero_matsubara/static_nk_convergence/raw/"
    "dwave_bond_metric_fixed_q_nk_convergence.csv"
)


def _relative_difference(left: float, right: float) -> float:
    return float(abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-30))


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    nk = int(task["nk"])
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    started = time.perf_counter()

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        components = finite_q_bdg_response_from_q_workspace(workspace, 0.0)
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    rhs = primitive_ward_rhs_from_q_workspace(workspace, 0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=task["mixed_ward_tolerance"],
        absolute_residual_tolerance=task["mixed_ward_absolute_tolerance"],
        condition_max=task["condition_max"],
    )
    strict = validate_strict_static_ward_closure(
        kernel,
        ward,
        primitive_tolerance=task["primitive_tolerance"],
        amplitude_tolerance=task["amplitude_tolerance"],
        phase_tolerance=task["phase_tolerance"],
        effective_direct_tolerance=task["effective_direct_tolerance"],
        effective_residual_tolerance=task["effective_residual_tolerance"],
        longitudinal_tolerance=task["longitudinal_tolerance"],
        condition_max=task["condition_max"],
    )
    sheet = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        reality_tolerance=1.0,
        longitudinal_tolerance=1.0,
        mixing_tolerance=1.0,
        passivity_tolerance=1.0,
    )

    step = 2.0 * np.pi / float(nk)
    grid_shift = q / step
    nearest_shift = np.rint(grid_shift).astype(int)
    shift_error = grid_shift - nearest_shift
    warning_messages = [str(item.message) for item in caught]
    return {
        "nk": nk,
        "num_k_points": int(points.shape[0]),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": float(np.linalg.norm(q)),
        "grid_step": float(step),
        "grid_shift_x": float(grid_shift[0]),
        "grid_shift_y": float(grid_shift[1]),
        "nearest_integer_mx": int(nearest_shift[0]),
        "nearest_integer_my": int(nearest_shift[1]),
        "integer_shift_error_norm": float(np.linalg.norm(shift_error)),
        "translation_by_q_exact_index_permutation": bool(
            np.all(np.abs(shift_error) <= 1e-12)
        ),
        "phase_hessian_policy": str(components.metadata["phase_hessian_policy"]),
        "phase_hessian_multiplier": float(components.metadata["phase_hessian_multiplier"]),
        "mixed_ward_passed": bool(ward.passed),
        "strict_gate_passed": bool(strict.passed),
        **{
            key: value
            for key, value in strict.to_dict().items()
            if key not in {"metadata", "passed"}
        },
        "relative_imaginary_norm": float(sheet.validation.relative_imaginary_norm),
        "relative_density_transverse_mixing": float(
            sheet.validation.relative_density_transverse_mixing
        ),
        "chi_bar": float(sheet.chi_bar),
        "dbar_t": float(sheet.dbar_t),
        "warning_count": len(warning_messages),
        "warning_first": warning_messages[0] if warning_messages else "",
        "wall_seconds": float(time.perf_counter() - started),
        "pid": os.getpid(),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }


def _add_convergence_metrics(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: int(row["nk"]))
    reference = rows[-1]
    previous: dict[str, Any] | None = None
    for row in rows:
        row["chi_bar_relative_to_finest"] = _relative_difference(
            row["chi_bar"], reference["chi_bar"]
        )
        row["dbar_t_relative_to_finest"] = _relative_difference(
            row["dbar_t"], reference["dbar_t"]
        )
        if previous is None:
            row["chi_bar_relative_to_previous"] = float("nan")
            row["dbar_t_relative_to_previous"] = float("nan")
        else:
            row["chi_bar_relative_to_previous"] = _relative_difference(
                row["chi_bar"], previous["chi_bar"]
            )
            row["dbar_t_relative_to_previous"] = _relative_difference(
                row["dbar_t"], previous["dbar_t"]
            )
        previous = row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], observable_tolerance: float) -> str:
    lines = [
        "d-wave bond-metric fixed-q nk convergence",
        "=" * 47,
        " nk     shift-error  phase/q      eff-direct/q  longitudinal  chi_bar      Dbar_T       strict",
        "-" * 103,
    ]
    for row in rows:
        lines.append(
            f"{int(row['nk']):4d} "
            f"{float(row['integer_shift_error_norm']):12.3e} "
            f"{float(row['phase_defect_over_q']):12.3e} "
            f"{float(row['effective_direct_over_q']):13.3e} "
            f"{float(row['relative_longitudinal_gauge_residual']):12.3e} "
            f"{float(row['chi_bar']):12.5e} "
            f"{float(row['dbar_t']):12.5e} "
            f"{str(bool(row['strict_gate_passed'])):>7s}"
        )
    finest_pair_converged = False
    if len(rows) >= 2:
        finest = rows[-1]
        finest_pair_converged = bool(
            float(finest["chi_bar_relative_to_previous"]) <= observable_tolerance
            and float(finest["dbar_t_relative_to_previous"]) <= observable_tolerance
        )
    lines.extend(
        [
            "",
            f"all strict gates passed = {all(bool(row['strict_gate_passed']) for row in rows)}",
            f"finest-pair observable convergence passed = {finest_pair_converged}",
            f"observable relative tolerance = {observable_tolerance:.3e}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qx", type=float, default=0.0300152164356)
    parser.add_argument("--qy", type=float, default=0.0200101442904)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--mixed-ward-tolerance", type=float, default=1e-9)
    parser.add_argument("--mixed-ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--primitive-tolerance", type=float, default=1e-9)
    parser.add_argument("--amplitude-tolerance", type=float, default=1e-9)
    parser.add_argument("--phase-tolerance", type=float, default=1e-9)
    parser.add_argument("--effective-direct-tolerance", type=float, default=1e-9)
    parser.add_argument("--effective-residual-tolerance", type=float, default=1e-9)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-9)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--observable-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if any(nk <= 0 for nk in args.nks):
        parser.error("all --nks values must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not np.isfinite([args.qx, args.qy]).all() or np.hypot(args.qx, args.qy) == 0.0:
        parser.error("(qx, qy) must be finite and nonzero")
    if not np.isfinite(args.condition_max) or args.condition_max <= 0.0:
        parser.error("--condition-max must be finite and positive")
    for name in (
        "mixed_ward_tolerance",
        "mixed_ward_absolute_tolerance",
        "primitive_tolerance",
        "amplitude_tolerance",
        "phase_tolerance",
        "effective_direct_tolerance",
        "effective_residual_tolerance",
        "longitudinal_tolerance",
        "observable_relative_tolerance",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")

    common = {
        key: value
        for key, value in vars(args).items()
        if key not in {"nks", "workers", "output", "observable_relative_tolerance"}
    }
    tasks = [{**common, "nk": nk} for nk in sorted(set(args.nks))]
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
                    f"completed nk={row['nk']} in {row['wall_seconds']:.3f} s",
                    flush=True,
                )

    _add_convergence_metrics(rows)
    _write_csv(args.output, rows)
    finest_pair_converged = bool(
        len(rows) >= 2
        and float(rows[-1]["chi_bar_relative_to_previous"])
        <= args.observable_relative_tolerance
        and float(rows[-1]["dbar_t_relative_to_previous"])
        <= args.observable_relative_tolerance
    )
    payload = {
        "schema": "dwave_bond_metric_fixed_q_nk_convergence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "convergence_gate": {
            "all_strict_gates_passed": all(
                bool(row["strict_gate_passed"]) for row in rows
            ),
            "finest_pair_observable_converged": finest_pair_converged,
            "observable_relative_tolerance": args.observable_relative_tolerance,
            "passed": bool(
                all(bool(row["strict_gate_passed"]) for row in rows)
                and finest_pair_converged
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
    summary = _summary(rows, args.observable_relative_tolerance)
    args.output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"CSV:     {args.output}")
    print(f"JSON:    {args.output.with_suffix('.json')}")
    print(f"summary: {args.output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
