"""Fixed-q d-wave convergence with primitive-before-Schur shift averaging.

A single uniform tensor-lattice origin can alias the nodal d-wave integrand at
arbitrary non-commensurate q.  This command evaluates deterministic nested
C4/antithetic periodic shifts and averages all primitive response blocks and the
analytic Ward RHS before forming one amplitude/phase Schur complement.

The nearest-neighbour bond phase-Hessian policy is applied exactly once after
that primitive merge.  All outputs remain diagnostic-only and invalid for
Casimir input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np

from lno327 import KuboConfig
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.phase_hessian import apply_phase_hessian_policy_to_components
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.dwave_periodic_shift_ensemble import (
    merge_shift_components_before_schur,
    nested_c4_antithetic_shifts,
    periodic_shift_mesh,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/static_nk_convergence/raw/"
    "dwave_bond_metric_fixed_q_shift_convergence.csv"
)


def _relative_difference(left: float, right: float) -> float:
    return float(
        abs(float(left) - float(right))
        / max(abs(float(left)), abs(float(right)), 1e-30)
    )


def _run_one(task: dict[str, Any]) -> list[dict[str, Any]]:
    nk = int(task["nk"])
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    shift_counts = tuple(int(value) for value in task["shift_counts"])
    max_shifts = max(shift_counts)
    started = time.perf_counter()

    if nk * nk * max_shifts > int(task["max_quadrature_points"]):
        raise RuntimeError(
            "periodic shift ensemble exceeded max_quadrature_points: "
            f"requested={nk * nk * max_shifts}, "
            f"maximum={int(task['max_quadrature_points'])}"
        )

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )
    # Merge the q-independent primitive counterterm first; apply the bond metric
    # exactly once after the complete shift ensemble has been merged.
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    shifts = nested_c4_antithetic_shifts(max_shifts)

    components_values = []
    rhs_values = []
    template_workspace = None
    per_shift_wall_seconds: list[float] = []
    warning_messages: list[str] = []

    for shift in shifts:
        shift_started = time.perf_counter()
        points, weights = periodic_shift_mesh(nk, shift)
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
        components_values.append(components)
        rhs_values.append(primitive_ward_rhs_from_q_workspace(workspace, 0.0))
        if template_workspace is None:
            template_workspace = workspace
        warning_messages.extend(str(item.message) for item in caught)
        per_shift_wall_seconds.append(float(time.perf_counter() - shift_started))

    assert template_workspace is not None
    rows: list[dict[str, Any]] = []
    for shift_count in shift_counts:
        merged_base, merged_rhs = merge_shift_components_before_schur(
            components_values[:shift_count],
            rhs_values[:shift_count],
            np.ones(shift_count, dtype=float),
            template_workspace,
            omega_eV=0.0,
        )
        corrected, application = apply_phase_hessian_policy_to_components(
            merged_base,
            ansatz,
            q,
            "nearest_neighbor_bond_metric",
            condition_threshold=task["condition_max"],
        )
        kernel = effective_em_kernel_from_components(corrected, q_model=q, xi_eV=0.0)
        ward = validate_effective_ward_xy(
            kernel,
            merged_rhs,
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
        rows.append(
            {
                "nk": nk,
                "shift_count": shift_count,
                "num_k_points_per_shift": nk * nk,
                "total_quadrature_points": nk * nk * shift_count,
                "qx": float(q[0]),
                "qy": float(q[1]),
                "q_norm": float(np.linalg.norm(q)),
                "shift_sequence_kind": "halton_bases_2_3_with_c4_antithetic_orbits",
                "primitive_merged_before_schur": True,
                "ward_rhs_merged_before_validation": True,
                "phase_hessian_policy": application.policy,
                "phase_hessian_multiplier": application.multiplier,
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
                "shift_prefix_wall_seconds": float(sum(per_shift_wall_seconds[:shift_count])),
                "full_nk_wall_seconds": float(time.perf_counter() - started),
                "warning_count": len(warning_messages),
                "warning_first": warning_messages[0] if warning_messages else "",
                "pid": os.getpid(),
                "diagnostic_only": True,
                "production_reference_established": False,
                "valid_for_casimir_input": False,
            }
        )
    return rows


def _add_convergence_metrics(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: (int(row["nk"]), int(row["shift_count"])))
    by_nk: dict[int, list[dict[str, Any]]] = {}
    by_shift: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_nk.setdefault(int(row["nk"]), []).append(row)
        by_shift.setdefault(int(row["shift_count"]), []).append(row)

    for group in by_nk.values():
        group.sort(key=lambda row: int(row["shift_count"]))
        previous = None
        for row in group:
            if previous is None:
                row["chi_bar_relative_to_previous_shift_prefix"] = float("nan")
                row["dbar_t_relative_to_previous_shift_prefix"] = float("nan")
            else:
                row["chi_bar_relative_to_previous_shift_prefix"] = _relative_difference(
                    row["chi_bar"], previous["chi_bar"]
                )
                row["dbar_t_relative_to_previous_shift_prefix"] = _relative_difference(
                    row["dbar_t"], previous["dbar_t"]
                )
            previous = row

    for group in by_shift.values():
        group.sort(key=lambda row: int(row["nk"]))
        previous = None
        for row in group:
            if previous is None:
                row["chi_bar_relative_to_previous_nk"] = float("nan")
                row["dbar_t_relative_to_previous_nk"] = float("nan")
            else:
                row["chi_bar_relative_to_previous_nk"] = _relative_difference(
                    row["chi_bar"], previous["chi_bar"]
                )
                row["dbar_t_relative_to_previous_nk"] = _relative_difference(
                    row["dbar_t"], previous["dbar_t"]
                )
            previous = row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], tolerance: float) -> str:
    lines = [
        "d-wave fixed-q periodic-shift convergence",
        "=" * 48,
        " nk shifts total-points phase/q    eff-direct/q longitudinal chi_bar    Dbar_T     strict",
        "-" * 105,
    ]
    for row in rows:
        lines.append(
            f"{int(row['nk']):3d} {int(row['shift_count']):6d} "
            f"{int(row['total_quadrature_points']):12d} "
            f"{float(row['phase_defect_over_q']):10.3e} "
            f"{float(row['effective_direct_over_q']):13.3e} "
            f"{float(row['relative_longitudinal_gauge_residual']):12.3e} "
            f"{float(row['chi_bar']):10.5e} {float(row['dbar_t']):10.5e} "
            f"{str(bool(row['strict_gate_passed'])):>7s}"
        )

    finest_nk = max(int(row["nk"]) for row in rows)
    finest_rows = [row for row in rows if int(row["nk"]) == finest_nk]
    finest = max(finest_rows, key=lambda row: int(row["shift_count"]))
    prefix_passed = bool(
        np.isfinite(float(finest["chi_bar_relative_to_previous_shift_prefix"]))
        and np.isfinite(float(finest["dbar_t_relative_to_previous_shift_prefix"]))
        and float(finest["chi_bar_relative_to_previous_shift_prefix"]) <= tolerance
        and float(finest["dbar_t_relative_to_previous_shift_prefix"]) <= tolerance
    )
    nk_passed = bool(
        np.isfinite(float(finest["chi_bar_relative_to_previous_nk"]))
        and np.isfinite(float(finest["dbar_t_relative_to_previous_nk"]))
        and float(finest["chi_bar_relative_to_previous_nk"]) <= tolerance
        and float(finest["dbar_t_relative_to_previous_nk"]) <= tolerance
    )
    lines.extend(
        [
            "",
            f"finest shift-prefix convergence passed = {prefix_passed}",
            f"finest nk convergence at max shifts passed = {nk_passed}",
            f"observable relative tolerance = {tolerance:.3e}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", type=int, nargs="+", required=True)
    parser.add_argument("--shift-counts", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-quadrature-points", type=int, default=1_000_000)
    parser.add_argument("--qx", type=float, default=0.0300152164356)
    parser.add_argument("--qy", type=float, default=0.0200101442904)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--mixed-ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixed-ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--primitive-tolerance", type=float, default=1e-6)
    parser.add_argument("--amplitude-tolerance", type=float, default=1e-6)
    parser.add_argument("--phase-tolerance", type=float, default=1e-6)
    parser.add_argument("--effective-direct-tolerance", type=float, default=1e-6)
    parser.add_argument("--effective-residual-tolerance", type=float, default=1e-6)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-6)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--observable-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if any(nk <= 0 for nk in args.nks):
        parser.error("all --nks values must be positive")
    counts = sorted(set(int(value) for value in args.shift_counts))
    if not counts or any(value <= 0 or value % 4 != 0 for value in counts):
        parser.error("all --shift-counts values must be positive multiples of four")
    if args.workers <= 0 or args.max_quadrature_points <= 0:
        parser.error("--workers and --max-quadrature-points must be positive")
    if not np.isfinite([args.qx, args.qy]).all() or np.hypot(args.qx, args.qy) == 0.0:
        parser.error("(qx, qy) must be finite and nonzero")

    common = {
        key: value
        for key, value in vars(args).items()
        if key not in {"nks", "workers", "output", "observable_relative_tolerance"}
    }
    common["shift_counts"] = counts
    tasks = [{**common, "nk": nk} for nk in sorted(set(args.nks))]
    if args.workers == 1:
        nested_rows = [_run_one(task) for task in tasks]
    else:
        nested_rows = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {executor.submit(_run_one, task): task["nk"] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                nested_rows.append(result)
                print(f"completed nk={futures[future]}", flush=True)

    rows = [row for group in nested_rows for row in group]
    _add_convergence_metrics(rows)
    _write_csv(args.output, rows)
    summary = _summary(rows, args.observable_relative_tolerance)
    args.output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_bond_metric_fixed_q_shift_convergence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "status": {
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary, end="")
    print(f"CSV:     {args.output}")
    print(f"JSON:    {args.output.with_suffix('.json')}")
    print(f"summary: {args.output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
