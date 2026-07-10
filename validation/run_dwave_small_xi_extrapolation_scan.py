"""Complete-periodic small-positive-frequency extrapolation for d-wave response.

The runner evaluates one fixed periodic ``nk x nk`` lattice at exact ``xi=0`` and
at a user-supplied set of positive imaginary-frequency energies.  Positive
frequencies share one material and q workspace.  The real local-LT kernel entries
``-K_00`` and ``-K_TT`` are extrapolated to ``xi -> 0+`` without dividing by
frequency or interpreting them as a zero-mode sheet response.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lno327 import KuboConfig
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_responses_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.dwave_periodic_shift_ensemble import periodic_shift_mesh
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_global_extrapolation import (
    local_lt_kernel_proxies,
    relative_difference,
    small_xi_fits,
    summarize_fit_ensemble,
)
from validation.lib.dwave_shift_batch import ShiftBatchConfig, postprocess_merged
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_small_xi/raw/"
    "dwave_q003_002_nk224_small_xi.csv"
)


def _physical_config(args: argparse.Namespace) -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=args.nk,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
        raw_longitudinal_ceiling=args.raw_longitudinal_ceiling,
        longitudinal_tolerance=args.longitudinal_tolerance,
        mixing_tolerance=args.mixing_tolerance,
        reality_tolerance=args.reality_tolerance,
        passivity_tolerance=args.passivity_tolerance,
        separation_nm=args.separation_nm,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_reference(
    path: Path | None, *, allow_unconverged: bool
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    estimate = payload.get("reference_estimate")
    status = payload.get("reference_status")
    if not isinstance(estimate, dict) or not isinstance(status, dict):
        raise ValueError("reference JSON must contain reference_estimate and reference_status")
    converged = bool(status.get("numerical_reference_converged", False))
    if not converged and not allow_unconverged:
        raise ValueError(
            "reference JSON is not numerically converged; use --allow-unconverged-reference "
            "only for explicitly diagnostic comparisons"
        )
    result = {
        "chi_bar": float(estimate["chi_bar"]),
        "dbar_t": float(estimate["dbar_t"]),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("reference estimate contains non-finite values")
    return result, dict(status)


def _fit_channel(
    xi_values: np.ndarray,
    values: list[float],
    field: str,
    tail_sizes: tuple[int, ...],
    same_grid_static: float,
    external_reference: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fits = small_xi_fits(xi_values, values, tail_sizes=tail_sizes)
    for row in fits:
        row["field"] = field
        row["relative_intercept_to_same_grid_static"] = relative_difference(
            row["intercept"], same_grid_static
        )
        row["relative_intercept_to_external_reference"] = (
            relative_difference(row["intercept"], external_reference)
            if external_reference is not None
            else float("nan")
        )
    summary = summarize_fit_ensemble(fits)
    result = {
        "estimate": summary.estimate,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
        "relative_spread": summary.relative_spread,
        "best_model": summary.best_model,
        "best_tail_points": summary.best_tail_points,
        "best_normalized_rms": summary.best_normalized_rms,
        "num_accepted_models": summary.num_accepted_models,
        "relative_to_same_grid_static": relative_difference(
            summary.estimate, same_grid_static
        ),
        "relative_to_external_reference": (
            relative_difference(summary.estimate, external_reference)
            if external_reference is not None
            else float("nan")
        ),
    }
    return fits, result


def _summary_text(
    args: argparse.Namespace,
    static: Mapping[str, Any],
    rows: list[dict[str, Any]],
    fits: dict[str, Any],
    external_reference: dict[str, float] | None,
    external_status: dict[str, Any] | None,
    same_grid_pass: bool,
    external_pass: bool | None,
    production_candidate: bool,
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave complete-periodic small-xi extrapolation scan",
        "=" * 56,
        f"q = ({args.qx:.8g}, {args.qy:.8g}); nk = {args.nk}; "
        f"shift = ({args.shift[0]:.8g}, {args.shift[1]:.8g})",
        f"T = {args.temperature_K:.8g} K; wall time = {wall_seconds:.3f} s",
        "",
        "Exact-static result on the same grid",
        "------------------------------------",
        f"chi_bar={float(static['chi_bar']):.10g}, Dbar_T={float(static['dbar_t']):.10g}, "
        f"Ward_prim={float(static['ward_primitive_mixed_ratio_max']):.3e}, "
        f"raw_long={float(static['raw_longitudinal']):.3e}, "
        f"projection={bool(static['projection_eligible'])}",
        "",
        "Positive-frequency kernel proxies",
        "---------------------------------",
        " xi[eV]      chi-proxy    D_T-proxy   rChi-static rD-static Ward-prim cond(Keta)",
    ]
    for row in rows:
        lines.append(
            f"{float(row['xi_eV']):11.4e} "
            f"{float(row['chi_bar_proxy']):12.8f} {float(row['dbar_t_proxy']):12.8f} "
            f"{float(row['relative_chi_to_same_grid_static']):11.3e} "
            f"{float(row['relative_dbar_to_same_grid_static']):10.3e} "
            f"{float(row['ward_primitive_mixed_ratio_max']):10.3e} "
            f"{float(row['schur_condition_number']):10.3e}"
        )
    lines.extend(["", "Zero-frequency fit ensemble", "---------------------------"])
    for field in ("chi_bar_proxy", "dbar_t_proxy"):
        value = fits[field]
        lines.append(
            f"{field}: estimate={value['estimate']:.10g}, interval=[{value['minimum']:.10g}, "
            f"{value['maximum']:.10g}], spread={value['relative_spread']:.3e}, "
            f"same-grid residual={value['relative_to_same_grid_static']:.3e}, "
            f"external residual={value['relative_to_external_reference']:.3e}, "
            f"best={value['best_model']} tail={value['best_tail_points']}"
        )
    lines.extend(["", "Fail-closed continuity status", "-----------------------------"])
    lines.append(f"same_grid_continuity_pass = {same_grid_pass}")
    if external_reference is None:
        lines.append("external_reference_pass = unavailable")
        lines.append("external reference was not supplied; production_candidate remains false")
    else:
        lines.append(
            f"external reference: chi_bar={external_reference['chi_bar']:.10g}, "
            f"Dbar_T={external_reference['dbar_t']:.10g}"
        )
        lines.append(
            f"external_reference_numerically_converged = "
            f"{bool(external_status and external_status.get('numerical_reference_converged', False))}"
        )
        lines.append(f"external_reference_pass = {external_pass}")
    lines.extend(
        [
            f"candidate_for_production_small_xi_method = {production_candidate}",
            f"continuation_tolerance = {args.continuation_tolerance:.3e}",
            f"model_spread_tolerance = {args.model_spread_tolerance:.3e}",
            f"external_reference_tolerance = {args.external_reference_tolerance:.3e}",
            "",
            "Positive-frequency proxies are kernel-continuity diagnostics only. They are not "
            "zero-mode conductivities or zero-Matsubara reflections.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, required=True)
    parser.add_argument("--shift", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--xis-eV", type=float, nargs="+", required=True)
    parser.add_argument("--fit-tail-sizes", type=int, nargs="+", default=[4, 5, 6, 8])
    parser.add_argument("--max-evaluation-points", type=int, default=2_000_000)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--allow-unconverged-reference", action="store_true")
    parser.add_argument("--continuation-tolerance", type=float, default=1e-2)
    parser.add_argument("--model-spread-tolerance", type=float, default=1e-2)
    parser.add_argument("--external-reference-tolerance", type=float, default=1e-2)
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
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
    if args.nk <= 0:
        parser.error("--nk must be positive")
    shift = np.asarray(args.shift, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all() or np.any(shift < 0.0) or np.any(shift >= 1.0):
        parser.error("--shift coordinates must lie in [0,1)")
    xis = np.asarray(sorted(set(float(value) for value in args.xis_eV)), dtype=float)
    if xis.size < 4 or np.any(xis <= 0.0) or not np.isfinite(xis).all():
        parser.error("--xis-eV requires at least four distinct positive finite values")
    args.xis_eV = xis.tolist()
    requested = args.nk * args.nk * (1 + xis.size)
    if args.max_evaluation_points <= 0 or requested > args.max_evaluation_points:
        parser.error(
            f"requested frequency-weighted points {requested} exceed "
            f"--max-evaluation-points={args.max_evaluation_points}"
        )
    if any(
        value <= 0.0
        for value in (
            args.continuation_tolerance,
            args.model_spread_tolerance,
            args.external_reference_tolerance,
        )
    ):
        parser.error("continuity tolerances must be positive")
    if float(np.hypot(args.qx, args.qy)) == 0.0:
        parser.error("q must be nonzero")
    return args


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    external_reference, external_status = _load_reference(
        args.reference_json, allow_unconverged=args.allow_unconverged_reference
    )
    q = np.asarray([args.qx, args.qy], dtype=float)
    shift = np.asarray(args.shift, dtype=float)
    xis = np.asarray(args.xis_eV, dtype=float)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    points, weights = periodic_shift_mesh(args.nk, shift)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    material_started = time.perf_counter()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        kubo,
        pairing,
        FiniteQEngineOptions(),
    )
    material_seconds = time.perf_counter() - material_started
    q_started = time.perf_counter()
    workspace = precompute_finite_q_q_workspace(material, q)
    q_workspace_seconds = time.perf_counter() - q_started
    response_started = time.perf_counter()
    all_xis = np.concatenate([[0.0], xis])
    components = finite_q_bdg_responses_from_q_workspace(workspace, all_xis)
    response_seconds = time.perf_counter() - response_started

    physical = _physical_config(args)
    static_rhs = primitive_ward_rhs_from_q_workspace(workspace, 0.0)
    static = postprocess_merged(components[0], static_rhs, physical)
    rows: list[dict[str, Any]] = []
    for xi, component in zip(xis, components[1:], strict=True):
        kernel = effective_em_kernel_from_components(component, q_model=q, xi_eV=float(xi))
        rhs = primitive_ward_rhs_from_q_workspace(workspace, float(xi))
        ward = validate_effective_ward_xy(
            kernel,
            rhs,
            residual_tolerance=args.ward_tolerance,
            absolute_residual_tolerance=args.ward_absolute_tolerance,
            condition_max=args.condition_max,
        )
        proxies = local_lt_kernel_proxies(kernel, q)
        rows.append(
            {
                "nk": args.nk,
                "num_k_points": args.nk * args.nk,
                "shift_x": float(shift[0]),
                "shift_y": float(shift[1]),
                "xi_eV": float(xi),
                **proxies,
                "relative_chi_to_same_grid_static": relative_difference(
                    proxies["chi_bar_proxy"], static["chi_bar"]
                ),
                "relative_dbar_to_same_grid_static": relative_difference(
                    proxies["dbar_t_proxy"], static["dbar_t"]
                ),
                "ward_passed": bool(ward.passed),
                "ward_primitive_mixed_ratio_max": max(
                    ward.left.primitive_mixed_ratio, ward.right.primitive_mixed_ratio
                ),
                "ward_effective_mixed_ratio_max": max(
                    ward.left.effective_mixed_ratio, ward.right.effective_mixed_ratio
                ),
                "schur_condition_number": float(ward.schur_condition_number),
                "schur_inverse_method": ward.schur_inverse_method,
            }
        )
    rows.sort(key=lambda row: float(row["xi_eV"]))

    all_fits: list[dict[str, Any]] = []
    fit_summaries: dict[str, Any] = {}
    for field, static_field, reference_field in (
        ("chi_bar_proxy", "chi_bar", "chi_bar"),
        ("dbar_t_proxy", "dbar_t", "dbar_t"),
    ):
        fits, summary = _fit_channel(
            np.asarray([row["xi_eV"] for row in rows], dtype=float),
            [float(row[field]) for row in rows],
            field,
            tuple(int(value) for value in args.fit_tail_sizes),
            float(static[static_field]),
            external_reference[reference_field] if external_reference is not None else None,
        )
        all_fits.extend(fits)
        fit_summaries[field] = summary

    ward_ok = all(
        bool(row["ward_passed"])
        and str(row["schur_inverse_method"]) == "inv"
        and float(row["ward_primitive_mixed_ratio_max"]) < 1.0
        and float(row["ward_effective_mixed_ratio_max"]) < 1.0
        for row in rows
    )
    same_grid_residual = max(
        float(fit_summaries["chi_bar_proxy"]["relative_to_same_grid_static"]),
        float(fit_summaries["dbar_t_proxy"]["relative_to_same_grid_static"]),
    )
    model_spread = max(
        float(fit_summaries["chi_bar_proxy"]["relative_spread"]),
        float(fit_summaries["dbar_t_proxy"]["relative_spread"]),
    )
    same_grid_pass = bool(
        ward_ok
        and same_grid_residual <= args.continuation_tolerance
        and model_spread <= args.model_spread_tolerance
    )
    external_pass: bool | None = None
    if external_reference is not None:
        external_residual = max(
            float(fit_summaries["chi_bar_proxy"]["relative_to_external_reference"]),
            float(fit_summaries["dbar_t_proxy"]["relative_to_external_reference"]),
        )
        external_pass = bool(
            external_status
            and external_status.get("numerical_reference_converged", False)
            and external_residual <= args.external_reference_tolerance
        )
    production_candidate = bool(same_grid_pass and external_pass is True)
    wall_seconds = time.perf_counter() - started

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fits_path = output.with_name(output.stem + ".fits.csv")
    summary_path = output.with_name(output.stem + ".summary.txt")
    _write_csv(output, rows)
    _write_csv(fits_path, all_fits)
    summary = _summary_text(
        args,
        static,
        rows,
        fit_summaries,
        external_reference,
        external_status,
        same_grid_pass,
        external_pass,
        production_candidate,
        wall_seconds,
    )
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "one complete periodic lattice; exact xi=0 plus batched xi>0; kernel-level local-LT "
            "continuity proxies; no frequency division; Ward validated at every frequency"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "timing": {
            "material_seconds": material_seconds,
            "q_workspace_seconds": q_workspace_seconds,
            "response_batch_seconds": response_seconds,
            "wall_seconds": wall_seconds,
        },
        "same_grid_exact_static": static,
        "positive_frequency_rows": rows,
        "fits": all_fits,
        "fit_summaries": fit_summaries,
        "external_reference": external_reference,
        "external_reference_status": external_status,
        "continuity_status": {
            "all_positive_frequency_ward_ok": ward_ok,
            "same_grid_continuity_pass": same_grid_pass,
            "external_reference_pass": external_pass,
            "candidate_for_production_small_xi_method": production_candidate,
            "same_grid_relative_residual_max": same_grid_residual,
            "fit_relative_spread_max": model_spread,
        },
        "files": {
            "rows_csv": str(output),
            "fits_csv": str(fits_path),
            "summary_txt": str(summary_path),
        },
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + summary)
    print(f"CSV:     {output}")
    print(f"Fits:    {fits_path}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
