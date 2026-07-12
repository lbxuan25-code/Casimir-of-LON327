"""Equal-cost zero-Matsubara comparison of midpoint and multi-shift quadrature.

For every ``base_nk`` this CLI compares two integrations with the same total
number of k points:

* a conventional midpoint mesh with ``nk = 2 * base_nk``;
* a composite two-node Gauss-Legendre rule in each base cell, represented as
  four symmetry-related periodic grid shifts with ``nk = base_nk``.

All shifted points and normalized weights are concatenated before the material
workspace is built. Primitive EM/collective blocks and the Goldstone
counterterm are therefore integrated first, followed by one effective-kernel
Schur complement. Effective kernels from separate shifts are never averaged.
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
from itertools import product
from pathlib import Path
from typing import Any, Literal

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
from validation.commands.static.nk_scan import (
    _collective_channel_diagnostics,
    _kll_decomposition_diagnostics,
    _longitudinal_component_diagnostics,
    _phase_channel_factor_diagnostics,
    _ward_side_diagnostics,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

QuadratureRule = Literal["midpoint", "gauss2_shift4"]
DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/static_nk_convergence/raw/"
    "static_quadrature_compare.csv"
)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _periodic_shift_mesh(nk: int, shift_x: float, shift_y: float) -> np.ndarray:
    """Return one periodic tensor grid with fractional cell shifts in ``[0, 1)``."""

    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    shifts = np.asarray([shift_x, shift_y], dtype=float)
    if not np.isfinite(shifts).all() or np.any(shifts < 0.0) or np.any(shifts >= 1.0):
        raise ValueError("grid shifts must be finite and lie in [0, 1)")
    step = 2.0 * np.pi / float(nk)
    kx = -np.pi + (np.arange(nk, dtype=float) + float(shift_x)) * step
    ky = -np.pi + (np.arange(nk, dtype=float) + float(shift_y)) * step
    gx, gy = np.meshgrid(kx, ky, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def _quadrature_points(
    nk: int,
    rule: QuadratureRule,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one normalized BZ quadrature before any response/Schur operation."""

    nk = int(nk)
    if nk <= 0:
        raise ValueError("nk must be positive")

    if rule == "midpoint":
        points = uniform_bz_mesh(nk)
        weights = k_weights(points)
        shifts = ((0.5, 0.5),)
        description = "uniform midpoint tensor grid"
    elif rule == "gauss2_shift4":
        delta = 1.0 / (2.0 * np.sqrt(3.0))
        nodes = (0.5 - delta, 0.5 + delta)
        shifts = tuple((float(x), float(y)) for x, y in product(nodes, repeat=2))
        points = np.concatenate(
            [_periodic_shift_mesh(nk, shift_x, shift_y) for shift_x, shift_y in shifts],
            axis=0,
        )
        weights = np.full(points.shape[0], 1.0 / float(points.shape[0]), dtype=float)
        description = "composite 2x2 Gauss-Legendre rule as four periodic shifts"
    else:
        raise ValueError(f"unsupported quadrature rule {rule!r}")

    if points.ndim != 2 or points.shape[1] != 2:
        raise RuntimeError("quadrature points must have shape (N, 2)")
    if weights.shape != (points.shape[0],):
        raise RuntimeError("quadrature weights must have shape (N,)")
    if not np.all(weights > 0.0):
        raise RuntimeError("quadrature weights must be positive")
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 1e-12:
        raise RuntimeError("quadrature weights must sum to one")
    if np.any(points < -np.pi) or np.any(points >= np.pi):
        raise RuntimeError("periodic quadrature points must lie in [-pi, pi)")

    metadata = {
        "quadrature_rule": rule,
        "quadrature_description": description,
        "base_nk": nk,
        "num_grid_shifts": len(shifts),
        "grid_shifts": [[float(x), float(y)] for x, y in shifts],
        "num_k_points": int(points.shape[0]),
        "weight_sum": weight_sum,
        "primitive_merge_before_schur": True,
        "primitive_merge_method": "concatenate_points_with_normalized_weights",
    }
    return points, weights, metadata


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    base_nk = int(task["base_nk"])
    rule = str(task["quadrature_rule"])
    cell_nk = 2 * base_nk if rule == "midpoint" else base_nk
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    cpu_start = time.process_time()
    total_start = time.perf_counter()

    points, weights, quadrature = _quadrature_points(cell_nk, rule)  # type: ignore[arg-type]
    expected_points = 4 * base_nk * base_nk
    if int(points.shape[0]) != expected_points:
        raise RuntimeError("equal-cost quadrature pair does not have 4 * base_nk**2 points")

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(task["pairing"], phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
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

    longitudinal = _longitudinal_component_diagnostics(
        static.kernel_lt,
        static.energy_scale_eV,
    )
    transform = np.asarray(static.metadata["local_projection_matrix"], dtype=float)
    static_scale = float(longitudinal["static_kernel_real_scale"])
    kll = _kll_decomposition_diagnostics(
        components,
        kernel,
        transform,
        static.energy_scale_eV,
        static_scale,
    )
    collective = _collective_channel_diagnostics(
        kernel,
        transform,
        static.energy_scale_eV,
        static_scale,
    )
    phase = _phase_channel_factor_diagnostics(
        components,
        kernel,
        transform,
        static.energy_scale_eV,
        static_scale,
    )
    ward_left = _ward_side_diagnostics(ward.left, "ward_left")
    ward_right = _ward_side_diagnostics(ward.right, "ward_right")

    if not np.isclose(
        longitudinal["longitudinal_components_relative_norm"],
        static.validation.relative_longitudinal_gauge_residual,
        rtol=5e-13,
        atol=5e-15,
    ):
        raise RuntimeError("longitudinal decomposition disagrees with static validation")
    if not np.isclose(
        kll["scaled_kll_effective_relative_abs"],
        longitudinal["relative_kll"],
        rtol=5e-13,
        atol=5e-15,
    ):
        raise RuntimeError("K_LL decomposition disagrees with longitudinal diagnostics")
    if not np.isclose(
        collective["scaled_kll_channel_sum_real"],
        kll["scaled_kll_collective_correction_real"],
        rtol=5e-13,
        atol=5e-15,
    ):
        raise RuntimeError("collective channel sum disagrees with K_LL correction")
    if not np.isclose(
        phase["scaled_phase_factorized_correction_real"],
        collective["scaled_kll_channel_eta2_phase_eta2_phase_real"],
        rtol=5e-13,
        atol=5e-15,
    ):
        raise RuntimeError("factorized phase correction disagrees with eta2 term")

    total_seconds = time.perf_counter() - total_start
    warning_messages = [str(item.message) for item in caught]
    return {
        "comparison_base_nk": base_nk,
        "quadrature_rule": rule,
        "cell_nk": cell_nk,
        **quadrature,
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
        "process_cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_mb": _peak_rss_mb(),
        "midpoint_eigensystems": int(material.metadata["midpoint_eigensystem_count"]),
        "shifted_eigensystems": int(q_workspace.metadata["shifted_eigensystem_count"]),
        "ward_left_primitive": ward.left.primitive_relative_residual,
        "ward_right_primitive": ward.right.primitive_relative_residual,
        "ward_left_effective": ward.left.effective_relative_residual,
        "ward_right_effective": ward.right.effective_relative_residual,
        "schur_condition_number": ward.schur_condition_number,
        "ward_passed": bool(ward.passed),
        **ward_left,
        **ward_right,
        "relative_imaginary_norm": static.validation.relative_imaginary_norm,
        "relative_longitudinal_gauge_residual": (
            static.validation.relative_longitudinal_gauge_residual
        ),
        **longitudinal,
        **kll,
        **collective,
        **phase,
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
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "workers": args.workers,
        "comparison_contract": (
            "midpoint uses cell_nk=2*base_nk; gauss2_shift4 uses cell_nk=base_nk; "
            "both have 4*base_nk**2 points; all primitive blocks are merged before one Schur"
        ),
        "rows": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("Equal-cost quadrature comparison (primitive merge before one Schur)")
    header = (
        " base  rule             cell_nk       Nk  total[s]  longitudinal  "
        "rel(KLL)      Re(KSS)     Re(Kcoll)    Re(Keff)     chi_bar      Dbar_T"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['comparison_base_nk']:5d} "
            f"{row['quadrature_rule']:<16s} "
            f"{row['cell_nk']:7d} "
            f"{row['num_k_points']:8d} "
            f"{row['total_wall_seconds']:9.3f} "
            f"{row['relative_longitudinal_gauge_residual']:13.3e} "
            f"{row['relative_kll']:10.3e} "
            f"{row['scaled_kll_bare_total_real']:12.5e} "
            f"{row['scaled_kll_collective_correction_real']:12.5e} "
            f"{row['scaled_kll_effective_real']:12.5e} "
            f"{row['chi_bar']:12.5e} "
            f"{row['dbar_t']:12.5e}"
        )

    print("\nPhase factors")
    header = (
        " base  rule                 |K_L2|       Re(K22)     Re(inv22)   "
        "Re(corr22)  B+C cancel  Ward-eff(max)"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ward_eff = max(row["ward_left_effective"], row["ward_right_effective"])
        print(
            f"{row['comparison_base_nk']:5d} "
            f"{row['quadrature_rule']:<20s} "
            f"{row['raw_phase_left_coupling_abs']:12.4e} "
            f"{row['raw_phase_collective_total_real']:12.4e} "
            f"{row['raw_phase_inverse_22_real']:12.4e} "
            f"{row['scaled_phase_factorized_correction_real']:12.4e} "
            f"{row['phase_bubble_counterterm_cancellation_ratio']:10.3e} "
            f"{ward_eff:13.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nks", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if any(nk <= 0 for nk in args.base_nks):
        parser.error("all --base-nks values must be positive")
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
    tasks = [
        {**common, "base_nk": base_nk, "quadrature_rule": rule}
        for base_nk in sorted(set(args.base_nks))
        for rule in ("midpoint", "gauss2_shift4")
    ]

    sweep_start = time.perf_counter()
    if args.workers == 1:
        rows = [_run_one(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {executor.submit(_run_one, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"completed base={row['comparison_base_nk']} "
                    f"rule={row['quadrature_rule']} in {row['total_wall_seconds']:.3f} s "
                    f"(peak RSS {row['peak_rss_mb']:.1f} MiB)",
                    flush=True,
                )
        rows.sort(key=lambda row: (row["comparison_base_nk"], row["quadrature_rule"]))

    sweep_seconds = time.perf_counter() - sweep_start
    _write_outputs(rows, args.output, args)
    _print_summary(rows)
    print(f"\nSweep wall time: {sweep_seconds:.4f} s")
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
