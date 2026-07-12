"""Exact-static d-wave validation on a complete q orbit and transverse Gauss rule.

The external momentum is constructed as ``q=(2 pi/nk)(mx,my)``.  An integer
unimodular torus basis aligns one coordinate with q, where a complete periodic
orbit preserves translation by q as an exact index permutation.  The orthogonal
torus coordinate uses a deterministic Gauss-Legendre rule.  Complete primitive
response blocks are integrated before the bond phase metric and collective Schur.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_adaptive_bond_metric import (
    AdaptiveStaticValidationConfig,
    postprocess_adaptive_bond_metric_static,
)
from validation.lib.dwave_commensurate_orbit_gauss import (
    OrbitEvaluationBudgetExceeded,
    integrate_commensurate_orbit_gauss_vector,
)
from validation.lib.dwave_static_primitives import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_static/raw/"
    "dwave_bond_metric_commensurate_orbit_gauss.csv"
)


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=628)
    parser.add_argument("--mx", type=int, default=3)
    parser.add_argument("--my", type=int, default=2)
    parser.add_argument("--transverse-order", type=int, default=64)
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument(
        "--subgrid-average",
        choices=("auto", "none"),
        default="auto",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-point-evaluations", type=int, default=100_000)
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
    parser.add_argument("--reality-tolerance", type=float, default=1e-8)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-6)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--energy-scale-eV", type=float, default=1.0)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.nk <= 0 or args.transverse_order <= 0:
        parser.error("--nk and --transverse-order must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if args.chunk_size <= 0 or args.max_point_evaluations <= 0:
        parser.error("--chunk-size and --max-point-evaluations must be positive")
    if not np.isfinite(args.shift_s) or not 0.0 <= args.shift_s < 1.0:
        parser.error("--shift-s must lie in [0,1)")
    return args


def _validation_config(args: argparse.Namespace) -> AdaptiveStaticValidationConfig:
    return AdaptiveStaticValidationConfig(
        mixed_ward_tolerance=args.mixed_ward_tolerance,
        mixed_ward_absolute_tolerance=args.mixed_ward_absolute_tolerance,
        primitive_tolerance=args.primitive_tolerance,
        amplitude_tolerance=args.amplitude_tolerance,
        phase_tolerance=args.phase_tolerance,
        effective_direct_tolerance=args.effective_direct_tolerance,
        effective_residual_tolerance=args.effective_residual_tolerance,
        longitudinal_tolerance=args.longitudinal_tolerance,
        condition_max=args.condition_max,
        reality_tolerance=args.reality_tolerance,
        mixing_tolerance=args.mixing_tolerance,
        passivity_tolerance=args.passivity_tolerance,
        energy_scale_eV=args.energy_scale_eV,
        degeneracy=args.degeneracy,
    )


def _summary_text(row: dict[str, Any], total_wall_seconds: float) -> str:
    return "\n".join(
        [
            "d-wave bond-metric commensurate-orbit transverse-Gauss validation",
            "=" * 72,
            f"grid q = (2 pi/{row['nk']}) ({row['mx']}, {row['my']})",
            f"q = ({row['qx']:.12g}, {row['qy']:.12g}); |q| = {row['q_norm']:.12g}",
            f"orbit basis p = {row['primitive_direction']}; transverse n = "
            f"{row['transverse_direction']}",
            f"orbit origins = {row['orbit_origins']}; transverse order = "
            f"{row['transverse_order']}",
            f"point evaluations = {row['point_evaluations']}; wall time = "
            f"{total_wall_seconds:.3f} s",
            "",
            "Closure and observables",
            "-----------------------",
            f"primitive/q = {row['primitive_residual_over_q']:.12e}",
            f"amplitude/q = {row['amplitude_defect_over_q']:.12e}",
            f"phase/q = {row['phase_defect_over_q']:.12e}",
            f"effective-direct/q = {row['effective_direct_over_q']:.12e}",
            f"effective-residual/q = {row['effective_residual_over_q']:.12e}",
            f"longitudinal = {row['relative_longitudinal_gauge_residual']:.12e}",
            f"strict static gate = {row['strict_gate_passed']}",
            f"sheet validation = {row['sheet_validation_passed']}",
            f"chi_bar = {row['chi_bar']:.12e}",
            f"Dbar_T = {row['dbar_t']:.12e}",
            "",
            "Fail-closed status",
            "------------------",
            "integration_strategy = commensurate_q_orbit_transverse_gauss",
            "projection_applied = False",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    q = (2.0 * np.pi / float(args.nk)) * np.asarray([args.mx, args.my], dtype=float)
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )

    print(
        "starting d-wave commensurate orbit: "
        f"nk={args.nk}, m=({args.mx},{args.my}), "
        f"transverse_order={args.transverse_order}",
        flush=True,
    )
    try:
        quadrature = integrate_commensurate_orbit_gauss_vector(
            context.evaluate_complex,
            nk=args.nk,
            mx=args.mx,
            my=args.my,
            transverse_order=args.transverse_order,
            shift_s=args.shift_s,
            subgrid_average=args.subgrid_average,
            chunk_size=args.chunk_size,
            max_point_evaluations=args.max_point_evaluations,
        )
    except OrbitEvaluationBudgetExceeded as exc:
        raise SystemExit(str(exc)) from exc

    components, rhs, _ = assemble_dwave_static_primitives(
        context,
        quadrature.value,
        metadata={
            "integration_strategy": "commensurate_q_orbit_transverse_gauss",
            "commensurate_nk": int(args.nk),
            "integer_q_shift": (int(args.mx), int(args.my)),
            "translation_by_q_is_exact_orbit_permutation": True,
            "primitive_direction": tuple(int(value) for value in quadrature.primitive_direction),
            "transverse_direction": tuple(
                int(value) for value in quadrature.transverse_direction
            ),
            "orbit_shift_steps": int(quadrature.orbit_shift_steps),
            "orbit_origins": tuple(float(value) for value in quadrature.orbit_origins),
            "transverse_order": int(args.transverse_order),
            "point_evaluations": int(quadrature.point_evaluations),
            "summation_method": quadrature.summation_method,
            "outer_discretization_error_estimated": False,
        },
    )
    processed = postprocess_adaptive_bond_metric_static(
        components,
        rhs,
        ansatz=ansatz,
        q_model=q,
        config=_validation_config(args),
    )
    row = {
        "nk": int(args.nk),
        "mx": int(args.mx),
        "my": int(args.my),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": float(np.linalg.norm(q)),
        "transverse_order": int(args.transverse_order),
        "shift_s": float(args.shift_s),
        "subgrid_average": str(args.subgrid_average),
        "orbit_origins": repr(quadrature.orbit_origins),
        "primitive_direction": repr(tuple(int(v) for v in quadrature.primitive_direction)),
        "transverse_direction": repr(
            tuple(int(v) for v in quadrature.transverse_direction)
        ),
        "orbit_shift_steps": int(quadrature.orbit_shift_steps),
        "point_evaluations": int(quadrature.point_evaluations),
        "chunks": int(quadrature.chunks),
        "chunk_size": int(quadrature.chunk_size),
        "quadrature_wall_seconds": float(quadrature.wall_seconds),
        "temperature_K": float(args.temperature_K),
        "delta0_eV": float(args.delta0_eV),
        "eta_eV": float(args.eta_eV),
        **processed.to_row_fields(),
    }
    total_wall = time.perf_counter() - started
    output = args.output
    _write_csv(output, row)
    summary_path = output.with_suffix(".summary.txt")
    json_path = output.with_suffix(".json")
    summary = _summary_text(row, total_wall)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_bond_metric_commensurate_orbit_gauss_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "row": row,
        "total_wall_seconds": total_wall,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"finished: points={quadrature.point_evaluations}, "
        f"chi={row['chi_bar']:.10g}, D_T={row['dbar_t']:.10g}, "
        f"phase/q={row['phase_defect_over_q']:.3e}, "
        f"longitudinal={row['relative_longitudinal_gauge_residual']:.3e}, "
        f"strict={row['strict_gate_passed']}",
        flush=True,
    )
    print()
    print(summary)
    print(f"CSV:     {output}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
