"""Exact-static d-wave Ward diagnostic on a q-commensurate periodic grid.

The external momentum is constructed from an integer lattice translation,
``q=(2*pi/nk)*(mx,my)``. All primitive response and analytic Ward-RHS channels
share the same complete periodic grid, and the amplitude/phase Schur is formed
only after the full primitive average. This command is diagnostic-only and does
not apply a longitudinal projection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.microscopic_model import get_finite_q_microscopic_model
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)
from validation.lib.dwave_static_primitives import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
)

DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw/"
    "dwave_commensurate.csv"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        scalar = complex(value)
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=628)
    parser.add_argument("--mx", type=int, default=3)
    parser.add_argument("--my", type=int, default=2)
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def main() -> None:
    args = _parse_args()
    grid = CommensuratePeriodicGrid(
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        max_points=args.max_points,
    )
    q = np.asarray(grid.q_model, dtype=float)
    model = get_finite_q_microscopic_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )

    integral = integrate_commensurate_periodic_vector(
        grid,
        context.evaluate_complex,
        chunk_size=args.chunk_size,
    )
    components, rhs, _ = assemble_dwave_static_primitives(
        context,
        integral.value,
        metadata={
            "integration_strategy": "commensurate_complete_periodic_tensor_grid",
            "integer_q_shift": (int(grid.mx), int(grid.my)),
            "translation_by_q_is_exact_index_permutation": True,
            "primitive_vector_integrated_before_schur": True,
        },
    )
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=args.ward_tolerance,
        absolute_residual_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
    )
    sheet = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=1.0,
        mixing_tolerance=1.0,
        reality_tolerance=1.0,
        passivity_tolerance=1.0,
    )

    row = {
        "nk": int(grid.nk),
        "mx": int(grid.mx),
        "my": int(grid.my),
        "shift_x": float(grid.shift_x),
        "shift_y": float(grid.shift_y),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": float(np.linalg.norm(q)),
        "translation_permutation_exact": bool(grid.translation_permutation_exact),
        "point_evaluations": int(integral.point_evaluations),
        "chunks": int(integral.chunks),
        "summation_method": str(integral.summation_method),
        "integration_wall_seconds": float(integral.wall_seconds),
        "ward_passed": bool(ward.passed),
        "ward_primitive_mixed_ratio_max": max(
            float(ward.left.primitive_mixed_ratio),
            float(ward.right.primitive_mixed_ratio),
        ),
        "ward_effective_mixed_ratio_max": max(
            float(ward.left.effective_mixed_ratio),
            float(ward.right.effective_mixed_ratio),
        ),
        "schur_condition_number": float(ward.schur_condition_number),
        "schur_inverse_method": str(ward.schur_inverse_method),
        "raw_longitudinal": float(sheet.validation.relative_longitudinal_gauge_residual),
        "chi_bar": float(sheet.chi_bar),
        "dbar_t": float(sheet.dbar_t),
        "diagnostic_only": True,
        "projection_applied": False,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    summary = "\n".join(
        [
            "d-wave commensurate periodic Ward diagnostic",
            "=" * 47,
            f"grid = {grid.nk} x {grid.nk}; shift = ({grid.mx}, {grid.my})",
            f"q = ({q[0]:.12g}, {q[1]:.12g})",
            f"points = {integral.point_evaluations}",
            f"ward_passed = {ward.passed}",
            f"primitive_mixed_ratio = {row['ward_primitive_mixed_ratio_max']:.6e}",
            f"effective_mixed_ratio = {row['ward_effective_mixed_ratio_max']:.6e}",
            f"raw_longitudinal = {row['raw_longitudinal']:.6e}",
            f"chi_bar = {row['chi_bar']:.10f}",
            f"Dbar_T = {row['dbar_t']:.10f}",
            "diagnostic_only = True",
            "valid_for_casimir_input = False",
            "",
        ]
    )
    output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_commensurate_ward_diagnostic_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": _jsonable(vars(args)),
        "row": _jsonable(row),
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary, end="")


if __name__ == "__main__":
    main()
