"""Run a reduced exact-static d-wave phase-column commensurate audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)
from validation.lib.dwave_iterated_adaptive import build_dwave_static_integrand_context
from validation.lib.dwave_phase_column_commensurate import (
    DWavePhaseColumnContext,
    assemble_phase_column_result,
    phase_column_result_as_audit_payload,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.run_dwave_static_commensurate_periodic_audit import _jsonable


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw/"
    "dwave_phase_column_commensurate_n942_m3_2_T10.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=942)
    parser.add_argument("--mx", type=int, default=3)
    parser.add_argument("--my", type=int, default=2)
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
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
    q = grid.q_model
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    full_context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(),
    )
    context = DWavePhaseColumnContext(full_context)

    print(
        "starting reduced commensurate phase-column audit: "
        f"nk={grid.nk}, m=({grid.mx},{grid.my}), "
        f"q=({q[0]:.12g},{q[1]:.12g}), points={grid.num_points}",
        flush=True,
    )
    print(
        "q/2 translation permutation exact = "
        f"{grid.half_translation_permutation_exact}",
        flush=True,
    )
    if not grid.half_translation_permutation_exact:
        print(
            "WARNING: k +/- q/2 lie on a complementary half-step sublattice; "
            "a single-origin comparison with the q=0 counterterm can contain an "
            "odd/even grid-origin alias. Use even integer shifts or average the "
            "required complementary grid origins before interpreting a residual.",
            flush=True,
        )

    integral = integrate_commensurate_periodic_vector(
        grid, context.evaluate_complex, chunk_size=args.chunk_size
    )
    result = assemble_phase_column_result(context, integral.value)
    payload = phase_column_result_as_audit_payload(
        result,
        metadata={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nk": grid.nk,
            "mx": grid.mx,
            "my": grid.my,
            "grid_shift": (grid.shift_x, grid.shift_y),
            "point_evaluations": integral.point_evaluations,
            "chunks": integral.chunks,
            "chunk_size": integral.chunk_size,
            "wall_seconds": integral.wall_seconds,
            "summation_method": integral.summation_method,
            "translation_by_q_is_exact_index_permutation": True,
            "translation_by_half_q_is_exact_index_permutation": (
                grid.half_translation_permutation_exact
            ),
            "half_q_sublattice_offset": grid.half_translation_sublattice_offset,
            "single_origin_half_q_alias_risk": (
                not grid.half_translation_permutation_exact
            ),
            "full_48_channel_integral_repeated": False,
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    q_norm = result.q_norm
    required = 0.5 * (
        result.left_required_counterterm_multiplier
        + result.right_required_counterterm_multiplier
    )
    print("d-wave reduced exact-static phase-column audit")
    print("=" * 50)
    print(f"grid = {grid.nk} x {grid.nk}; integer shift = ({grid.mx}, {grid.my})")
    print(f"q = ({q[0]:.12g}, {q[1]:.12g}); |q| = {q_norm:.12g}")
    print(f"points = {integral.point_evaluations}; chunks = {integral.chunks}")
    print(f"wall time = {integral.wall_seconds:.3f} s")
    print(
        "q/2 translation permutation exact = "
        f"{grid.half_translation_permutation_exact}"
    )
    print("")
    print(f"current phase defect / |q| = {max(abs(result.left_phase_defect), abs(result.right_phase_defect)) / q_norm:.12e}")
    print(f"required multiplier        = {required.real:+.12e}{required.imag:+.12e}j")
    print(f"required shift             = {abs(1.0 - required):.12e}")
    print(f"bond metric multiplier     = {result.bond_metric_multiplier:.12e}")
    print(f"bond multiplier error      = {abs(result.bond_metric_multiplier - required):.12e}")
    print(f"bond defect / |q|          = {max(abs(result.left_bond_metric_defect), abs(result.right_bond_metric_defect)) / q_norm:.12e}")
    print("")
    print("Fail-closed status")
    print("------------------")
    print(f"half_q_grid_compatible = {grid.half_translation_permutation_exact}")
    print("reduced_phase_column_only = True")
    print("diagnostic_only = True")
    print("projection_applied = False")
    print("production_reference_established = False")
    print("valid_for_casimir_input = False")
    print(f"output = {args.output}")


if __name__ == "__main__":
    main()
