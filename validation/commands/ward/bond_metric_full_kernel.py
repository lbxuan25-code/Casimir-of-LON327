"""Canonical commensurate d-wave full-kernel bond-metric Ward audit.

The complete 48-component exact-static primitive vector is integrated on one or
more complementary periodic subgrids.  Primitive vectors are averaged before any
Schur complement.  The same integrated response is then evaluated with the q=0
Goldstone counterterm and with the diagnosed nearest-neighbour bond metric applied
only to the phase-phase HS entry.  No longitudinal projection is performed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from itertools import product
from pathlib import Path
import platform
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.ward.commensurate import _jsonable
from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)
from validation.lib.dwave_bond_phase_counterterm import (
    apply_nearest_neighbor_dwave_phase_counterterm,
)
from validation.lib.dwave_static_primitives import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.static_ward_component_sources import (
    audit_static_ward_contract_with_components,
)

DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw/"
    "dwave_bond_metric_full_kernel.csv"
)


@dataclass(frozen=True)
class SubgridAverageSummary:
    point_evaluations: int
    chunks: int
    chunk_size: int
    wall_seconds: float
    summation_method: str


def complementary_subgrid_origins(
    mx: int,
    my: int,
    shift_x: float,
    shift_y: float,
) -> tuple[tuple[float, float], ...]:
    """Return half-step partner origins required by odd integer q components."""

    sx, sy = float(shift_x), float(shift_y)
    if not np.isfinite([sx, sy]).all() or not (0.0 <= sx < 1.0 and 0.0 <= sy < 1.0):
        raise ValueError("grid shifts must be finite and lie in [0, 1)")
    xs = (sx,) if int(mx) % 2 == 0 else (sx, (sx + 0.5) % 1.0)
    ys = (sy,) if int(my) % 2 == 0 else (sy, (sy + 0.5) % 1.0)
    return tuple((float(x), float(y)) for x, y in product(xs, ys))


def _evaluate_components(
    components: Any,
    rhs: Any,
    q: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any]]:
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
    audit = audit_static_ward_contract_with_components(kernel, rhs, components)
    return kernel, ward, sheet, audit


def _max_collective_channel_defect_over_q(
    audit: dict[str, Any], channel: int
) -> float:
    q_norm = float(audit["q_norm"])
    return max(
        abs(complex(np.asarray(audit[side]["collective_defect"], dtype=complex)[channel]))
        / q_norm
        for side in ("left", "right")
    )


def _max_ward_effective_mixed_ratio(ward: Any) -> float:
    return max(
        float(ward.left.effective_mixed_ratio),
        float(ward.right.effective_mixed_ratio),
    )


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _summary_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "d-wave full-kernel bond phase metric audit",
            "=" * 45,
            f"grid = {row['nk']} x {row['nk']}; m = ({row['mx']}, {row['my']})",
            f"q = ({row['qx']:.12g}, {row['qy']:.12g}); |q| = {row['q_norm']:.12g}",
            f"subgrid origins = {row['subgrid_origins']}",
            f"points evaluated = {row['point_evaluations']}",
            "",
            f"bond metric multiplier = {row['bond_metric_multiplier']:.15e}",
            f"only phase diagonal changed = {row['counterterm_changed_only_22']}",
            "",
            f"baseline phase/q = {row['baseline_phase_defect_over_q']:.12e}",
            f"corrected phase/q = {row['corrected_phase_defect_over_q']:.12e}",
            f"baseline effective-direct/q = {row['baseline_effective_direct_over_q']:.12e}",
            f"corrected effective-direct/q = {row['corrected_effective_direct_over_q']:.12e}",
            f"baseline longitudinal = {row['baseline_raw_longitudinal']:.12e}",
            f"corrected longitudinal = {row['corrected_raw_longitudinal']:.12e}",
            f"baseline chi_bar = {row['baseline_chi_bar']:.12e}",
            f"corrected chi_bar = {row['corrected_chi_bar']:.12e}",
            f"baseline Dbar_T = {row['baseline_dbar_t']:.12e}",
            f"corrected Dbar_T = {row['corrected_dbar_t']:.12e}",
            "",
            "diagnostic_only = True",
            "projection_applied = False",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=628)
    parser.add_argument("--mx", type=int, default=3)
    parser.add_argument("--my", type=int, default=2)
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument(
        "--subgrid-average",
        choices=("auto", "none"),
        default="auto",
    )
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
    if args.nk <= 0 or args.chunk_size <= 0 or args.max_points <= 0:
        parser.error("--nk, --chunk-size, and --max-points must be positive")
    return args


def main() -> None:
    args = _parse_args()
    base_grid = CommensuratePeriodicGrid(
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        max_points=args.max_points,
    )
    q = np.asarray(base_grid.q_model, dtype=float)
    origins = (
        complementary_subgrid_origins(args.mx, args.my, args.shift_x, args.shift_y)
        if args.subgrid_average == "auto"
        else ((float(args.shift_x), float(args.shift_y)),)
    )

    model = get_finite_q_validation_model("symmetry_bdg_2band")
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

    values: list[np.ndarray] = []
    integrals: list[Any] = []
    for index, (shift_x, shift_y) in enumerate(origins, start=1):
        grid = CommensuratePeriodicGrid(
            nk=args.nk,
            mx=args.mx,
            my=args.my,
            shift_x=shift_x,
            shift_y=shift_y,
            max_points=args.max_points,
        )
        print(
            f"integrating subgrid {index}/{len(origins)}: "
            f"shift=({shift_x:.6g},{shift_y:.6g}), points={grid.num_points}",
            flush=True,
        )
        integral = integrate_commensurate_periodic_vector(
            grid,
            context.evaluate_complex,
            chunk_size=args.chunk_size,
        )
        values.append(np.asarray(integral.value, dtype=complex))
        integrals.append(integral)

    primitive = np.mean(np.stack(values, axis=0), axis=0)
    integration = SubgridAverageSummary(
        point_evaluations=sum(int(item.point_evaluations) for item in integrals),
        chunks=sum(int(item.chunks) for item in integrals),
        chunk_size=int(args.chunk_size),
        wall_seconds=sum(float(item.wall_seconds) for item in integrals),
        summation_method=(
            "componentwise_equal_average_of_complementary_subgrids_after_"
            + str(integrals[0].summation_method)
        ),
    )
    common_metadata = {
        "integration_strategy": "commensurate_complete_periodic_subgrid_average",
        "nk": int(args.nk),
        "integer_q_shift": (int(args.mx), int(args.my)),
        "translation_by_q_is_exact_index_permutation": True,
        "half_q_single_subgrid_compatible": bool(
            base_grid.half_translation_permutation_exact
        ),
        "complementary_subgrid_origins": origins,
        "componentwise_subgrid_average_before_schur": bool(len(origins) > 1),
        "subgrid_count": len(origins),
        "primitive_vector_averaged_before_schur": True,
        "summation_method": integration.summation_method,
    }
    baseline_components, rhs, primitive_metadata = assemble_dwave_static_primitives(
        context,
        primitive,
        metadata=common_metadata,
    )
    corrected_components, application = apply_nearest_neighbor_dwave_phase_counterterm(
        baseline_components,
        q,
        condition_threshold=args.condition_max,
    )

    baseline_kernel, baseline_ward, baseline_sheet, baseline_audit = _evaluate_components(
        baseline_components, rhs, q, args
    )
    corrected_kernel, corrected_ward, corrected_sheet, corrected_audit = _evaluate_components(
        corrected_components, rhs, q, args
    )

    base_22 = complex(application.base_counterterm[1, 1])
    applied_22 = complex(application.applied_counterterm[1, 1])
    row = {
        "nk": int(args.nk),
        "mx": int(args.mx),
        "my": int(args.my),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": float(np.linalg.norm(q)),
        "subgrid_average_mode": args.subgrid_average,
        "subgrid_averaged": bool(len(origins) > 1),
        "subgrid_count": len(origins),
        "subgrid_origins": repr(origins),
        "point_evaluations": integration.point_evaluations,
        "chunks": integration.chunks,
        "integration_wall_seconds": integration.wall_seconds,
        "summation_method": integration.summation_method,
        "bond_metric_multiplier": float(application.multiplier),
        "base_counterterm_22_real": float(base_22.real),
        "base_counterterm_22_imag": float(base_22.imag),
        "applied_counterterm_22_real": float(applied_22.real),
        "applied_counterterm_22_imag": float(applied_22.imag),
        "counterterm_changed_only_22": bool(
            corrected_components.metadata[
                "diagnostic_phase_counterterm_changed_only_22"
            ]
        ),
        "baseline_phase_defect_over_q": _max_collective_channel_defect_over_q(
            baseline_audit, 1
        ),
        "corrected_phase_defect_over_q": _max_collective_channel_defect_over_q(
            corrected_audit, 1
        ),
        "baseline_amplitude_defect_over_q": _max_collective_channel_defect_over_q(
            baseline_audit, 0
        ),
        "corrected_amplitude_defect_over_q": _max_collective_channel_defect_over_q(
            corrected_audit, 0
        ),
        "baseline_effective_direct_over_q": float(
            baseline_audit["max_effective_direct_over_q"]
        ),
        "corrected_effective_direct_over_q": float(
            corrected_audit["max_effective_direct_over_q"]
        ),
        "baseline_primitive_residual_over_q": float(
            baseline_audit["max_primitive_residual_over_q"]
        ),
        "corrected_primitive_residual_over_q": float(
            corrected_audit["max_primitive_residual_over_q"]
        ),
        "baseline_raw_longitudinal": float(
            baseline_audit["relative_longitudinal_gauge_residual"]
        ),
        "corrected_raw_longitudinal": float(
            corrected_audit["relative_longitudinal_gauge_residual"]
        ),
        "baseline_ward_passed": bool(baseline_ward.passed),
        "corrected_ward_passed": bool(corrected_ward.passed),
        "baseline_effective_mixed_ratio_max": _max_ward_effective_mixed_ratio(
            baseline_ward
        ),
        "corrected_effective_mixed_ratio_max": _max_ward_effective_mixed_ratio(
            corrected_ward
        ),
        "baseline_collective_condition": float(
            baseline_kernel.schur_condition_number
        ),
        "corrected_collective_condition": float(
            corrected_kernel.schur_condition_number
        ),
        "baseline_collective_inverse_method": str(
            baseline_kernel.schur_inverse_method
        ),
        "corrected_collective_inverse_method": str(
            corrected_kernel.schur_inverse_method
        ),
        "baseline_chi_bar": float(baseline_sheet.chi_bar),
        "corrected_chi_bar": float(corrected_sheet.chi_bar),
        "chi_bar_delta": float(corrected_sheet.chi_bar - baseline_sheet.chi_bar),
        "baseline_dbar_t": float(baseline_sheet.dbar_t),
        "corrected_dbar_t": float(corrected_sheet.dbar_t),
        "dbar_t_delta": float(corrected_sheet.dbar_t - baseline_sheet.dbar_t),
        "diagnostic_only": True,
        "projection_applied": False,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }

    output = args.output
    _write_csv(output, row)
    output.with_suffix(".summary.txt").write_text(_summary_text(row), encoding="utf-8")
    payload = {
        "schema": "dwave_bond_metric_full_kernel_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": vars(args),
        "row": row,
        "primitive_metadata": primitive_metadata,
        "counterterm_application": application,
        "baseline": {
            "components_metadata": baseline_components.metadata,
            "ward_audit": baseline_audit,
        },
        "corrected": {
            "components_metadata": corrected_components.metadata,
            "ward_audit": corrected_audit,
        },
        "status": {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    print(_summary_text(row), end="")
    print(f"CSV:     {output}")
    print(f"JSON:    {output.with_suffix('.json')}")
    print(f"summary: {output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
