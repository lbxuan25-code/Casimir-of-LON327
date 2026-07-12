"""Optimized d-wave full-kernel bond-metric Ward audit.

The command preserves the established output schema and complete 48-component
primitive contract.  For commensurate q, every required tensor subgrid is
pre-diagonalized once; midpoint and endpoint bands are then selected by exact
integer index maps during streamed integration.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.ward import bond_metric_full_kernel as _reference
from validation.commands.ward.commensurate import _jsonable
from validation.lib.commensurate_dwave_bdg_cache import (
    CachedCommensurateDWaveContext,
    build_commensurate_dwave_bdg_cache,
)
from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)
from validation.lib.dwave_bond_phase_counterterm import (
    apply_nearest_neighbor_dwave_phase_counterterm,
)
from validation.lib.dwave_iterated_adaptive import assemble_dwave_static_primitives
from validation.lib.dwave_iterated_adaptive_fast import (
    build_dwave_static_integrand_context,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _summary_text(row: dict[str, Any]) -> str:
    base = _reference._summary_text(row).rstrip()
    lines = [
        base,
        "",
        "Optimization",
        "------------",
        f"commensurate eigensystem cache = {row['eigensystem_cache_enabled']}",
        f"cached subgrids = {row['cached_subgrid_count']}",
        f"cached eigensystems = {row['cached_eigensystem_count']}",
        f"cache build wall time = {row['cache_build_wall_seconds']:.3f} s",
        f"integration-only wall time = {row['integration_wall_seconds']:.3f} s",
        f"total cached compute wall time = {row['total_compute_wall_seconds']:.3f} s",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _reference._parse_args()
    base_grid = CommensuratePeriodicGrid(
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        max_points=args.max_points,
    )
    q = base_grid.q_model
    origins = (
        _reference.complementary_subgrid_origins(
            args.mx, args.my, args.shift_x, args.shift_y
        )
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
        FiniteQEngineOptions(),
    )
    cache = build_commensurate_dwave_bdg_cache(
        context,
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        origins=origins,
        chunk_size=args.chunk_size,
        max_points=args.max_points,
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
        evaluator = CachedCommensurateDWaveContext(cache, (shift_x, shift_y))
        print(
            f"integrating cached subgrid {index}/{len(origins)}: "
            f"shift=({shift_x:.6g},{shift_y:.6g}), points={grid.num_points}",
            flush=True,
        )
        integral = integrate_commensurate_periodic_vector(
            grid,
            evaluator.evaluate_complex,
            chunk_size=args.chunk_size,
        )
        values.append(np.asarray(integral.value, dtype=complex))
        integrals.append(integral)

    primitive = np.mean(np.stack(values, axis=0), axis=0)
    integration = _reference.SubgridAverageSummary(
        point_evaluations=sum(int(item.point_evaluations) for item in integrals),
        chunks=sum(int(item.chunks) for item in integrals),
        chunk_size=args.chunk_size,
        wall_seconds=sum(float(item.wall_seconds) for item in integrals),
        summation_method=(
            "componentwise_equal_average_of_complementary_subgrids_after_"
            + str(integrals[0].summation_method)
        ),
    )
    common_metadata = {
        "integration_strategy": "commensurate_cached_complete_periodic_tensor_subgrid_average",
        "nk": int(args.nk),
        "integer_q_shift": (int(args.mx), int(args.my)),
        "q_constructed_from_integer_grid_shift": True,
        "translation_by_q_is_exact_index_permutation": True,
        "half_q_single_subgrid_compatible": bool(
            base_grid.half_translation_permutation_exact
        ),
        "complementary_subgrid_origins": origins,
        "componentwise_subgrid_average_before_schur": bool(len(origins) > 1),
        "subgrid_count": len(origins),
        "all_primitive_channels_share_each_periodic_grid": True,
        "primitive_vector_averaged_before_schur": True,
        "summation_method": integration.summation_method,
        "commensurate_eigensystem_cache_enabled": True,
        "cached_subgrid_count": len(cache.subgrids),
        "cached_eigensystem_count": cache.eigensystem_count,
        "cache_build_wall_seconds": cache.build_wall_seconds,
        "endpoint_bands_selected_by_exact_integer_index_maps": True,
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

    baseline_kernel, baseline_ward, baseline_sheet, baseline_audit = (
        _reference._evaluate_components(baseline_components, rhs, q, args)
    )
    corrected_kernel, corrected_ward, corrected_sheet, corrected_audit = (
        _reference._evaluate_components(corrected_components, rhs, q, args)
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
        "eigensystem_cache_enabled": True,
        "cached_subgrid_count": len(cache.subgrids),
        "cached_eigensystem_count": cache.eigensystem_count,
        "cache_build_wall_seconds": float(cache.build_wall_seconds),
        "total_compute_wall_seconds": float(
            cache.build_wall_seconds + integration.wall_seconds
        ),
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
        "baseline_phase_defect_over_q": (
            _reference._max_collective_channel_defect_over_q(baseline_audit, 1)
        ),
        "corrected_phase_defect_over_q": (
            _reference._max_collective_channel_defect_over_q(corrected_audit, 1)
        ),
        "baseline_amplitude_defect_over_q": (
            _reference._max_collective_channel_defect_over_q(baseline_audit, 0)
        ),
        "corrected_amplitude_defect_over_q": (
            _reference._max_collective_channel_defect_over_q(corrected_audit, 0)
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
        "baseline_effective_mixed_ratio_max": (
            _reference._max_ward_effective_mixed_ratio(baseline_ward)
        ),
        "corrected_effective_mixed_ratio_max": (
            _reference._max_ward_effective_mixed_ratio(corrected_ward)
        ),
        "baseline_collective_condition": float(
            baseline_kernel.schur_condition_number
        ),
        "corrected_collective_condition": float(
            corrected_kernel.schur_condition_number
        ),
        "baseline_collective_inverse_method": baseline_kernel.schur_inverse_method,
        "corrected_collective_inverse_method": corrected_kernel.schur_inverse_method,
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
    _reference._write_csv(output, row)
    summary_path = output.with_suffix(".summary.txt")
    json_path = output.with_suffix(".json")
    summary_path.write_text(_summary_text(row), encoding="utf-8")
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
        "optimization": {
            "commensurate_eigensystem_cache_enabled": True,
            "cached_subgrid_count": len(cache.subgrids),
            "cached_eigensystem_count": cache.eigensystem_count,
            "cache_build_wall_seconds": cache.build_wall_seconds,
            "integration_wall_seconds": integration.wall_seconds,
            "endpoint_index_mapping_exact": True,
        },
        "status": {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    json_path.write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )

    print(_summary_text(row), end="")
    print(f"CSV:     {output}")
    print(f"JSON:    {json_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
