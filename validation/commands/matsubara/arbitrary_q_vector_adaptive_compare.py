"""Compare cached vector-adaptive arbitrary-q cubature with the fixed-grid reference.

This command is diagnostic only. It reports cold material construction, warm
new-q response, exact-response-cache hit, primitive/post-processing time, physical
validation time, and numerical agreement. It never authorizes Casimir input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.workflows.arbitrary_q_matsubara import (
    integrate_arbitrary_q_periodic_bz,
    rotate_lab_q_to_crystal,
)
from lno327.workflows.arbitrary_q_vector_adaptive import (
    ArbitraryQVectorAdaptiveOptions,
    ArbitraryQVectorAdaptiveResponseCache,
)
from lno327.workflows.arbitrary_q_vector_adaptive_cached import (
    build_reusable_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive_cached,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.arbitrary_q_periodic_bz_qualification import _physical
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_vector_adaptive_compare/compare.json"
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="dwave")
    parser.add_argument(
        "--q-model", nargs=2, type=float,
        default=(2.0 * np.pi * 6.0 / 1256.0, 2.0 * np.pi * 4.0 / 1256.0),
        metavar=("QX", "QY"),
    )
    parser.add_argument("--warm-angle-deg", type=float, default=17.0)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--fixed-N", type=int, default=128)
    parser.add_argument("--fixed-canonical-block", type=int, default=4096)
    parser.add_argument("--fixed-runtime-chunk", type=int, default=16384)
    parser.add_argument("--coarse-grid", type=int, default=6)
    parser.add_argument("--low-order", type=int, default=2)
    parser.add_argument("--high-order", type=int, default=3)
    parser.add_argument("--adaptive-rtol", type=float, default=1e-3)
    parser.add_argument("--adaptive-atol", type=float, default=1e-9)
    parser.add_argument("--adaptive-ward-atol", type=float, default=1e-9)
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--refine-fraction", type=float, default=0.15)
    parser.add_argument("--min-refine-cells", type=int, default=4)
    parser.add_argument("--max-cells", type=int, default=4000)
    parser.add_argument("--max-evaluation-points", type=int, default=60000)
    parser.add_argument("--cell-batch-size", type=int, default=64)
    parser.add_argument("--max-cache-nodes", type=int, default=None)
    parser.add_argument("--max-cache-bytes", type=int, default=None)
    parser.add_argument("--comparison-rtol", type=float, default=1e-3)
    parser.add_argument("--comparison-atol", type=float, default=1e-12)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--allow-nonconverged", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.matsubara_indices = tuple(sorted(set(args.matsubara_indices)))
    if 0 not in args.matsubara_indices or not any(v > 0 for v in args.matsubara_indices):
        parser.error("exact zero and at least one positive Matsubara index are required")
    if args.fixed_N <= 0 or args.fixed_N % 2:
        parser.error("--fixed-N must be positive and even")
    if args.fixed_canonical_block <= 0 or args.fixed_runtime_chunk <= 0:
        parser.error("fixed block sizes must be positive")
    return args


def _mixed(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    first = np.asarray(left, dtype=complex)
    second = np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(first - second))
    scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
        "scale": scale,
        "threshold": threshold,
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _physical_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": int(row["n"]),
        "passed": bool(row["passed"]),
        "ward": bool(row["ward"]),
        "strict_static": bool(row["strict_static"]),
        "sheet": bool(row["sheet"]),
        "reflection_constructed": bool(row["reflection_constructed"]),
        "logdet_passed": bool(row["logdet_passed"]),
        "logdet": float(row["logdet"]),
        "primary_norm": float(np.linalg.norm(row["primary"])),
        "reflection_norm": float(np.linalg.norm(row["reflection"])),
        "error": str(row["error"]),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _time_call(function, /, *args, **kwargs):
    started = perf_counter()
    value = function(*args, **kwargs)
    return value, float(perf_counter() - started)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(args.pairing, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    q = np.asarray(args.q_model, dtype=float)
    warm_q = rotate_lab_q_to_crystal(q, np.deg2rad(float(args.warm_angle_deg)))
    xi = np.asarray(
        [0.0 if index == 0 else matsubara_energy_eV(index, args.temperature_K)
         for index in args.matsubara_indices],
        dtype=float,
    )

    fixed_grid, fixed_grid_build_seconds = _time_call(
        build_periodic_bz_grid, args.fixed_N, (0.5, 0.5)
    )
    fixed_cache, fixed_material_build_seconds = _time_call(
        build_material_grid_cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=KuboConfig.from_kelvin(
            omega_eV=0.0, temperature_K=args.temperature_K,
            eta_eV=args.eta_eV, output_si=False,
        ),
        options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
        grid=fixed_grid,
    )
    fixed, fixed_response_wall_seconds = _time_call(
        integrate_arbitrary_q_periodic_bz,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        q_model=q,
        n=fixed_grid.n,
        shift=fixed_grid.shift,
        canonical_reduction_block_size=args.fixed_canonical_block,
        runtime_chunk_size=args.fixed_runtime_chunk,
        material_cache=fixed_cache,
    )

    adaptive_options = ArbitraryQVectorAdaptiveOptions(
        coarse_grid=args.coarse_grid,
        low_order=args.low_order,
        high_order=args.high_order,
        relative_tolerance=args.adaptive_rtol,
        absolute_tolerance=args.adaptive_atol,
        ward_error_tolerance=args.adaptive_ward_atol,
        max_level=args.max_level,
        max_iterations=args.max_iterations,
        refine_fraction=args.refine_fraction,
        min_refine_cells=args.min_refine_cells,
        max_cells=args.max_cells,
        max_evaluation_points=args.max_evaluation_points,
        cell_batch_size=args.cell_batch_size,
    )
    adaptive_cache, adaptive_cache_object_build_seconds = _time_call(
        build_reusable_hierarchical_material_node_cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        max_cache_nodes=args.max_cache_nodes,
        max_cache_bytes=args.max_cache_bytes,
    )
    response_cache = ArbitraryQVectorAdaptiveResponseCache()
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        adaptive_options=adaptive_options,
        node_cache=adaptive_cache,
        response_cache=response_cache,
        require_converged=not args.allow_nonconverged,
    )
    adaptive, adaptive_cold_total_wall_seconds = _time_call(
        integrate_arbitrary_q_vector_adaptive_cached, q_model=q, **common
    )
    adaptive_exact_hit, adaptive_exact_response_cache_hit_seconds = _time_call(
        integrate_arbitrary_q_vector_adaptive_cached, q_model=q, **common
    )
    adaptive_warm_new_q, adaptive_warm_new_q_total_wall_seconds = _time_call(
        integrate_arbitrary_q_vector_adaptive_cached, q_model=warm_q, **common
    )
    if adaptive_exact_hit is not adaptive:
        raise RuntimeError("adaptive exact-q response cache did not return stored object")

    fixed_states, fixed_physical_pipeline_seconds = _time_call(_physical, fixed, q, args)
    adaptive_states, adaptive_physical_pipeline_seconds = _time_call(
        _physical, adaptive, q, args
    )
    frequency_rows = []
    for index, n_value in enumerate(args.matsubara_indices):
        frequency_rows.append(
            {
                "n": int(n_value),
                "primary": _mixed(
                    fixed_states[index]["primary"], adaptive_states[index]["primary"],
                    atol=args.comparison_atol, rtol=args.comparison_rtol,
                ),
                "reflection": _mixed(
                    fixed_states[index]["reflection"], adaptive_states[index]["reflection"],
                    atol=args.comparison_atol, rtol=args.comparison_rtol,
                ),
                "logdet": _mixed(
                    fixed_states[index]["logdet"], adaptive_states[index]["logdet"],
                    atol=args.comparison_atol, rtol=args.comparison_rtol,
                ),
                "fixed_physical": _physical_summary(fixed_states[index]),
                "adaptive_physical": _physical_summary(adaptive_states[index]),
            }
        )
    primitive = _mixed(
        fixed.packed_primitives, adaptive.packed_primitives,
        atol=args.comparison_atol, rtol=args.comparison_rtol,
    )
    passed = bool(
        adaptive.profile.converged
        and fixed.operator_ward.passed
        and adaptive.operator_ward.passed
        and primitive["passed"]
        and all(
            row["primary"]["passed"] and row["reflection"]["passed"]
            and row["logdet"]["passed"] and row["fixed_physical"]["passed"]
            and row["adaptive_physical"]["passed"]
            for row in frequency_rows
        )
    )
    fixed_primitive_seconds = float(fixed.profile.total_seconds)
    fixed_postprocess_seconds = max(
        fixed_response_wall_seconds - fixed_primitive_seconds, 0.0
    )
    payload = {
        "schema": "arbitrary-q-vector-adaptive-compare-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairing": args.pairing,
        "q_model": q.tolist(),
        "warm_new_q_model": warm_q.tolist(),
        "matsubara_indices": list(args.matsubara_indices),
        "xi_eV_values": xi.tolist(),
        "timing_contract": {
            "fixed_is_split_into_grid_material_response": True,
            "adaptive_cold_includes_lazy_material_nodes": True,
            "adaptive_warm_new_q_reuses_material_nodes": True,
            "exact_response_cache_hit_is_not_normal_new_q_cost": True,
        },
        "fixed": {
            "N": int(args.fixed_N),
            "point_count": int(fixed.profile.k_point_count),
            "grid_build_seconds": fixed_grid_build_seconds,
            "material_build_seconds": fixed_material_build_seconds,
            "response_wall_seconds": fixed_response_wall_seconds,
            "primitive_integration_seconds": fixed_primitive_seconds,
            "postprocess_seconds_estimate": fixed_postprocess_seconds,
            "physical_pipeline_seconds": fixed_physical_pipeline_seconds,
            "profile": fixed.profile.as_dict(),
            "material_cache": fixed_cache.metadata(),
        },
        "adaptive": {
            "options": adaptive_options.as_dict(),
            "converged": bool(adaptive.profile.converged),
            "stop_reason": str(adaptive.profile.stop_reason),
            "point_evaluations": int(adaptive.profile.total_point_evaluations),
            "accepted_cells": int(adaptive.profile.accepted_cell_count),
            "cache_object_build_seconds": adaptive_cache_object_build_seconds,
            "cold_total_wall_seconds": adaptive_cold_total_wall_seconds,
            "cold_primitive_integration_seconds": float(
                adaptive.profile.primitive_integration_seconds
            ),
            "cold_postprocess_seconds": float(adaptive.profile.postprocess_seconds),
            "warm_new_q_total_wall_seconds": adaptive_warm_new_q_total_wall_seconds,
            "warm_new_q_profile": adaptive_warm_new_q.profile.as_dict(),
            "exact_response_cache_hit_seconds": adaptive_exact_response_cache_hit_seconds,
            "physical_pipeline_seconds": adaptive_physical_pipeline_seconds,
            "profile": adaptive.profile.as_dict(),
            "material_cache": adaptive_cache.metadata(),
            "response_cache": response_cache.metadata(),
        },
        "primitive_comparison": primitive,
        "frequency_comparisons": frequency_rows,
        "diagnostic_comparison_passed": passed,
        "adaptive_backend_promoted": False,
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    _atomic_write(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "diagnostic_comparison_passed": passed,
        "adaptive_converged": bool(adaptive.profile.converged),
        "fixed_points": int(fixed.profile.k_point_count),
        "adaptive_points": int(adaptive.profile.total_point_evaluations),
        "fixed_response_wall_seconds": fixed_response_wall_seconds,
        "adaptive_cold_total_wall_seconds": adaptive_cold_total_wall_seconds,
        "adaptive_warm_new_q_total_wall_seconds": adaptive_warm_new_q_total_wall_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
