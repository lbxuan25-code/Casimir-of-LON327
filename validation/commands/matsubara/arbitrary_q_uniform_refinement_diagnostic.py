"""Diagnose arbitrary-q convergence on retained full periodic BZ grids only.

This command evaluates a strictly global sequence of fixed even-N periodic grids.
It reports block-resolved primitive changes and per-frequency physical observables.
It contains no local cell ranking, no partial refinement, and no adaptive tree.
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
from lno327.workflows.arbitrary_q_matsubara import integrate_arbitrary_q_periodic_bz
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_uniform_refinement/diagnostic.json"
)
_HEADER_GROUPS = (
    ("direct_contact", 9),
    ("collective_counterterm", 4),
    ("phase_direct", 2),
    ("ward_rhs", 3),
)
_FREQUENCY_GROUPS = (
    ("em_bubble", 9),
    ("collective_bubble", 4),
    ("em_collective_left", 6),
    ("collective_em_right", 6),
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="dwave")
    parser.add_argument("--q-model", nargs=2, type=float, default=(0.0300152, 0.0200101))
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--N-values", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--shift", nargs=2, type=float, default=(0.5, 0.5))
    parser.add_argument("--canonical-block", type=int, default=4096)
    parser.add_argument("--runtime-chunk", type=int, default=16384)
    parser.add_argument("--primitive-rtol", type=float, default=1e-3)
    parser.add_argument("--primitive-atol", type=float, default=1e-12)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.matsubara_indices = tuple(sorted(set(int(v) for v in args.matsubara_indices)))
    args.N_values = tuple(int(v) for v in args.N_values)
    if not args.matsubara_indices or any(v < 0 for v in args.matsubara_indices):
        parser.error("--matsubara-indices must be nonempty and non-negative")
    if len(args.N_values) < 2 or any(v <= 0 or v % 2 for v in args.N_values):
        parser.error("--N-values must contain at least two positive even integers")
    if tuple(sorted(set(args.N_values))) != args.N_values:
        parser.error("--N-values must be strictly increasing and unique")
    for name in (
        "primitive_rtol",
        "primitive_atol",
        "ward_tolerance",
        "ward_absolute_tolerance",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def primitive_block_slices(frequency_count: int) -> tuple[tuple[str, slice], ...]:
    count = int(frequency_count)
    if count <= 0:
        raise ValueError("frequency_count must be positive")
    groups: list[tuple[str, slice]] = []
    cursor = 0
    for name, width in _HEADER_GROUPS:
        groups.append((name, slice(cursor, cursor + width)))
        cursor += width
    for index in range(count):
        for name, width in _FREQUENCY_GROUPS:
            groups.append((f"n_index_{index}:{name}", slice(cursor, cursor + width)))
            cursor += width
    return tuple(groups)


def _max_abs(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=complex)
    return float(np.max(np.abs(array))) if array.size else 0.0


def block_resolved_primitive_change(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    frequency_count: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = np.asarray(previous, dtype=complex).reshape(-1)
    right = np.asarray(current, dtype=complex).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("successive primitive vectors must have equal shape")
    rows: list[dict[str, Any]] = []
    cursor_end = 0
    tiny = np.finfo(float).tiny
    for name, block in primitive_block_slices(frequency_count):
        cursor_end = block.stop
        a = left[block]
        b = right[block]
        absolute = _max_abs(b - a)
        previous_norm = _max_abs(a)
        current_norm = _max_abs(b)
        scale = max(previous_norm, current_norm)
        threshold = float(atol) + float(rtol) * scale
        ratio = absolute / max(threshold, tiny)
        rows.append(
            {
                "name": name,
                "previous_max_abs": previous_norm,
                "current_max_abs": current_norm,
                "absolute_max_change": absolute,
                "relative_max_change": absolute / max(scale, tiny),
                "mixed_threshold": threshold,
                "mixed_ratio": ratio,
                "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
            }
        )
    if cursor_end != left.size:
        raise ValueError(
            f"packed primitive width mismatch: covered={cursor_end}, actual={left.size}"
        )
    worst = max(rows, key=lambda row: float(row["mixed_ratio"]))
    return {
        "passed": all(bool(row["passed"]) for row in rows),
        "max_mixed_ratio": float(worst["mixed_ratio"]),
        "worst_block": str(worst["name"]),
        "blocks": rows,
    }


def _mixed_array(left: Any, right: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {
            "finite": False,
            "absolute": float("nan"),
            "relative": float("nan"),
            "mixed_ratio": float("nan"),
            "passed": False,
        }
    absolute = float(np.linalg.norm(b - a))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "finite": True,
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
        "mixed_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
    }


def _physical_states(result: object, q: np.ndarray, args: argparse.Namespace):
    config = OrbitAcceptancePhysicsConfig(
        separation_nm=float(args.separation_nm),
        ward_tolerance=float(args.ward_tolerance),
        ward_absolute_tolerance=float(args.ward_absolute_tolerance),
    )
    rows = []
    for n, frequency, component, rhs in zip(
        args.matsubara_indices,
        result.xi_eV_values,
        result.components,
        result.rhs,
        strict=True,
    ):
        state = evaluate_matsubara_pipeline(
            components=component,
            rhs=rhs,
            q_model=q,
            xi_eV=float(frequency),
            config=config,
        )
        rows.append((int(n), state))
    return rows


def _physical_summary(n: int, state: dict[str, Any]) -> dict[str, Any]:
    primary = np.asarray(state["primary_response"], dtype=complex)
    reflection = np.asarray(state["reflection"], dtype=complex)
    return {
        "n": int(n),
        "physical_passed": bool(state["physical_passed"]),
        "ward_passed": bool(state["ward_passed"]),
        "strict_static_ward_passed": bool(state["strict_static_ward_passed"]),
        "sheet_validation_passed": bool(state["sheet_validation_passed"]),
        "reflection_constructed": bool(state["reflection_constructed"]),
        "logdet_passed": bool(state["logdet_passed"]),
        "primary_norm": float(np.linalg.norm(primary)),
        "reflection_norm": float(np.linalg.norm(reflection)),
        "logdet": float(state["logdet"]),
        "chi_bar": float(state["chi_bar"]),
        "dbar_t": float(state["dbar_t"]),
        "ward_effective_mixed_ratio_max": float(
            state["ward_effective_mixed_ratio_max"]
        ),
        "schur_condition_number": float(state["schur_condition_number"]),
        "error": str(state["error"]),
    }


def _observable_change(previous, current, *, rtol: float, atol: float):
    rows = []
    for (previous_n, left), (current_n, right) in zip(previous, current, strict=True):
        if previous_n != current_n:
            raise ValueError("physical rows are not frequency aligned")
        rows.append(
            {
                "n": int(current_n),
                "primary": _mixed_array(
                    left["primary_response"], right["primary_response"], rtol=rtol, atol=atol
                ),
                "reflection": _mixed_array(
                    left["reflection"], right["reflection"], rtol=rtol, atol=atol
                ),
                "logdet": _mixed_array(
                    np.asarray([left["logdet"]]),
                    np.asarray([right["logdet"]]),
                    rtol=rtol,
                    atol=atol,
                ),
                "ward_ratio_previous": float(left["ward_effective_mixed_ratio_max"]),
                "ward_ratio_current": float(right["ward_effective_mixed_ratio_max"]),
                "ward_passed_current": bool(right["ward_passed"]),
            }
        )
    return rows


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(args.pairing, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    q = np.asarray(args.q_model, dtype=float)
    xi = np.asarray(
        [
            0.0 if n == 0 else matsubara_energy_eV(n, args.temperature_K)
            for n in args.matsubara_indices
        ],
        dtype=float,
    )
    levels = []
    previous_result = None
    previous_physical = None
    for n_grid in args.N_values:
        grid_started = perf_counter()
        grid = build_periodic_bz_grid(int(n_grid), tuple(float(v) for v in args.shift))
        grid_seconds = perf_counter() - grid_started
        material_started = perf_counter()
        cache = build_material_grid_cache(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            config=KuboConfig.from_kelvin(
                omega_eV=0.0,
                temperature_K=float(args.temperature_K),
                eta_eV=float(args.eta_eV),
                output_si=False,
            ),
            options=FiniteQEngineOptions(phase_hessian_policy="q_independent"),
            grid=grid,
        )
        material_seconds = perf_counter() - material_started
        response_started = perf_counter()
        result = integrate_arbitrary_q_periodic_bz(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=xi,
            temperature_K=float(args.temperature_K),
            eta_eV=float(args.eta_eV),
            q_model=q,
            n=grid.n,
            shift=grid.shift,
            canonical_reduction_block_size=int(args.canonical_block),
            runtime_chunk_size=int(args.runtime_chunk),
            material_cache=cache,
        )
        response_seconds = perf_counter() - response_started
        physical = _physical_states(result, q, args)
        level: dict[str, Any] = {
            "N": int(n_grid),
            "point_count": int(result.profile.k_point_count),
            "grid_build_seconds": float(grid_seconds),
            "material_build_seconds": float(material_seconds),
            "response_seconds": float(response_seconds),
            "profile": result.profile.as_dict(),
            "operator_ward_passed": bool(result.operator_ward.passed),
            "packed_primitive_norm": float(np.linalg.norm(result.packed_primitives)),
            "physical_by_frequency": [
                _physical_summary(n, state) for n, state in physical
            ],
            "successive_primitive": None,
            "successive_observables": None,
        }
        if previous_result is not None and previous_physical is not None:
            level["successive_primitive"] = block_resolved_primitive_change(
                previous_result.packed_primitives,
                result.packed_primitives,
                frequency_count=len(args.matsubara_indices),
                rtol=float(args.primitive_rtol),
                atol=float(args.primitive_atol),
            )
            level["successive_observables"] = _observable_change(
                previous_physical,
                physical,
                rtol=float(args.primitive_rtol),
                atol=float(args.primitive_atol),
            )
        levels.append(level)
        previous_result = result
        previous_physical = physical

    payload = {
        "schema": "arbitrary-q-uniform-refinement-diagnostic-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integration_family": "fixed_even_N_full_periodic_BZ_only",
        "local_refinement_present": False,
        "pairing": args.pairing,
        "q_model": q.tolist(),
        "matsubara_indices": list(args.matsubara_indices),
        "xi_eV_values": xi.tolist(),
        "N_values": list(args.N_values),
        "shift": list(args.shift),
        "primitive_rtol": float(args.primitive_rtol),
        "primitive_atol": float(args.primitive_atol),
        "levels": levels,
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    _atomic_write(args.output, payload)
    final = levels[-1]
    previous = final["successive_primitive"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "final_N": int(final["N"]),
                "final_point_count": int(final["point_count"]),
                "successive_primitive_passed": None if previous is None else previous["passed"],
                "successive_primitive_max_ratio": None if previous is None else previous["max_mixed_ratio"],
                "successive_primitive_worst_block": None if previous is None else previous["worst_block"],
                "all_final_physical_passed": all(
                    bool(row["physical_passed"])
                    for row in final["physical_by_frequency"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
