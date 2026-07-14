"""Profile d-wave complete-orbit primitive integrands on composite-Gauss nodes.

This is a localization diagnostic for difficult diagonal q points.  It evaluates the
same batched complete-orbit primitive callback used by the formal Matsubara backend,
retains every packed primitive vector, reports cyclic adjacent-node variation, probes
all composite-panel boundaries from both sides, and reconstructs the fully integrated
physical response after the primitive sum.  No response operation is performed before
the full transverse integral.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.commensurate_orbit_gauss_aggregate import (
    _composite_gauss_rule,
    _wrap_periodic_bz,
)
from validation.lib.dwave_commensurate_orbit_gauss import (
    commensurate_orbit_basis,
    complementary_orbit_origins,
)
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.dwave_positive_orbit_adaptive import (
    _HEADER_WIDTH,
    _PER_FREQUENCY_WIDTH,
    _unpack_integrated_primitives,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_primitive_evaluator import (
    PositiveOrbitPrimitiveEvaluator,
)


DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/dwave_orbit_integrand_profile/"
    "dwave_diagonal_integrand_profile.json"
)

_HEADER_BLOCKS = (
    ("direct", 0, 9),
    ("counterterm", 9, 13),
    ("phase_direct", 13, 15),
    ("ward_rhs", 15, 18),
)
_FREQUENCY_BLOCKS = (
    ("bubble", 0, 9),
    ("collective_bubble", 9, 13),
    ("em_collective_left", 13, 19),
    ("collective_em_right", 19, 25),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=1)
    parser.add_argument("--my", type=int, default=1)
    parser.add_argument(
        "--matsubara-indices", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--gauss-order", type=int, default=384)
    parser.add_argument("--panel-count", type=int, default=16)
    parser.add_argument("--integration-start", type=float, default=-np.pi)
    parser.add_argument("--boundary-epsilon", type=float, default=1e-7)
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--transverse-workers", type=int, default=1)
    parser.add_argument("--transverse-task-size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--top-count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.nk <= 0 or args.gauss_order <= 0 or args.panel_count <= 0:
        parser.error("nk, Gauss order and panel count must be positive")
    if args.gauss_order % args.panel_count != 0:
        parser.error("Gauss order must be divisible by panel count")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index < 0 for index in args.matsubara_indices):
        parser.error("Matsubara indices must be non-negative")
    if len(set(args.matsubara_indices)) != len(args.matsubara_indices):
        parser.error("Matsubara indices must be unique")
    if not np.isfinite(args.integration_start):
        parser.error("integration start must be finite")
    if not np.isfinite(args.boundary_epsilon) or args.boundary_epsilon <= 0.0:
        parser.error("boundary epsilon must be finite and positive")
    if args.transverse_workers <= 0 or args.transverse_task_size <= 0:
        parser.error("workers and task size must be positive")
    if args.top_count <= 0:
        parser.error("top count must be positive")
    return args


def _paths(output: Path) -> dict[str, Path]:
    base = output.with_suffix("")
    return {
        "json": output,
        "nodes": base.with_name(base.name + ".nodes.csv"),
        "boundaries": base.with_name(base.name + ".boundaries.csv"),
        "panels": base.with_name(base.name + ".panels.csv"),
        "npz": base.with_suffix(".npz"),
        "summary": base.with_name(base.name + ".summary.txt"),
    }


def _block_slices(indices: tuple[int, ...]) -> dict[str, slice]:
    slices = {name: slice(start, stop) for name, start, stop in _HEADER_BLOCKS}
    offset = _HEADER_WIDTH
    for index in indices:
        for name, start, stop in _FREQUENCY_BLOCKS:
            slices[f"n{index}_{name}"] = slice(offset + start, offset + stop)
        offset += _PER_FREQUENCY_WIDTH
    return slices


def _relative(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    delta = float(np.linalg.norm(b - a))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    relative = delta / max(scale, np.finfo(float).tiny)
    return delta, relative


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def _task_chunks(values: Sequence[float], size: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in values[start : start + size])
        for start in range(0, len(values), size)
    )


def _summary_text(payload: dict[str, Any]) -> str:
    config = payload["config"]
    endpoint = payload["endpoint_periodicity"]
    adjacency = payload["adjacent_node_diagnostics"]
    boundaries = payload["boundary_diagnostics"]
    physical = payload["full_integral_physical_rows"]
    lines = [
        "d-wave complete-orbit transverse integrand profile",
        "=" * 96,
        (
            f"nk={config['nk']}; m=({config['mx']},{config['my']}); "
            f"order={config['gauss_order']}; panels={config['panel_count']}; "
            f"cut={config['integration_start']:.12f}"
        ),
        f"indices={tuple(config['matsubara_indices'])}; workers/task={config['transverse_workers']}/{config['transverse_task_size']}",
        "",
        f"endpoint packed relative = {endpoint['packed_relative']:.6e}",
        (
            "maximum adjacent packed relative = "
            f"{adjacency['maximum_packed_relative']:.6e} at node "
            f"{adjacency['maximum_packed_relative_node']}"
        ),
        (
            "maximum boundary packed relative = "
            f"{boundaries['maximum_packed_relative']:.6e} at boundary "
            f"{boundaries['maximum_packed_relative_boundary']}"
        ),
        "",
        "full-integral physical rows:",
    ]
    for row in physical:
        lines.append(
            f"  n={row['matsubara_index']:2d} sector={row['response_sector']:<8s} "
            f"physical={row['physical_passed']} Ward={row['ward_passed']} "
            f"strict0={row['strict_static_ward_passed']} "
            f"chi={row['chi_bar']:.12e} D_T={row['dbar_t']:.12e} "
            f"logdet={row['logdet']:.12e}"
        )
    lines.extend(
        [
            "",
            f"material workspace = {payload['evaluator_profile']['material_workspace_implementation']}",
            f"q workspace = {payload['evaluator_profile']['q_workspace_implementation']}",
            f"callbacks = {payload['evaluator_profile']['callbacks']}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    indices = tuple(sorted(int(value) for value in args.matsubara_indices))
    xi_values = np.asarray(
        [
            0.0 if index == 0 else matsubara_energy_eV(index, args.temperature_K)
            for index in indices
        ],
        dtype=float,
    )
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    physics_config = OrbitAcceptancePhysicsConfig(
        degeneracy=args.degeneracy,
        separation_nm=args.separation_nm,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
    )

    primitive, transverse, orbit_shift_steps = commensurate_orbit_basis(
        args.mx, args.my
    )
    origins = complementary_orbit_origins(
        orbit_shift_steps, args.shift_s, args.subgrid_average
    )
    step = 2.0 * np.pi / float(args.nk)
    q_model = step * np.asarray([args.mx, args.my], dtype=float)
    orbit_base = np.concatenate(
        [
            (
                -np.pi + (np.arange(args.nk, dtype=float) + float(origin)) * step
            )[:, None]
            * primitive[None, :]
            for origin in origins
        ],
        axis=0,
    )
    orbit_weights = np.full(
        orbit_base.shape[0], 1.0 / float(orbit_base.shape[0]), dtype=float
    )
    nodes, weights, panel_order = _composite_gauss_rule(
        total_order=args.gauss_order,
        panel_count=args.panel_count,
        integration_start=args.integration_start,
    )
    panel_width = 2.0 * np.pi / float(args.panel_count)
    panel_indices = np.minimum(
        ((nodes - args.integration_start) / panel_width).astype(int),
        args.panel_count - 1,
    )
    boundaries = args.integration_start + panel_width * np.arange(
        args.panel_count, dtype=float
    )
    probe_values = np.ravel(
        np.column_stack(
            (boundaries - args.boundary_epsilon, boundaries + args.boundary_epsilon)
        )
    )
    endpoint_values = np.asarray(
        [args.integration_start, args.integration_start + 2.0 * np.pi], dtype=float
    )

    def geometry(t_value: float) -> tuple[np.ndarray, dict[str, Any]]:
        raw = orbit_base + float(t_value) * transverse[None, :]
        wrapped = _wrap_periodic_bz(raw)
        wrap_x = int(np.count_nonzero((raw[:, 0] < -np.pi) | (raw[:, 0] >= np.pi)))
        wrap_y = int(np.count_nonzero((raw[:, 1] < -np.pi) | (raw[:, 1] >= np.pi)))
        boundary_distance = float(np.min(np.pi - np.abs(wrapped)))
        return wrapped, {
            "wrap_count_x": wrap_x,
            "wrap_count_y": wrap_y,
            "minimum_bz_boundary_distance": boundary_distance,
        }

    with PositiveOrbitPrimitiveEvaluator(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_values,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        process_workers=args.transverse_workers,
    ) as evaluator:
        def evaluate_one(t_value: float) -> tuple[np.ndarray, dict[str, Any]]:
            points, diagnostics = geometry(t_value)
            packed = np.asarray(evaluator(points, orbit_weights), dtype=complex).reshape(-1)
            return packed, diagnostics

        def evaluate_many(values: Sequence[float]) -> list[tuple[np.ndarray, dict[str, Any]]]:
            chunks = _task_chunks(values, args.transverse_task_size)

            def evaluate_chunk(chunk: tuple[float, ...]):
                return tuple(evaluate_one(value) for value in chunk)

            effective_workers = min(args.transverse_workers, max(len(chunks), 1))
            if effective_workers == 1:
                chunk_results = tuple(evaluate_chunk(chunk) for chunk in chunks)
            else:
                with ThreadPoolExecutor(
                    max_workers=effective_workers,
                    thread_name_prefix="integrand-profile",
                ) as executor:
                    chunk_results = tuple(executor.map(evaluate_chunk, chunks))
            return [item for chunk in chunk_results for item in chunk]

        node_results = evaluate_many(nodes)
        probe_results = evaluate_many(probe_values)
        endpoint_results = evaluate_many(endpoint_values)
        profile = evaluator.profile_snapshot()
        base_config = evaluator.base_config
        options = evaluator.options
        phase_hessian_policy = evaluator.phase_hessian_policy

    packed_nodes = np.stack([result[0] for result in node_results], axis=0)
    expected_width = _HEADER_WIDTH + _PER_FREQUENCY_WIDTH * len(indices)
    if packed_nodes.shape != (args.gauss_order, expected_width):
        raise RuntimeError(
            f"unexpected packed-node shape {packed_nodes.shape}; expected "
            f"{(args.gauss_order, expected_width)}"
        )
    block_slices = _block_slices(indices)

    normalized_weights = np.asarray(weights, dtype=float) / (2.0 * np.pi)
    weighted_nodes = normalized_weights[:, None] * packed_nodes
    full_integral = np.sum(weighted_nodes, axis=0, dtype=complex)
    panel_integrals = np.stack(
        [
            np.sum(weighted_nodes[panel_indices == panel], axis=0, dtype=complex)
            for panel in range(args.panel_count)
        ],
        axis=0,
    )

    next_nodes = np.roll(packed_nodes, -1, axis=0)
    next_t = np.roll(nodes, -1)
    next_t[-1] += 2.0 * np.pi
    node_dt = next_t - nodes
    packed_deltas = np.linalg.norm(next_nodes - packed_nodes, axis=1)
    packed_scales = np.maximum(
        np.linalg.norm(packed_nodes, axis=1), np.linalg.norm(next_nodes, axis=1)
    )
    packed_relatives = packed_deltas / np.maximum(
        packed_scales, np.finfo(float).tiny
    )
    packed_derivatives = packed_deltas / np.maximum(node_dt, np.finfo(float).tiny)

    node_rows: list[dict[str, Any]] = []
    for position, (t_value, t_weight, panel, result) in enumerate(
        zip(nodes, normalized_weights, panel_indices, node_results, strict=True)
    ):
        packed, diagnostics = result
        row: dict[str, Any] = {
            "node_index": position,
            "panel_index": int(panel),
            "local_node_index": int(position % panel_order),
            "t": float(t_value),
            "normalized_weight": float(t_weight),
            **diagnostics,
            "packed_norm": float(np.linalg.norm(packed)),
            "next_dt": float(node_dt[position]),
            "next_packed_absolute": float(packed_deltas[position]),
            "next_packed_relative": float(packed_relatives[position]),
            "next_packed_derivative_norm": float(packed_derivatives[position]),
            "wrap_count_changed_to_next": bool(
                diagnostics["wrap_count_x"]
                != node_results[(position + 1) % len(node_results)][1]["wrap_count_x"]
                or diagnostics["wrap_count_y"]
                != node_results[(position + 1) % len(node_results)][1]["wrap_count_y"]
            ),
        }
        for name, block_slice in block_slices.items():
            block = packed[block_slice]
            next_block = next_nodes[position, block_slice]
            absolute, relative = _relative(block, next_block)
            row[f"{name}_norm"] = float(np.linalg.norm(block))
            row[f"{name}_next_absolute"] = absolute
            row[f"{name}_next_relative"] = relative
        node_rows.append(row)

    probe_vectors = [result[0] for result in probe_results]
    boundary_rows: list[dict[str, Any]] = []
    for boundary_index, boundary in enumerate(boundaries):
        left_vector = probe_vectors[2 * boundary_index]
        right_vector = probe_vectors[2 * boundary_index + 1]
        left_geometry = probe_results[2 * boundary_index][1]
        right_geometry = probe_results[2 * boundary_index + 1][1]
        absolute, relative = _relative(left_vector, right_vector)
        row = {
            "boundary_index": boundary_index,
            "boundary_t": float(boundary),
            "epsilon": float(args.boundary_epsilon),
            "packed_left_right_absolute": absolute,
            "packed_left_right_relative": relative,
            "left_wrap_count_x": left_geometry["wrap_count_x"],
            "right_wrap_count_x": right_geometry["wrap_count_x"],
            "left_wrap_count_y": left_geometry["wrap_count_y"],
            "right_wrap_count_y": right_geometry["wrap_count_y"],
            "wrap_count_changed": bool(
                left_geometry["wrap_count_x"] != right_geometry["wrap_count_x"]
                or left_geometry["wrap_count_y"] != right_geometry["wrap_count_y"]
            ),
        }
        for name, block_slice in block_slices.items():
            block_absolute, block_relative = _relative(
                left_vector[block_slice], right_vector[block_slice]
            )
            row[f"{name}_absolute"] = block_absolute
            row[f"{name}_relative"] = block_relative
        boundary_rows.append(row)

    full_norm = float(np.linalg.norm(full_integral))
    panel_rows: list[dict[str, Any]] = []
    for panel, vector in enumerate(panel_integrals):
        row = {
            "panel_index": panel,
            "left_t": float(args.integration_start + panel * panel_width),
            "right_t": float(args.integration_start + (panel + 1) * panel_width),
            "packed_integral_norm": float(np.linalg.norm(vector)),
            "packed_integral_norm_over_full": float(
                np.linalg.norm(vector) / max(full_norm, np.finfo(float).tiny)
            ),
        }
        for name, block_slice in block_slices.items():
            block = vector[block_slice]
            full_block = full_integral[block_slice]
            row[f"{name}_integral_norm"] = float(np.linalg.norm(block))
            row[f"{name}_integral_norm_over_full"] = float(
                np.linalg.norm(block)
                / max(float(np.linalg.norm(full_block)), np.finfo(float).tiny)
            )
        panel_rows.append(row)

    quadrature_view = SimpleNamespace(
        nk=args.nk,
        primitive_direction=primitive,
        transverse_direction=transverse,
        orbit_shift_steps=orbit_shift_steps,
        orbit_origins=origins,
        pilot_order=panel_order,
        epsabs=0.0,
        epsrel=0.0,
        limit=args.gauss_order,
        quadrature="composite_fixed_gauss_legendre_integrand_profile",
        norm="none",
        scaled_error_estimate=0.0,
        success=True,
        status=0,
        message="integrand profile primitive sum completed",
        transverse_evaluations=args.gauss_order,
        point_evaluations=args.gauss_order * orbit_base.shape[0],
    )
    components, rhs_values = _unpack_integrated_primitives(
        full_integral,
        xi_values=xi_values,
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        q_model=q_model,
        options=options,
        quadrature=quadrature_view,
        phase_hessian_policy=phase_hessian_policy,
    )
    physical_rows = []
    for index, xi, components_value, rhs in zip(
        indices, xi_values, components, rhs_values, strict=True
    ):
        physical = evaluate_matsubara_pipeline(
            components=components_value,
            rhs=rhs,
            q_model=q_model,
            xi_eV=float(xi),
            config=physics_config,
        )
        physical_rows.append(
            {
                "matsubara_index": index,
                "xi_eV": float(xi),
                "response_sector": str(physical["response_sector"]),
                "physical_passed": bool(physical["physical_passed"]),
                "ward_passed": bool(physical["ward_passed"]),
                "strict_static_ward_passed": bool(
                    physical["strict_static_ward_passed"]
                ),
                "sheet_validation_passed": bool(
                    physical["sheet_validation_passed"]
                ),
                "chi_bar": float(physical["chi_bar"]),
                "dbar_t": float(physical["dbar_t"]),
                "logdet": float(physical["logdet"]),
            }
        )

    endpoint_absolute, endpoint_relative = _relative(
        endpoint_results[0][0], endpoint_results[1][0]
    )
    max_adjacent_index = int(np.argmax(packed_relatives))
    max_derivative_index = int(np.argmax(packed_derivatives))
    boundary_relatives = np.asarray(
        [row["packed_left_right_relative"] for row in boundary_rows], dtype=float
    )
    max_boundary_index = int(np.argmax(boundary_relatives))
    top_count = min(args.top_count, args.gauss_order)
    top_adjacent = np.argsort(packed_relatives)[-top_count:][::-1]
    top_derivative = np.argsort(packed_derivatives)[-top_count:][::-1]

    paths = _paths(args.output)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["nodes"], node_rows)
    _write_csv(paths["boundaries"], boundary_rows)
    _write_csv(paths["panels"], panel_rows)
    np.savez_compressed(
        paths["npz"],
        t=np.asarray(nodes, dtype=float),
        normalized_weights=normalized_weights,
        panel_indices=np.asarray(panel_indices, dtype=int),
        packed_real=packed_nodes.real,
        packed_imag=packed_nodes.imag,
        weighted_packed_real=weighted_nodes.real,
        weighted_packed_imag=weighted_nodes.imag,
        panel_integrals_real=panel_integrals.real,
        panel_integrals_imag=panel_integrals.imag,
        full_integral_real=full_integral.real,
        full_integral_imag=full_integral.imag,
        primitive_direction=np.asarray(primitive, dtype=int),
        transverse_direction=np.asarray(transverse, dtype=int),
        orbit_origins=np.asarray(origins, dtype=float),
        matsubara_indices=np.asarray(indices, dtype=int),
        xi_eV_values=xi_values,
    )

    payload = {
        "schema": "dwave_complete_orbit_transverse_integrand_profile_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "nk": args.nk,
            "mx": args.mx,
            "my": args.my,
            "matsubara_indices": indices,
            "gauss_order": args.gauss_order,
            "panel_count": args.panel_count,
            "panel_order": panel_order,
            "integration_start": args.integration_start,
            "boundary_epsilon": args.boundary_epsilon,
            "shift_s": args.shift_s,
            "subgrid_average": args.subgrid_average,
            "transverse_workers": args.transverse_workers,
            "transverse_task_size": args.transverse_task_size,
            "primitive_direction": primitive,
            "transverse_direction": transverse,
            "orbit_shift_steps": orbit_shift_steps,
            "orbit_origins": origins,
            "q_model": q_model,
        },
        "endpoint_periodicity": {
            "packed_absolute": endpoint_absolute,
            "packed_relative": endpoint_relative,
        },
        "adjacent_node_diagnostics": {
            "maximum_packed_relative": float(packed_relatives[max_adjacent_index]),
            "maximum_packed_relative_node": max_adjacent_index,
            "maximum_packed_relative_t": float(nodes[max_adjacent_index]),
            "maximum_packed_relative_panel": int(panel_indices[max_adjacent_index]),
            "maximum_packed_relative_wrap_count_changed": bool(
                node_rows[max_adjacent_index]["wrap_count_changed_to_next"]
            ),
            "maximum_derivative_norm": float(
                packed_derivatives[max_derivative_index]
            ),
            "maximum_derivative_node": max_derivative_index,
            "maximum_derivative_t": float(nodes[max_derivative_index]),
            "top_relative_nodes": [int(value) for value in top_adjacent],
            "top_derivative_nodes": [int(value) for value in top_derivative],
        },
        "boundary_diagnostics": {
            "maximum_packed_relative": float(boundary_relatives[max_boundary_index]),
            "maximum_packed_relative_boundary": max_boundary_index,
            "maximum_packed_relative_t": float(boundaries[max_boundary_index]),
            "maximum_wrap_count_changed": bool(
                boundary_rows[max_boundary_index]["wrap_count_changed"]
            ),
        },
        "full_integral_physical_rows": physical_rows,
        "evaluator_profile": profile.as_dict(),
        "outputs": {name: str(path) for name, path in paths.items()},
        "status": {
            "same_batched_complete_orbit_evaluator": True,
            "primitive_integrated_before_postprocessing": True,
            "full_transverse_period_integrated": True,
            "symmetry_reduction_applied": False,
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    paths["json"].write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = _summary_text(payload)
    paths["summary"].write_text(summary, encoding="utf-8")
    print(summary)
    print(f"JSON:       {paths['json']}")
    print(f"nodes CSV:  {paths['nodes']}")
    print(f"bounds CSV: {paths['boundaries']}")
    print(f"panels CSV: {paths['panels']}")
    print(f"NPZ:        {paths['npz']}")


if __name__ == "__main__":
    main()
