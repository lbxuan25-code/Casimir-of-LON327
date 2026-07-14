"""Screen the angular width of d-wave diagonal Matsubara quadrature sensitivity.

The command evaluates exact-zero and low positive Matsubara responses for a user-
selected set of integer q directions. Each direction is integrated with the same
full-period composite Gauss rule at two periodic cuts. The comparison is performed
only after the complete primitive integral and reports response, reflection, and
logdet sensitivity as a function of angular distance from the crystal diagonal.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
    matrix_fields,
    mixed_matrix_gate,
    mixed_scalar_gate,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_matsubara_orbit_gauss


DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/dwave_diagonal_width_scan/"
    "dwave_diagonal_width_scan.json"
)
DEFAULT_DIRECTIONS = (
    (6, 6),
    (12, 12),
    (13, 12),
    (12, 13),
    (13, 11),
    (11, 13),
    (14, 10),
    (10, 14),
    (24, 24),
    (25, 24),
)


def _parse_direction(value: str) -> tuple[int, int]:
    text = str(value).strip()
    pieces = text.replace(":", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            f"direction must have form mx,my; received {value!r}"
        )
    try:
        mx, my = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"direction must contain integers; received {value!r}"
        ) from exc
    if mx == 0 and my == 0:
        raise argparse.ArgumentTypeError("direction (0,0) is forbidden")
    return mx, my


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument(
        "--directions",
        nargs="+",
        type=_parse_direction,
        default=list(DEFAULT_DIRECTIONS),
        metavar="MX,MY",
    )
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gauss-order", type=int, default=384)
    parser.add_argument("--panel-count", type=int, default=16)
    parser.add_argument(
        "--integration-starts",
        nargs=2,
        type=float,
        default=[-np.pi, -np.pi + np.pi / 32.0],
        metavar=("CUT_A", "CUT_B"),
    )
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--max-point-evaluations", type=int, default=2_500_000)
    parser.add_argument("--transverse-workers", type=int, default=8)
    parser.add_argument("--transverse-task-size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--strict-response-rtol", type=float, default=1e-3)
    parser.add_argument("--soft-response-rtol", type=float, default=2e-3)
    parser.add_argument("--observable-rtol", type=float, default=1e-3)
    parser.add_argument("--comparison-atol", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.nk <= 0:
        parser.error("--nk must be positive")
    if args.gauss_order <= 0 or args.panel_count <= 0:
        parser.error("Gauss order and panel count must be positive")
    if args.gauss_order % args.panel_count != 0:
        parser.error("--gauss-order must be divisible by --panel-count")
    if args.max_point_evaluations <= 0:
        parser.error("--max-point-evaluations must be positive")
    if args.transverse_workers <= 0 or args.transverse_task_size <= 0:
        parser.error("worker and task size controls must be positive")
    if any(index < 0 for index in args.matsubara_indices):
        parser.error("Matsubara indices must be non-negative")
    indices = tuple(sorted(set(int(index) for index in args.matsubara_indices)))
    if 0 not in indices or not any(index > 0 for index in indices):
        parser.error("angular-width screen requires n=0 and at least one positive index")
    args.matsubara_indices = indices
    directions = tuple(dict.fromkeys(tuple(map(int, pair)) for pair in args.directions))
    if not directions:
        parser.error("at least one direction is required")
    if any(abs(mx) > args.nk // 2 or abs(my) > args.nk // 2 for mx, my in directions):
        parser.error("all direction indices must lie in the principal periodic range")
    args.directions = directions
    if not np.isfinite(np.asarray(args.integration_starts, dtype=float)).all():
        parser.error("integration starts must be finite")
    for name in (
        "strict_response_rtol",
        "soft_response_rtol",
        "observable_rtol",
        "comparison_atol",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.soft_response_rtol < args.strict_response_rtol:
        parser.error("soft response tolerance cannot be smaller than strict tolerance")
    return args


def _angle_metrics(mx: int, my: int) -> dict[str, float]:
    angle_deg = math.degrees(math.atan2(float(my), float(mx)))
    diagonal_offset_deg = abs(angle_deg - 45.0)
    norm = math.hypot(float(mx), float(my))
    return {
        "angle_deg": angle_deg,
        "diagonal_offset_deg": diagonal_offset_deg,
        "integer_q_norm": norm,
        "normalized_diagonal_distance": abs(float(mx - my)) / max(norm, 1e-300),
    }


def _relative_matrix(left: np.ndarray, right: np.ndarray, atol: float) -> dict[str, float]:
    absolute, relative, ratio, _ = mixed_matrix_gate(left, right, atol=atol, rtol=0.0)
    return {
        "absolute": float(absolute),
        "relative": float(relative),
        "absolute_over_atol": float(ratio),
    }


def _relative_scalar(left: float, right: float, atol: float) -> dict[str, float]:
    absolute, relative, ratio, _ = mixed_scalar_gate(left, right, atol=atol, rtol=0.0)
    return {
        "absolute": float(absolute),
        "relative": float(relative),
        "absolute_over_atol": float(ratio),
    }


def _classify(
    *,
    static_relative: float,
    positive_relative: float,
    reflection_relative: float,
    logdet_relative: float,
    physical: bool,
    strict_response_rtol: float,
    soft_response_rtol: float,
    observable_rtol: float,
) -> str:
    if not physical:
        return "physical_failure"
    if reflection_relative > observable_rtol or logdet_relative > observable_rtol:
        return "observable_unresolved"
    response_max = max(static_relative, positive_relative)
    if response_max <= strict_response_rtol:
        return "response_strict"
    if response_max <= soft_response_rtol:
        return "response_soft"
    return "response_unresolved_observable_stable"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "d-wave diagonal angular-width screen",
        "=" * 124,
        (
            f"nk={config['nk']}; order={config['gauss_order']}; "
            f"panels={config['panel_count']}; indices={tuple(config['matsubara_indices'])}"
        ),
        (
            f"cuts={tuple(config['integration_starts'])}; "
            f"workers/task={config['transverse_workers']}/{config['transverse_task_size']}"
        ),
        "",
        (
            " mx  my   angle offset   |m|      static       positive          R       logdet "
            "physical classification"
        ),
        "-" * 124,
    ]
    ordered = sorted(
        payload["direction_summaries"],
        key=lambda row: (float(row["diagonal_offset_deg"]), float(row["integer_q_norm"])),
    )
    for row in ordered:
        lines.append(
            f"{int(row['mx']):3d} {int(row['my']):3d} "
            f"{float(row['angle_deg']):7.3f} {float(row['diagonal_offset_deg']):6.3f} "
            f"{float(row['integer_q_norm']):7.3f} "
            f"{float(row['static_primary_relative']):12.3e} "
            f"{float(row['positive_primary_relative_max']):12.3e} "
            f"{float(row['reflection_relative_max']):10.3e} "
            f"{float(row['logdet_relative_max']):10.3e} "
            f"{str(bool(row['physical_all'])):>8s} {row['classification']}"
        )
    status = payload["status"]
    lines.extend(
        [
            "",
            f"largest unresolved angular offset = {status['largest_response_unresolved_offset_deg']}",
            f"smallest stable angular offset = {status['smallest_response_stable_offset_deg']}",
            f"all observables stable = {status['all_observables_stable']}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    indices = tuple(args.matsubara_indices)
    xi_values = np.asarray(
        [0.0 if index == 0 else matsubara_energy_eV(index, args.temperature_K) for index in indices],
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

    raw_rows: list[dict[str, Any]] = []
    direction_summaries: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for direction_index, (mx, my) in enumerate(args.directions):
        print(
            f"starting angular-width direction {direction_index + 1}/{len(args.directions)}: "
            f"m=({mx},{my})",
            flush=True,
        )
        cut_values: list[dict[int, dict[str, Any]]] = []
        direction_wall = 0.0
        for cut_index, integration_start in enumerate(args.integration_starts):
            integrated = integrate_matsubara_orbit_gauss(
                spec=model.spec,
                ansatz=ansatz,
                pairing=pairing,
                xi_eV_values=xi_values,
                temperature_K=args.temperature_K,
                eta_eV=args.eta_eV,
                nk=args.nk,
                mx=mx,
                my=my,
                transverse_order=args.gauss_order,
                panel_count=args.panel_count,
                integration_start=float(integration_start),
                shift_s=args.shift_s,
                subgrid_average=args.subgrid_average,
                max_point_evaluations=args.max_point_evaluations,
                transverse_workers=args.transverse_workers,
                transverse_task_size=args.transverse_task_size,
            )
            direction_wall += float(integrated.quadrature.wall_seconds)
            q = np.asarray(integrated.quadrature.q_model, dtype=float)
            by_index: dict[int, dict[str, Any]] = {}
            for index, xi, components, rhs in zip(
                indices,
                integrated.xi_eV_values,
                integrated.components,
                integrated.rhs,
                strict=True,
            ):
                physical = evaluate_matsubara_pipeline(
                    components=components,
                    rhs=rhs,
                    q_model=q,
                    xi_eV=float(xi),
                    config=physics_config,
                )
                primary = np.asarray(physical["primary_response"], dtype=complex)
                reflection = np.asarray(physical["reflection"], dtype=complex)
                record = {
                    "primary": primary,
                    "reflection": reflection,
                    "logdet": float(physical["logdet"]),
                    "physical_passed": bool(physical["physical_passed"]),
                    "ward_passed": bool(physical["ward_passed"]),
                    "strict_static_ward_passed": bool(physical["strict_static_ward_passed"]),
                    "chi_bar": float(physical["chi_bar"]),
                    "dbar_t": float(physical["dbar_t"]),
                }
                by_index[index] = record
                raw_rows.append(
                    {
                        "mx": mx,
                        "my": my,
                        **_angle_metrics(mx, my),
                        "cut_index": cut_index,
                        "integration_start": float(integration_start),
                        "matsubara_index": index,
                        "xi_eV": float(xi),
                        "response_sector": str(physical["response_sector"]),
                        "physical_passed": bool(physical["physical_passed"]),
                        "ward_passed": bool(physical["ward_passed"]),
                        "strict_static_ward_passed": bool(physical["strict_static_ward_passed"]),
                        "chi_bar": float(physical["chi_bar"]),
                        "dbar_t": float(physical["dbar_t"]),
                        "logdet": float(physical["logdet"]),
                        **matrix_fields("primary_response", primary),
                        **matrix_fields("reflection", reflection),
                        "quadrature_wall_seconds": float(integrated.quadrature.wall_seconds),
                        "point_evaluations": int(integrated.quadrature.point_evaluations),
                        "evaluator_callbacks": int(integrated.evaluator_profile.callbacks),
                        "execution_strategy": str(integrated.quadrature.execution_strategy),
                    }
                )
            cut_values.append(by_index)

        left, right = cut_values
        comparisons: dict[int, dict[str, Any]] = {}
        for index in indices:
            first = left[index]
            second = right[index]
            comparisons[index] = {
                "primary": _relative_matrix(
                    first["primary"], second["primary"], args.comparison_atol
                ),
                "reflection": _relative_matrix(
                    first["reflection"], second["reflection"], args.comparison_atol
                ),
                "logdet": _relative_scalar(
                    first["logdet"], second["logdet"], args.comparison_atol
                ),
            }

        zero = comparisons[0]
        positive_indices = [index for index in indices if index > 0]
        static_relative = float(zero["primary"]["relative"])
        positive_relative = max(
            float(comparisons[index]["primary"]["relative"])
            for index in positive_indices
        )
        reflection_relative = max(
            float(comparisons[index]["reflection"]["relative"])
            for index in indices
        )
        logdet_relative = max(
            float(comparisons[index]["logdet"]["relative"])
            for index in indices
        )
        physical_all = all(
            bool(cut[index]["physical_passed"])
            for cut in cut_values
            for index in indices
        )
        classification = _classify(
            static_relative=static_relative,
            positive_relative=positive_relative,
            reflection_relative=reflection_relative,
            logdet_relative=logdet_relative,
            physical=physical_all,
            strict_response_rtol=args.strict_response_rtol,
            soft_response_rtol=args.soft_response_rtol,
            observable_rtol=args.observable_rtol,
        )
        direction_summaries.append(
            {
                "mx": mx,
                "my": my,
                **_angle_metrics(mx, my),
                "static_primary_relative": static_relative,
                "positive_primary_relative_max": positive_relative,
                "reflection_relative_max": reflection_relative,
                "logdet_relative_max": logdet_relative,
                "physical_all": physical_all,
                "strict_static_ward_all": all(
                    bool(cut[0]["strict_static_ward_passed"])
                    for cut in cut_values
                ),
                "classification": classification,
                "direction_wall_seconds": direction_wall,
                "comparison_by_matsubara": {
                    str(index): comparisons[index] for index in indices
                },
            }
        )

    unresolved_offsets = [
        float(row["diagonal_offset_deg"])
        for row in direction_summaries
        if row["classification"] == "response_unresolved_observable_stable"
    ]
    stable_offsets = [
        float(row["diagonal_offset_deg"])
        for row in direction_summaries
        if row["classification"] in {"response_strict", "response_soft"}
    ]
    status = {
        "largest_response_unresolved_offset_deg": (
            max(unresolved_offsets) if unresolved_offsets else None
        ),
        "smallest_response_stable_offset_deg": (
            min(stable_offsets) if stable_offsets else None
        ),
        "all_physical": all(bool(row["physical_all"]) for row in direction_summaries),
        "all_observables_stable": all(
            float(row["reflection_relative_max"]) <= args.observable_rtol
            and float(row["logdet_relative_max"]) <= args.observable_rtol
            for row in direction_summaries
        ),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    payload = {
        "schema": "dwave_diagonal_width_scan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "nk": args.nk,
            "directions": [list(pair) for pair in args.directions],
            "matsubara_indices": list(indices),
            "gauss_order": args.gauss_order,
            "panel_count": args.panel_count,
            "integration_starts": [float(value) for value in args.integration_starts],
            "shift_s": args.shift_s,
            "subgrid_average": args.subgrid_average,
            "max_point_evaluations": args.max_point_evaluations,
            "transverse_workers": args.transverse_workers,
            "transverse_task_size": args.transverse_task_size,
            "strict_response_rtol": args.strict_response_rtol,
            "soft_response_rtol": args.soft_response_rtol,
            "observable_rtol": args.observable_rtol,
            "comparison_atol": args.comparison_atol,
        },
        "direction_summaries": direction_summaries,
        "raw_rows": raw_rows,
        "status": status,
        "total_wall_seconds": float(time.perf_counter() - started_all),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stem = args.output.with_suffix("")
    csv_rows = [
        {key: value for key, value in row.items() if key != "comparison_by_matsubara"}
        for row in direction_summaries
    ]
    _write_csv(stem.with_name(stem.name + ".directions.csv"), csv_rows)
    _write_csv(stem.with_name(stem.name + ".raw.csv"), raw_rows)
    summary = _summary_text(payload)
    stem.with_name(stem.name + ".summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
