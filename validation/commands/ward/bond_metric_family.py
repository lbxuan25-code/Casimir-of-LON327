"""Run a resumable commensurate d-wave bond-metric static Ward family.

Each point delegates to ``ward bond-metric-full-kernel`` so the complete
48-component primitive vector, complementary-subgrid averaging, and Schur
assembly remain defined in one place.  This family command then applies an
independent hard q-normalized closure gate and evaluates available C4 partners.

All outputs remain diagnostic-only and invalid for Casimir input.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


DEFAULT_POINTS = ("2,0", "0,2", "2,2", "4,2", "2,4", "3,2", "2,3")
DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/families/"
    "dwave_bond_metric_static_family_n628.csv"
)


def parse_integer_point(text: str) -> tuple[int, int]:
    """Parse one ``mx,my`` point and reject q=0."""

    pieces = str(text).split(",")
    if len(pieces) != 2:
        raise ValueError(f"point must have form mx,my, got {text!r}")
    try:
        point = (int(pieces[0].strip()), int(pieces[1].strip()))
    except ValueError as exc:
        raise ValueError(f"point must contain integers, got {text!r}") from exc
    if point == (0, 0):
        raise ValueError("commensurate family does not accept q=0")
    return point


def _relative_difference(left: float, right: float) -> float:
    return float(abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-30))


def _q_normalized_effective_residual(payload: dict[str, Any], branch: str) -> float:
    audit = payload[branch]["ward_audit"]
    return max(
        float(audit["left"]["q_normalized_norms"]["effective_residual"]),
        float(audit["right"]["q_normalized_norms"]["effective_residual"]),
    )


def strict_gate_from_point_payload(
    payload: dict[str, Any],
    branch: str,
    *,
    primitive_tolerance: float,
    amplitude_tolerance: float,
    phase_tolerance: float,
    effective_direct_tolerance: float,
    effective_residual_tolerance: float,
    longitudinal_tolerance: float,
    condition_max: float,
) -> dict[str, Any]:
    """Evaluate the strict static scalar gate from one full-kernel payload."""

    if branch not in {"baseline", "corrected"}:
        raise ValueError("branch must be 'baseline' or 'corrected'")
    row = payload["row"]
    prefix = f"{branch}_"
    values = {
        "primitive_residual_over_q": float(row[prefix + "primitive_residual_over_q"]),
        "amplitude_defect_over_q": float(row[prefix + "amplitude_defect_over_q"]),
        "phase_defect_over_q": float(row[prefix + "phase_defect_over_q"]),
        "effective_direct_over_q": float(row[prefix + "effective_direct_over_q"]),
        "effective_residual_over_q": _q_normalized_effective_residual(payload, branch),
        "relative_longitudinal_gauge_residual": float(row[prefix + "raw_longitudinal"]),
        "schur_condition_number": float(row[prefix + "collective_condition"]),
        "schur_inverse_method": str(row[prefix + "collective_inverse_method"]),
    }
    passed = bool(
        values["primitive_residual_over_q"] <= primitive_tolerance
        and values["amplitude_defect_over_q"] <= amplitude_tolerance
        and values["phase_defect_over_q"] <= phase_tolerance
        and values["effective_direct_over_q"] <= effective_direct_tolerance
        and values["effective_residual_over_q"] <= effective_residual_tolerance
        and values["relative_longitudinal_gauge_residual"] <= longitudinal_tolerance
        and values["schur_inverse_method"] == "inv"
        and values["schur_condition_number"] <= condition_max
    )
    return {
        **values,
        "primitive_tolerance": float(primitive_tolerance),
        "amplitude_tolerance": float(amplitude_tolerance),
        "phase_tolerance": float(phase_tolerance),
        "effective_direct_tolerance": float(effective_direct_tolerance),
        "effective_residual_tolerance": float(effective_residual_tolerance),
        "longitudinal_tolerance": float(longitudinal_tolerance),
        "condition_max": float(condition_max),
        "passed": passed,
        "criterion": "strict_static_q_normalized_v1",
    }


def c4_comparisons(
    rows: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compare available ``(mx,my)`` and ``(my,mx)`` corrected observables."""

    by_point = {(int(row["mx"]), int(row["my"])): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for point, row in by_point.items():
        partner = (point[1], point[0])
        if partner == point or partner not in by_point:
            continue
        pair_key = frozenset((point, partner))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        other = by_point[partner]
        metrics = {
            "q_norm_relative_difference": _relative_difference(row["q_norm"], other["q_norm"]),
            "chi_bar_relative_difference": _relative_difference(
                row["corrected_chi_bar"], other["corrected_chi_bar"]
            ),
            "dbar_t_relative_difference": _relative_difference(
                row["corrected_dbar_t"], other["corrected_dbar_t"]
            ),
            "phase_defect_absolute_difference": abs(
                float(row["corrected_phase_defect_over_q"])
                - float(other["corrected_phase_defect_over_q"])
            ),
            "effective_direct_absolute_difference": abs(
                float(row["corrected_effective_direct_over_q"])
                - float(other["corrected_effective_direct_over_q"])
            ),
        }
        comparisons.append(
            {
                "point": point,
                "partner": partner,
                **metrics,
                "tolerance": float(tolerance),
                "passed": bool(
                    metrics["q_norm_relative_difference"] <= tolerance
                    and metrics["chi_bar_relative_difference"] <= tolerance
                    and metrics["dbar_t_relative_difference"] <= tolerance
                ),
            }
        )
    return comparisons


def _point_output(root: Path, nk: int, mx: int, my: int) -> Path:
    return root / f"dwave_bond_metric_full_kernel_n{nk}_m{mx}_{my}.csv"


def _load_point_payload(path: Path, nk: int, mx: int, my: int) -> dict[str, Any]:
    json_path = path.with_suffix(".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "dwave_bond_metric_full_kernel_audit_v1":
        raise ValueError(f"unexpected point schema in {json_path}")
    row = payload["row"]
    if (int(row["nk"]), int(row["mx"]), int(row["my"])) != (nk, mx, my):
        raise ValueError(f"point payload does not match requested point {(nk, mx, my)}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "d-wave commensurate bond-metric static Ward family",
        "=" * 57,
        " mx  my   |q|        subgrids  phase/q      eff-direct/q  longitudinal  strict",
        "-" * 88,
    ]
    for row in rows:
        lines.append(
            f"{int(row['mx']):3d} {int(row['my']):3d} "
            f"{float(row['q_norm']):10.4e} {int(row['subgrid_count']):9d} "
            f"{float(row['corrected_phase_defect_over_q']):12.3e} "
            f"{float(row['corrected_effective_direct_over_q']):13.3e} "
            f"{float(row['corrected_raw_longitudinal']):12.3e} "
            f"{str(bool(row['corrected_strict_gate_passed'])):>7s}"
        )
    lines.extend(["", "C4 comparisons", "--------------"])
    if comparisons:
        for item in comparisons:
            lines.append(
                f"{item['point']} <-> {item['partner']}: "
                f"chi={item['chi_bar_relative_difference']:.3e}, "
                f"D_T={item['dbar_t_relative_difference']:.3e}, "
                f"passed={item['passed']}"
            )
    else:
        lines.append("no C4 partner pairs were present")
    lines.extend(
        [
            "",
            f"all strict point gates passed = {all(bool(row['corrected_strict_gate_passed']) for row in rows)}",
            f"all available C4 comparisons passed = {all(bool(item['passed']) for item in comparisons)}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=628)
    parser.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS))
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--primitive-tolerance", type=float, default=1e-9)
    parser.add_argument("--amplitude-tolerance", type=float, default=1e-9)
    parser.add_argument("--phase-tolerance", type=float, default=1e-9)
    parser.add_argument("--effective-direct-tolerance", type=float, default=1e-9)
    parser.add_argument("--effective-residual-tolerance", type=float, default=1e-9)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-9)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--c4-tolerance", type=float, default=1e-8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.nk <= 0 or args.chunk_size <= 0 or args.max_points <= 0:
        parser.error("--nk, --chunk-size, and --max-points must be positive")
    points: list[tuple[int, int]] = []
    try:
        for item in args.points:
            point = parse_integer_point(item)
            if point not in points:
                points.append(point)
    except ValueError as exc:
        parser.error(str(exc))
    for name in (
        "primitive_tolerance",
        "amplitude_tolerance",
        "phase_tolerance",
        "effective_direct_tolerance",
        "effective_residual_tolerance",
        "longitudinal_tolerance",
        "c4_tolerance",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not np.isfinite(args.condition_max) or args.condition_max <= 0.0:
        parser.error("--condition-max must be finite and positive")

    point_root = args.output.parent / (args.output.stem + "_points")
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []

    for index, (mx, my) in enumerate(points, start=1):
        point_output = _point_output(point_root, args.nk, mx, my)
        json_path = point_output.with_suffix(".json")
        if args.resume and point_output.is_file() and json_path.is_file():
            print(f"[{index}/{len(points)}] reusing {json_path}", flush=True)
        else:
            point_output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "validation",
                "ward",
                "bond-metric-full-kernel",
                "--nk",
                str(args.nk),
                "--mx",
                str(mx),
                "--my",
                str(my),
                "--shift-x",
                str(args.shift_x),
                "--shift-y",
                str(args.shift_y),
                "--subgrid-average",
                "auto",
                "--chunk-size",
                str(args.chunk_size),
                "--max-points",
                str(args.max_points),
                "--temperature-K",
                str(args.temperature_K),
                "--delta0-eV",
                str(args.delta0_eV),
                "--eta-eV",
                str(args.eta_eV),
                "--condition-max",
                str(args.condition_max),
                "--output",
                str(point_output),
            ]
            print(f"[{index}/{len(points)}] running m=({mx},{my})", flush=True)
            subprocess.run(command, check=True)

        payload = _load_point_payload(point_output, args.nk, mx, my)
        baseline_gate = strict_gate_from_point_payload(
            payload,
            "baseline",
            primitive_tolerance=args.primitive_tolerance,
            amplitude_tolerance=args.amplitude_tolerance,
            phase_tolerance=args.phase_tolerance,
            effective_direct_tolerance=args.effective_direct_tolerance,
            effective_residual_tolerance=args.effective_residual_tolerance,
            longitudinal_tolerance=args.longitudinal_tolerance,
            condition_max=args.condition_max,
        )
        corrected_gate = strict_gate_from_point_payload(
            payload,
            "corrected",
            primitive_tolerance=args.primitive_tolerance,
            amplitude_tolerance=args.amplitude_tolerance,
            phase_tolerance=args.phase_tolerance,
            effective_direct_tolerance=args.effective_direct_tolerance,
            effective_residual_tolerance=args.effective_residual_tolerance,
            longitudinal_tolerance=args.longitudinal_tolerance,
            condition_max=args.condition_max,
        )
        row = dict(payload["row"])
        row.update(
            {
                "baseline_effective_residual_over_q": baseline_gate[
                    "effective_residual_over_q"
                ],
                "corrected_effective_residual_over_q": corrected_gate[
                    "effective_residual_over_q"
                ],
                "baseline_strict_gate_passed": bool(baseline_gate["passed"]),
                "corrected_strict_gate_passed": bool(corrected_gate["passed"]),
                "strict_gate_criterion": "strict_static_q_normalized_v1",
                "point_json": str(json_path),
            }
        )
        rows.append(row)
        payloads.append(
            {
                "point": (mx, my),
                "source": str(json_path),
                "baseline_strict_gate": baseline_gate,
                "corrected_strict_gate": corrected_gate,
            }
        )
        _write_csv(args.output, rows)

    comparisons = c4_comparisons(rows, args.c4_tolerance)
    family_passed = bool(
        all(bool(row["corrected_strict_gate_passed"]) for row in rows)
        and all(bool(item["passed"]) for item in comparisons)
    )
    metadata = {
        "schema": "dwave_bond_metric_static_family_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "point_gates": payloads,
        "c4_comparisons": comparisons,
        "family_gate": {
            "all_corrected_strict_points_passed": all(
                bool(row["corrected_strict_gate_passed"]) for row in rows
            ),
            "all_available_c4_comparisons_passed": all(
                bool(item["passed"]) for item in comparisons
            ),
            "passed": family_passed,
        },
        "status": {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary = _summary(rows, comparisons)
    args.output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"CSV:     {args.output}")
    print(f"JSON:    {args.output.with_suffix('.json')}")
    print(f"summary: {args.output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
