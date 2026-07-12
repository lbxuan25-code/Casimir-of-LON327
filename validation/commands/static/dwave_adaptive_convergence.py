"""Resumable fixed-q convergence scan for bond-metric adaptive d-wave response.

Each level runs the public ``dwave-iterated-adaptive`` command in both requested
nesting orders with an explicitly tighter quadrature tolerance and point budget.
The final gate requires the finest level to pass its adaptive, strict-Ward, sheet,
and xy/yx checks, and also requires the finest two levels to agree in chi_bar and
Dbar_T for every nesting order.

Level syntax is ``label:epsabs:epsrel:max_points[:inner_limit[:outer_limit]]``.
All outputs remain diagnostic-only and invalid for Casimir input.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

from validation.lib.dwave_global_extrapolation import relative_difference


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_iterated_adaptive/raw/"
    "dwave_bond_metric_adaptive_convergence.csv"
)


@dataclass(frozen=True)
class AdaptiveLevel:
    label: str
    epsabs: float
    epsrel: float
    max_point_evaluations: int
    inner_limit: int
    outer_limit: int


def parse_level(value: str) -> AdaptiveLevel:
    parts = str(value).split(":")
    if len(parts) not in {4, 5, 6}:
        raise argparse.ArgumentTypeError(
            "adaptive level must be label:epsabs:epsrel:max_points"
            "[:inner_limit[:outer_limit]]"
        )
    label = parts[0].strip()
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(
            "adaptive level label must use only letters, numbers, dot, underscore or dash"
        )
    try:
        epsabs = float(parts[1])
        epsrel = float(parts[2])
        max_points = int(parts[3])
        inner = int(parts[4]) if len(parts) >= 5 else 160
        outer = int(parts[5]) if len(parts) >= 6 else inner
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid adaptive level {value!r}") from exc
    if not np.isfinite(epsabs) or epsabs <= 0.0:
        raise argparse.ArgumentTypeError("adaptive level epsabs must be finite and positive")
    if not np.isfinite(epsrel) or epsrel <= 0.0:
        raise argparse.ArgumentTypeError("adaptive level epsrel must be finite and positive")
    if max_points <= 0 or inner <= 0 or outer <= 0:
        raise argparse.ArgumentTypeError("adaptive level budgets and limits must be positive")
    return AdaptiveLevel(label, epsabs, epsrel, max_points, inner, outer)


def _point_output(base: Path, level: AdaptiveLevel) -> Path:
    return base.parent / f"{base.stem}_level-{level.label}.csv"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _add_cross_level_metrics(rows: list[dict[str, Any]]) -> None:
    by_order: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_order.setdefault(str(row["order"]), []).append(row)
    for group in by_order.values():
        group.sort(key=lambda row: int(row["level_index"]))
        previous: dict[str, Any] | None = None
        for row in group:
            if previous is None:
                row["chi_bar_relative_to_previous_level"] = float("nan")
                row["dbar_t_relative_to_previous_level"] = float("nan")
                row["phase_defect_relative_to_previous_level"] = float("nan")
                row["longitudinal_relative_to_previous_level"] = float("nan")
            else:
                row["chi_bar_relative_to_previous_level"] = relative_difference(
                    row["chi_bar"], previous["chi_bar"]
                )
                row["dbar_t_relative_to_previous_level"] = relative_difference(
                    row["dbar_t"], previous["dbar_t"]
                )
                row["phase_defect_relative_to_previous_level"] = relative_difference(
                    row["phase_defect_over_q"], previous["phase_defect_over_q"]
                )
                row["longitudinal_relative_to_previous_level"] = relative_difference(
                    row["relative_longitudinal_gauge_residual"],
                    previous["relative_longitudinal_gauge_residual"],
                )
            previous = row
    rows.sort(key=lambda row: (int(row["level_index"]), str(row["order"])))


def _summary(
    rows: list[dict[str, Any]],
    level_payloads: list[dict[str, Any]],
    observable_tolerance: float,
    convergence_passed: bool,
) -> str:
    lines = [
        "d-wave bond-metric adaptive budget convergence",
        "=" * 51,
        " level       order points   epsabs    epsrel    phase/q   longitudinal "
        "strict      chi_bar       Dbar_T    rel-chi    rel-D",
        "-" * 126,
    ]
    for row in rows:
        rel_chi = float(row["chi_bar_relative_to_previous_level"])
        rel_dbar = float(row["dbar_t_relative_to_previous_level"])
        lines.append(
            f" {str(row['level_label']):<11s} {str(row['order']):>3s} "
            f"{int(row['point_evaluations']):7d} "
            f"{float(row['epsabs']):9.2e} {float(row['epsrel']):9.2e} "
            f"{float(row['phase_defect_over_q']):10.3e} "
            f"{float(row['relative_longitudinal_gauge_residual']):12.3e} "
            f"{str(bool(row['strict_gate_passed'])):>6s} "
            f"{float(row['chi_bar']):12.8f} {float(row['dbar_t']):12.8f} "
            f"{rel_chi:10.3e} {rel_dbar:10.3e}"
        )
    lines.extend(
        [
            "",
            "Level gates",
            "-----------",
        ]
    )
    for payload in level_payloads:
        comparison = payload["comparison"]
        lines.append(
            f"{payload['level'].label}: adaptive_feasibility_pass="
            f"{bool(comparison['adaptive_feasibility_pass'])}, "
            f"order_agreement={float(comparison['physical_order_disagreement']):.3e}"
        )
    lines.extend(
        [
            "",
            f"observable relative tolerance = {observable_tolerance:.3e}",
            f"adaptive budget convergence passed = {convergence_passed}",
            "projection_applied = False",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=parse_level, required=True)
    parser.add_argument("--orders", nargs="+", choices=("xy", "yx"), default=["xy", "yx"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quadrature", choices=("gk15", "gk21", "trapezoid"), default="gk15")
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--cache-size-bytes", type=int, default=64_000_000)
    parser.add_argument("--split-points", type=float, nargs="*", default=[0.0])
    parser.add_argument("--order-agreement-tolerance", type=float, default=5e-3)
    parser.add_argument("--observable-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--qx", type=float, default=0.0300152164356)
    parser.add_argument("--qy", type=float, default=0.0200101442904)
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

    labels = [level.label for level in args.levels]
    if len(set(labels)) != len(labels):
        parser.error("adaptive level labels must be unique")
    if len(set(args.orders)) != len(args.orders):
        parser.error("--orders entries must be distinct")
    if len(args.levels) < 2:
        parser.error("at least two adaptive levels are required for convergence")
    if args.observable_relative_tolerance <= 0.0:
        parser.error("--observable-relative-tolerance must be positive")
    return args


def _command(args: argparse.Namespace, level: AdaptiveLevel, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "validation",
        "static",
        "dwave-iterated-adaptive",
        "--orders",
        *args.orders,
        "--level-label",
        level.label,
        "--epsabs",
        str(level.epsabs),
        "--epsrel",
        str(level.epsrel),
        "--inner-limit",
        str(level.inner_limit),
        "--outer-limit",
        str(level.outer_limit),
        "--max-point-evaluations",
        str(level.max_point_evaluations),
        "--cache-size-bytes",
        str(args.cache_size_bytes),
        "--quadrature",
        args.quadrature,
        "--norm",
        args.norm,
        "--order-agreement-tolerance",
        str(args.order_agreement_tolerance),
        "--qx",
        str(args.qx),
        "--qy",
        str(args.qy),
        "--temperature-K",
        str(args.temperature_K),
        "--delta0-eV",
        str(args.delta0_eV),
        "--eta-eV",
        str(args.eta_eV),
        "--mixed-ward-tolerance",
        str(args.mixed_ward_tolerance),
        "--mixed-ward-absolute-tolerance",
        str(args.mixed_ward_absolute_tolerance),
        "--primitive-tolerance",
        str(args.primitive_tolerance),
        "--amplitude-tolerance",
        str(args.amplitude_tolerance),
        "--phase-tolerance",
        str(args.phase_tolerance),
        "--effective-direct-tolerance",
        str(args.effective_direct_tolerance),
        "--effective-residual-tolerance",
        str(args.effective_residual_tolerance),
        "--longitudinal-tolerance",
        str(args.longitudinal_tolerance),
        "--condition-max",
        str(args.condition_max),
        "--reality-tolerance",
        str(args.reality_tolerance),
        "--mixing-tolerance",
        str(args.mixing_tolerance),
        "--passivity-tolerance",
        str(args.passivity_tolerance),
        "--energy-scale-eV",
        str(args.energy_scale_eV),
        "--degeneracy",
        str(args.degeneracy),
    ]
    if args.split_points:
        command.extend(["--split-points", *[str(value) for value in args.split_points]])
    else:
        command.append("--split-points")
    command.extend(["--output", str(output)])
    return command


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    level_payloads: list[dict[str, Any]] = []

    for level_index, level in enumerate(args.levels):
        point_output = _point_output(args.output, level)
        json_path = point_output.with_suffix(".json")
        if args.resume and json_path.exists():
            print(f"reusing adaptive level {level.label}: {json_path}", flush=True)
        else:
            print(
                f"starting adaptive level {level.label}: epsabs={level.epsabs:.3e}, "
                f"epsrel={level.epsrel:.3e}, budget={level.max_point_evaluations}",
                flush=True,
            )
            subprocess.run(_command(args, level, point_output), check=True)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for source_row in payload["rows"]:
            row = dict(source_row)
            row["level_index"] = level_index
            row["level_json"] = str(json_path)
            aggregate_rows.append(row)
        level_payloads.append(
            {
                "level": level,
                "comparison": dict(payload["comparison"]),
                "source": str(json_path),
            }
        )
        _add_cross_level_metrics(aggregate_rows)
        _write_csv(args.output, aggregate_rows)

    finest_index = len(args.levels) - 1
    finest_rows = [
        row for row in aggregate_rows if int(row["level_index"]) == finest_index
    ]
    observable_converged = bool(
        finest_rows
        and all(
            np.isfinite(float(row["chi_bar_relative_to_previous_level"]))
            and np.isfinite(float(row["dbar_t_relative_to_previous_level"]))
            and float(row["chi_bar_relative_to_previous_level"])
            <= args.observable_relative_tolerance
            and float(row["dbar_t_relative_to_previous_level"])
            <= args.observable_relative_tolerance
            for row in finest_rows
        )
    )
    finest_feasible = bool(
        level_payloads[-1]["comparison"]["adaptive_feasibility_pass"]
    )
    convergence_passed = bool(observable_converged and finest_feasible)

    summary = _summary(
        aggregate_rows,
        level_payloads,
        args.observable_relative_tolerance,
        convergence_passed,
    )
    summary_path = args.output.with_suffix(".summary.txt")
    json_path = args.output.with_suffix(".json")
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_bond_metric_adaptive_convergence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            key: (
                [level.__dict__ for level in value]
                if key == "levels"
                else str(value) if isinstance(value, Path) else value
            )
            for key, value in vars(args).items()
        },
        "rows": aggregate_rows,
        "levels": [
            {
                "definition": item["level"].__dict__,
                "comparison": item["comparison"],
                "source": item["source"],
            }
            for item in level_payloads
        ],
        "convergence_gate": {
            "finest_level_adaptive_feasibility_pass": finest_feasible,
            "finest_pair_observables_converged": observable_converged,
            "observable_relative_tolerance": args.observable_relative_tolerance,
            "passed": convergence_passed,
        },
        "status": {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(summary, end="")
    print(f"CSV:     {args.output}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
