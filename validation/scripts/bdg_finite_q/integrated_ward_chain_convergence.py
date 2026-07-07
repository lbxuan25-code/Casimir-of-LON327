#!/usr/bin/env python3
"""Write integrated finite-q Ward chain convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from validation.lib.finite_q_integrated_ward_convergence import run_integrated_ward_chain_convergence  # noqa: E402
from validation.lib.finite_q_validation_models import available_finite_q_validation_models  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "validation" / "outputs" / "integrated_ward_chain_convergence"


def _command_text(args: argparse.Namespace) -> str:
    rel_script = "validation/scripts/bdg_finite_q/integrated_ward_chain_convergence.py"
    try:
        output_dir = str(args.output_dir.resolve().relative_to(ROOT))
    except ValueError:
        output_dir = str(args.output_dir)
    parts = [
        "python",
        rel_script,
        "--model",
        args.model,
        "--output-dir",
        output_dir,
        "--pairings",
        *args.pairings,
        "--omega",
        str(args.omega),
        "--delta0",
        str(args.delta0),
        "--q-values",
        *(str(value) for value in args.q_values),
        "--q-direction",
        str(args.q_direction[0]),
        str(args.q_direction[1]),
        "--nk-values",
        *(str(value) for value in args.nk_values),
        "--shift-fractions",
        *(str(value) for value in args.shift_fractions),
    ]
    return " ".join(shlex.quote(part) for part in parts) + "\n"


def _format_text(report: dict[str, Any]) -> str:
    lines = [
        "integrated finite-q Ward chain convergence",
        f"model_name: {report['model_name']}",
        f"pairings: {', '.join(report['pairings'])}",
        f"omega_eV: {report['omega_eV']:.12g}",
        f"delta0_eV: {report['delta0_eV']:.12g}",
        f"q_values: {report['q_values']}",
        f"nk_values: {report['nk_values']}",
        f"shift_fractions: {report['shift_fractions']}",
        "",
        "summary rows:",
    ]
    for item in report["summaries"]:
        unshifted = item.get("unshifted") or {}
        avg = item["shifted_mesh_average"]
        avg_equal = avg["equal_time_to_contact_difference"]
        avg_full = avg["full_chain_residual"]
        lines.append(
            f"- {item['pairing_name']} q={item['q_model']} nk={item['nk']}: "
            f"unshifted_equal={unshifted.get('equal_time_to_contact')} "
            f"shift_mean_equal={avg_equal['mean_norm']} "
            f"shift_rms_equal={avg_equal['rms_norm']} "
            f"shift_mean_full={avg_full['mean_norm']}"
        )
    lines.extend(["", "valid_for_casimir_input: False"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path, command_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "convergence.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "convergence.txt").write_text(_format_text(report), encoding="utf-8")
    (output_dir / "command.sh").write_text(command_text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated finite-q Ward chain convergence diagnostics.")
    parser.add_argument("--model", choices=available_finite_q_validation_models(), default="symmetry_bdg_2band")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairings", nargs="+", default=["spm", "dwave"])
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--delta0", type=float, default=0.1)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--q-direction", nargs=2, type=float, default=[1.0, 0.0])
    parser.add_argument("--nk-values", nargs="+", type=int, default=[9, 13, 17])
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_integrated_ward_chain_convergence(
        model_name=args.model,
        pairings=tuple(args.pairings),
        q_values=tuple(args.q_values),
        q_direction=(float(args.q_direction[0]), float(args.q_direction[1])),
        nk_values=tuple(args.nk_values),
        shift_fractions=tuple(args.shift_fractions),
        omega_eV=float(args.omega),
        delta0_eV=float(args.delta0),
    )
    write_report(report, args.output_dir, _command_text(args))
    print(_format_text(report), end="")
    print(f"Wrote integrated Ward chain convergence report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
