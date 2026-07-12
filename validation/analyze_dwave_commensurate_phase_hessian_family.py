"""Fit the q scaling of d-wave phase-Hessian candidates from audit JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.lib.dwave_phase_hessian_analysis import complex_jsonable
from validation.lib.dwave_phase_hessian_scaling import (
    analyze_dwave_phase_hessian_family,
)


def _format_optional(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    analysis = analyze_dwave_phase_hessian_family(
        payloads,
        labels=[path.name for path in args.inputs],
    )

    print("d-wave commensurate phase-Hessian q-scaling analysis")
    print("=" * 58)
    print(
        "direction = "
        f"({analysis.direction[0]:.12g}, {analysis.direction[1]:.12g})"
    )
    print(f"delta0_eV = {analysis.delta0_eV:.12g}")
    print("")
    print(
        "q_norm          required_shift   bond_shift       |bond-required| "
        "current_defect/q bond_defect/q"
    )
    print("-" * 104)
    for point in analysis.points:
        print(
            f"{point.q_norm: .12e} "
            f"{point.required_shift_abs: .12e} "
            f"{point.bond_shift_abs: .12e} "
            f"{point.bond_multiplier_error_abs: .12e} "
            f"{point.current_phase_defect_over_q: .12e} "
            f"{point.bond_phase_defect_over_q: .12e}"
        )
        print(f"  {point.label}")

    print("")
    print("Power-law fits y ~ q^p")
    print("----------------------")
    print(
        "required counterterm shift p = "
        f"{_format_optional(analysis.required_shift_exponent)}"
    )
    print(f"bond metric shift p         = {_format_optional(analysis.bond_shift_exponent)}")
    print(f"bond multiplier error p     = {_format_optional(analysis.bond_error_exponent)}")
    print(
        "current phase defect/q p     = "
        f"{_format_optional(analysis.current_defect_over_q_exponent)}"
    )
    print(
        "bond phase defect/q p        = "
        f"{_format_optional(analysis.bond_defect_over_q_exponent)}"
    )
    print(
        "required pairwise p          = "
        + ", ".join(f"{value:.8f}" for value in analysis.required_shift_pairwise_exponents)
    )
    print(
        "bond-error pairwise p        = "
        + ", ".join(f"{value:.8f}" for value in analysis.bond_error_pairwise_exponents)
    )

    print("")
    print("Classification")
    print("--------------")
    print(analysis.classification)
    print(analysis.interpretation)

    print("")
    print("Fail-closed status")
    print("------------------")
    print("diagnostic_only = True")
    print("projection_applied = False")
    print("production_reference_established = False")
    print("valid_for_casimir_input = False")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(complex_jsonable(analysis.to_dict()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"output = {args.output}")


if __name__ == "__main__":
    main()
