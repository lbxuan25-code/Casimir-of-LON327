"""Analyze d-wave phase-Hessian hypotheses from an existing commensurate audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.lib.dwave_phase_hessian_analysis import (
    analyze_dwave_phase_hessian_payload,
    complex_jsonable,
)


DEFAULT_INPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw/"
    "dwave_commensurate_n628_m3_2_T10.json"
)


def _format_complex(value: complex) -> str:
    scalar = complex(value)
    return f"{scalar.real:+.12e}{scalar.imag:+.12e}j"


def _relative_shift_mismatch(candidate: complex, required: complex) -> float:
    return float(abs(candidate - required) / max(abs(1.0 - required), 1e-30))


def _print_side(name: str, side: Any, q_norm: float, bond_metric: float, direct_multiplier: complex | None) -> None:
    required = complex(side.required_counterterm_multiplier)
    print(f"\n{name} phase column")
    print("-" * (len(name) + 13))
    print(f"EM-collective phase / |q|       = {abs(side.em_collective_phase) / q_norm:.12e}")
    print(f"bubble rotation phase / |q|     = {abs(side.phase_rotation_bubble) / q_norm:.12e}")
    print(f"counterterm rotation / |q|      = {abs(side.phase_rotation_counterterm) / q_norm:.12e}")
    print(f"current defect / |q|            = {abs(side.current_phase_defect) / q_norm:.12e}")
    print(f"required multiplier             = {_format_complex(required)}")
    print(f"bond metric multiplier          = {bond_metric:.12e}")
    print(f"bond metric defect / |q|        = {abs(side.bond_metric_phase_defect) / q_norm:.12e}")
    print(
        "bond shift mismatch / required shift = "
        f"{_relative_shift_mismatch(complex(bond_metric), required):.12e}"
    )
    if direct_multiplier is not None and side.phase_direct_phase_defect is not None:
        print(f"phase-direct multiplier         = {_format_complex(direct_multiplier)}")
        print(
            "phase-direct defect / |q|       = "
            f"{abs(side.phase_direct_phase_defect) / q_norm:.12e}"
        )
        print(
            "phase-direct shift mismatch     = "
            f"{_relative_shift_mismatch(direct_multiplier, required):.12e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_dwave_phase_hessian_payload(payload)

    print("d-wave commensurate phase-Hessian post-analysis")
    print("=" * 50)
    print(f"input = {args.input}")
    print(
        "q = "
        f"({analysis.q_model[0]:.12g}, {analysis.q_model[1]:.12g}); "
        f"|q| = {analysis.q_norm:.12g}"
    )
    print(f"delta0_eV = {analysis.delta0_eV:.12g}")
    print(f"bond metric = {analysis.bond_metric_multiplier:.12e}")
    print(f"counterterm curvature = {_format_complex(analysis.counterterm_curvature)}")
    if analysis.phase_direct_plus is not None:
        print(f"phase direct plus = {_format_complex(analysis.phase_direct_plus)}")
        print(f"phase-direct curvature = {_format_complex(analysis.phase_direct_curvature)}")
        print(
            "phase-direct / counterterm multiplier = "
            f"{_format_complex(analysis.phase_direct_counterterm_multiplier)}"
        )

    _print_side(
        "left",
        analysis.left,
        analysis.q_norm,
        analysis.bond_metric_multiplier,
        analysis.phase_direct_counterterm_multiplier,
    )
    _print_side(
        "right",
        analysis.right,
        analysis.q_norm,
        analysis.bond_metric_multiplier,
        analysis.phase_direct_counterterm_multiplier,
    )

    print("\nFail-closed status")
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
