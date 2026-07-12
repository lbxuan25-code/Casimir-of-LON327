"""Average complementary d-wave phase-column JSON payloads before Ward ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.lib.dwave_phase_column_subgrid_average import (
    average_dwave_phase_column_payloads,
)
from validation.lib.dwave_phase_hessian_analysis import (
    analyze_dwave_phase_hessian_payload,
    complex_jsonable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    averaged = average_dwave_phase_column_payloads(
        payloads,
        labels=[path.name for path in args.inputs],
    )
    analysis = analyze_dwave_phase_hessian_payload(averaged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(complex_jsonable(averaged), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    required = 0.5 * (
        complex(analysis.left.required_counterterm_multiplier)
        + complex(analysis.right.required_counterterm_multiplier)
    )
    q_norm = float(analysis.q_norm)
    current_defect = max(
        abs(complex(analysis.left.current_phase_defect)),
        abs(complex(analysis.right.current_phase_defect)),
    ) / q_norm
    bond_defect = max(
        abs(complex(analysis.left.bond_metric_phase_defect)),
        abs(complex(analysis.right.bond_metric_phase_defect)),
    ) / q_norm

    print("d-wave complementary-subgrid phase-column average")
    print("=" * 55)
    print(f"sources = {len(args.inputs)}")
    for path in args.inputs:
        print(f"  {path}")
    print(
        "q = "
        f"({analysis.q_model[0]:.12g}, {analysis.q_model[1]:.12g}); "
        f"|q| = {q_norm:.12g}"
    )
    print(f"current phase defect / |q| = {current_defect:.12e}")
    print(f"required multiplier        = {required.real:+.12e}{required.imag:+.12e}j")
    print(f"bond metric multiplier     = {analysis.bond_metric_multiplier:.12e}")
    print(f"bond multiplier error      = {abs(analysis.bond_metric_multiplier - required):.12e}")
    print(f"bond defect / |q|          = {bond_defect:.12e}")
    print("")
    print("Fail-closed status")
    print("------------------")
    print("subgrid_averaged = True")
    print("diagnostic_only = True")
    print("projection_applied = False")
    print("production_reference_established = False")
    print("valid_for_casimir_input = False")
    print(f"output = {args.output}")


if __name__ == "__main__":
    main()
