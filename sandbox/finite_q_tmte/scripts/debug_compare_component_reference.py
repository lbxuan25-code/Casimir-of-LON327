#!/usr/bin/env python3
"""Debug-only comparison against existing component-basis finite-q response."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.adapters.model_adapter import available_models  # noqa: E402
from tmte.pipeline.scan_runner import debug_compare_component_reference  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug-only component reference comparison.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--xi", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--omega", type=float, default=0.01)
    args = parser.parse_args(argv)
    payload = debug_compare_component_reference(
        model_name=args.model,
        pairing_name=args.pairing,
        xi=args.xi,
        q_value=args.q,
        nk=args.nk,
        omega_eV=args.omega,
    )
    for key, value in payload.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
