#!/usr/bin/env python3
"""Debug-only primitive response closure suite CLI."""

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
from tmte.pipeline.primitive_response_closure_suite import (  # noqa: E402
    DEFAULT_CANDIDATE,
    run_and_write_primitive_response_closure_suite,
)
from tmte.pipeline.primitive_response_ward_audit import primitive_ward_candidate_vectors  # noqa: E402


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _candidate_choices() -> tuple[str, ...]:
    return tuple(row["candidate"] for row in primitive_ward_candidate_vectors(0.01, 0.02, 0.1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only primitive response closure suite; not a production fix.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--candidate", choices=_candidate_choices(), default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _fmt_complex(value: object) -> str:
    if isinstance(value, dict):
        real = float(value.get("real", 0.0))
        imag = float(value.get("imag", 0.0))
        return f"{real:+.6e}{imag:+.6e}j"
    z = complex(value)  # type: ignore[arg-type]
    return f"{z.real:+.6e}{z.imag:+.6e}j"


def _best_one_scale(sector: dict[str, object]) -> tuple[str, dict[str, object]]:
    best_name = ""
    best_fit: dict[str, object] | None = None
    for family_name, family in sector["one_scale_fits"].items():  # type: ignore[union-attr]
        if family_name == "valid_for_casimir_input":
            continue
        for kind in ["real", "complex"]:
            fit = family[kind]
            if best_fit is None or float(fit["residual_norm"]) < float(best_fit["residual_norm"]):
                best_name = f"{family_name}.{kind}"
                best_fit = fit
    assert best_fit is not None
    return best_name, best_fit


def _two_scale_fit(sector: dict[str, object]) -> dict[str, object]:
    family = next(value for key, value in sector["two_scale_fits"].items() if key != "valid_for_casimir_input")  # type: ignore[union-attr]
    return family["complex"]


def _print_sector(prefix: str, sector: dict[str, object]) -> None:
    current = sector["current_total"]
    best_name, best_fit = _best_one_scale(sector)
    two = _two_scale_fit(sector)
    grid = sector["phase_sign_grid"]
    print(
        f"{prefix}: current={float(current['norm']):.8e} "
        f"best1={best_name} coeff={_fmt_complex(best_fit['coefficient'])} res={float(best_fit['residual_norm']):.8e} "
        f"best2_coeffs={[ _fmt_complex(v) for v in two['coefficients'] ]} res2={float(two['residual_norm']):.8e} "
        f"grid=({_fmt_complex(grid['first_coefficient'])},{_fmt_complex(grid['second_coefficient'])}) grid_res={float(grid['residual_norm']):.8e}"
    )


def _print_summary(payload: dict[str, object]) -> None:
    print("primitive_response_closure_suite summary")
    print("status:", payload["status"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    analysis = payload["candidate_analysis"]
    print("candidate:", analysis["candidate"])
    for group_name in ["em_balance", "schur_balance", "collective_balance"]:
        print("\n", group_name)
        group = analysis[group_name]
        _print_sector("  left", group["left"])
        _print_sector("  right", group["right"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_primitive_response_closure_suite(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        delta0_eV=args.delta0,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        contact_scale=args.contact_scale,
        candidate_name=args.candidate,
    )
    print(f"primitive_response_closure_suite.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
