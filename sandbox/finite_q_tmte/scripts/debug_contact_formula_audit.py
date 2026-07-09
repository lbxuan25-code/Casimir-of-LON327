#!/usr/bin/env python3
"""Debug-only contact formula audit CLI."""

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
from tmte.pipeline.contact_formula_audit import run_and_write_contact_formula_audit  # noqa: E402
from tmte.pipeline.primitive_response_closure_suite import DEFAULT_CANDIDATE  # noqa: E402
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


def _fmt_complex(value: object) -> str:
    if isinstance(value, dict):
        real = float(value.get("real", 0.0))
        imag = float(value.get("imag", 0.0))
        return f"{real:+.8e}{imag:+.8e}j"
    z = complex(value)  # type: ignore[arg-type]
    return f"{z.real:+.8e}{z.imag:+.8e}j"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only current-vs-Ward-required contact formula audit.")
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


def _print_component_ratios(side_payload: dict[str, object]) -> None:
    print("    componentwise required/current:")
    for row in side_payload["componentwise_required_over_current"]:
        print(
            f"      {row['label']}: ratio={_fmt_complex(row['required_over_current'])} "
            f"current={_fmt_complex(row['current'])} required={_fmt_complex(row['required'])} "
            f"defined={row['ratio_defined']}"
        )


def _print_side(name: str, side_payload: dict[str, object]) -> None:
    scalar = side_payload["required_over_current_scalar_projection"]
    overlap = side_payload["parallelism"]
    residual = side_payload["ward_residual_with_current_contact"]
    diff = side_payload["current_minus_required"]
    projected = side_payload["residual_after_scalar_projection"]
    print(f"\n{name}")
    print(f"  current_norm={float(side_payload['contact_current']['norm']):.8e}")
    print(f"  required_norm={float(side_payload['contact_required']['norm']):.8e}")
    print(f"  current_minus_required_norm={float(diff['norm']):.8e}")
    print(f"  ward_residual_with_current_contact_norm={float(residual['norm']):.8e}")
    print(
        "  scalar alpha_required_over_current="
        f"{_fmt_complex(scalar['alpha_required_over_current'])} "
        f"projection_residual={float(scalar['residual_norm']):.8e} "
        f"projection_residual/required={float(scalar['residual_over_required_norm']):.8e}"
    )
    print(
        "  parallelism "
        f"abs_overlap={float(overlap['abs_overlap']):.8e} "
        f"overlap={_fmt_complex(overlap['normalized_overlap'])}"
    )
    print(f"  residual_after_scalar_projection_norm={float(projected['norm']):.8e}")
    _print_component_ratios(side_payload)


def _print_summary(payload: dict[str, object]) -> None:
    print("contact_formula_audit summary")
    print("status:", payload["status"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    analysis = payload["contact_formula_analysis"]
    print("candidate:", analysis["candidate"])
    consistency = analysis["left_right_scalar_consistency"]
    print(
        "left/right alpha consistency: "
        f"left={_fmt_complex(consistency['alpha_left'])} "
        f"right={_fmt_complex(consistency['alpha_right'])} "
        f"abs_diff={float(consistency['abs_difference']):.8e}"
    )
    _print_side("left_observable_row", analysis["left"])
    _print_side("right_source_column", analysis["right"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_contact_formula_audit(
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
    print(f"contact_formula_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
