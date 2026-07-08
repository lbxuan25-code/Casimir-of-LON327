#!/usr/bin/env python3
"""Debug-only primitive response Ward decomposition CLI."""

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
from tmte.pipeline.primitive_response_ward_decomposition import (  # noqa: E402
    DEFAULT_CANDIDATES,
    run_and_write_primitive_response_ward_decomposition,
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
    parser = argparse.ArgumentParser(description="Debug-only primitive response Ward residual decomposition; not a production fix.")
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
    parser.add_argument("--candidate", dest="candidates", action="append", choices=_candidate_choices(), default=None)
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_term_block(title: str, block: dict[str, object]) -> None:
    total = block["total"]
    print(f"    {title} total_norm={float(total['norm']):.8e} total_over_ref={float(total['norm_over_reference']):.8e}")
    for term in block["terms"]:
        print(f"      {term['term']}: norm={float(term['norm']):.8e} over_ref={float(term['norm_over_reference']):.8e}")


def _print_summary(payload: dict[str, object]) -> None:
    print("primitive_response_ward_decomposition summary")
    print("status:", payload["status"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    for row in payload["candidate_decompositions"]:  # type: ignore[index]
        n = row["norm_summary"]
        print("\nCANDIDATE", row["candidate"])
        print(
            "  summary "
            f"left_em={float(n['left_em_total_norm']):.8e} "
            f"left_coll={float(n['left_collective_total_norm']):.8e} "
            f"right_em={float(n['right_em_total_norm']):.8e} "
            f"right_coll={float(n['right_collective_total_norm']):.8e} "
            f"left_schur={float(n['left_schur_total_norm']):.8e} "
            f"right_schur={float(n['right_schur_total_norm']):.8e} "
            f"left_schur/eff={float(n['left_schur_over_effective_norm']):.8e} "
            f"right_schur/eff={float(n['right_schur_over_effective_norm']):.8e}"
        )
        _print_term_block("left_em", row["left_em_decomposition"])
        _print_term_block("left_collective", row["left_collective_decomposition"])
        _print_term_block("right_em", row["right_em_decomposition"])
        _print_term_block("right_collective", row["right_collective_decomposition"])
        schur = row["schur_effective_decomposition"]
        print(f"    schur_left total_norm={float(schur['left_total']['norm']):.8e} over_eff={float(schur['left_total']['norm_over_reference']):.8e}")
        for term in schur["left_terms"]:
            print(f"      {term['term']}: norm={float(term['norm']):.8e} over_eff={float(term['norm_over_reference']):.8e}")
        print(f"    schur_right total_norm={float(schur['right_total']['norm']):.8e} over_eff={float(schur['right_total']['norm_over_reference']):.8e}")
        for term in schur["right_terms"]:
            print(f"      {term['term']}: norm={float(term['norm']):.8e} over_eff={float(term['norm_over_reference']):.8e}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_candidates:
        candidate_names = _candidate_choices()
    else:
        candidate_names = tuple(args.candidates) if args.candidates else DEFAULT_CANDIDATES
    payload = run_and_write_primitive_response_ward_decomposition(
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
        candidate_names=candidate_names,
    )
    print(f"primitive_response_ward_decomposition.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
