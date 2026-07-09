#!/usr/bin/env python3
"""Debug-only normal response convention audit CLI."""

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
from tmte.pipeline.normal_response_convention_audit import (  # noqa: E402
    DEFAULT_CANDIDATE_NAMES,
    run_and_write_normal_response_convention_audit,
)


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


def _fmt_complex(value: object) -> str:
    if isinstance(value, dict):
        real = float(value.get("real", 0.0))
        imag = float(value.get("imag", 0.0))
        return f"{real:+.8e}{imag:+.8e}j"
    z = complex(value)  # type: ignore[arg-type]
    return f"{z.real:+.8e}{z.imag:+.8e}j"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only normal response convention audit.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--candidate", dest="candidates", action="append", choices=DEFAULT_CANDIDATE_NAMES)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_ranked(payload: dict[str, object]) -> None:
    print("ranked candidates")
    print("rank name total/ref alpha_real left_right_diff")
    for row in payload["ranked_candidates"][:20]:
        print(
            f"{int(row['rank']):02d} "
            f"{row['name']} "
            f"{float(row['total_max_residual_over_reference']):.8e} "
            f"{float(row['alpha_real_mean']):.8e} "
            f"{float(row['left_right_alpha_abs_diff']):.8e}"
        )


def _print_candidate_details(payload: dict[str, object]) -> None:
    by_name = {row["candidate"]["name"]: row for row in payload["candidates"]}
    for ranked in payload["ranked_candidates"][:5]:
        name = ranked["name"]
        row = by_name[name]
        summary = row["summary"]
        candidate = row["candidate"]
        print(f"\n{name}")
        print("  candidate:", candidate)
        print(
            "  total: "
            f"left={float(summary['total_left_norm']):.8e} "
            f"right={float(summary['total_right_norm']):.8e} "
            f"max/ref={float(summary['total_max_residual_over_reference']):.8e}"
        )
        print(
            "  alpha: "
            f"left={_fmt_complex(summary['alpha_left'])} "
            f"right={_fmt_complex(summary['alpha_right'])} "
            f"projL={float(summary['left_projection_residual_over_required']):.8e} "
            f"projR={float(summary['right_projection_residual_over_required']):.8e}"
        )
        vertex = row["vertex_identity"]
        print(
            "  vertex: "
            f"max_abs={float(vertex['max_abs_error_over_shifted_meshes']):.8e} "
            f"max_rel={float(vertex['max_rel_error_over_shifted_meshes']):.8e}"
        )


def _print_summary(payload: dict[str, object]) -> None:
    print("normal_response_convention_audit summary")
    print("status:", payload["status"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    _print_ranked(payload)
    _print_candidate_details(payload)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_normal_response_convention_audit(
        args.output_dir,
        model_name=args.model,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        candidate_names=tuple(args.candidates) if args.candidates else None,
        full_grid=bool(args.full_grid),
    )
    print(f"normal_response_convention_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
