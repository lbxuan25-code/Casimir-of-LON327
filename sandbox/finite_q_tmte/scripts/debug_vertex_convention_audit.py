#!/usr/bin/env python3
"""Debug-only primitive vertex convention audit CLI."""

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
from tmte.pipeline.vertex_convention_audit import run_and_write_vertex_convention_audit  # noqa: E402


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only primitive vertex convention audit; not a production fix.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--kx", type=float, default=0.37)
    parser.add_argument("--ky", type=float, default=-0.41)
    parser.add_argument("--nk-for-model", type=_positive_int, default=5)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--current-vertex", default="peierls")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _report_by_name(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(row for row in rows if row["name"] == name)


def _combo_by_candidate(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(row for row in rows if row["candidate"] == name)


def _print_summary(payload: dict[str, object]) -> None:
    print("vertex_convention_audit summary")
    print("status:", payload["status"])
    print("point:", payload["point"])

    primitive = payload["primitive_vertex_reports"]
    print("\nprimitive Hermiticity ratios:")
    for side in ["source", "observable"]:
        for row in primitive[side]:
            print(
                f"{side:>10s} {row['name']:>3s} "
                f"herm/norm={float(row['hermitian_residual_over_norm']):.6e} "
                f"anti/norm={float(row['antihermitian_residual_over_norm']):.6e} "
                f"iM_herm/norm={float(row['i_times_matrix_hermitian_residual_over_norm']):.6e}"
            )

    print("\nsource-observable relations:")
    for row in primitive["source_observable_relations"]:
        print(
            f"{row['name']:>3s} "
            f"left-right†={float(row['left_minus_right_dagger_over_norm']):.6e} "
            f"left+right†={float(row['left_plus_right_dagger_over_norm']):.6e}"
        )

    coll = payload["collective_vertex_reports"]
    print("\ncollective Hermiticity ratios:")
    for row in coll["reports"]:
        print(
            f"{row['name']:>16s} "
            f"herm/norm={float(row['hermitian_residual_over_norm']):.6e} "
            f"anti/norm={float(row['antihermitian_residual_over_norm']):.6e} "
            f"iM_herm/norm={float(row['i_times_matrix_hermitian_residual_over_norm']):.6e}"
        )

    print("\nfinite-difference current checks:")
    checks = payload["finite_difference_current_checks"]
    for side in ["source", "observable"]:
        print(side)
        for row in checks[side]:
            print(f"  {row['name']}: residual/ref={float(row['residual_to_reference_over_reference']):.6e}")

    print("\nWard-like matrix combinations: residual_to_deltaH norms")
    combos = payload["ward_like_matrix_combinations"]
    for side in ["source", "observable"]:
        print(side)
        for row in combos[side]:
            m = row["combination_minus_deltaH"]
            print(f"  {row['candidate']}: {float(m['residual_to_reference_norm']):.6e}  over_ref={float(m['residual_to_reference_over_reference']):.6e}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_vertex_convention_audit(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        kx=args.kx,
        ky=args.ky,
        nk_for_model=args.nk_for_model,
        delta0_eV=args.delta0,
        eta_eV=args.eta,
        current_vertex=args.current_vertex,
    )
    print(f"vertex_convention_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
