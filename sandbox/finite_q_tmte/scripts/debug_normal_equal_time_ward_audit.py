#!/usr/bin/env python3
"""Debug-only normal equal-time Ward audit CLI."""

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
from tmte.pipeline.normal_equal_time_ward_audit import run_and_write_normal_equal_time_ward_audit  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only normal equal-time Ward audit.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    print("normal_equal_time_ward_audit summary")
    print("status:", payload["status"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    vertex = payload["vertex_identity"]
    print(
        "vertex_identity: "
        f"max_abs={float(vertex['max_abs_error_over_shifted_meshes']):.8e} "
        f"max_rel={float(vertex['max_rel_error_over_shifted_meshes']):.8e}"
    )
    ward = payload["ward_decomposition"]
    for side in ["left", "right"]:
        row = ward[side]
        fit = row["contact_required_over_current"]
        print(
            f"{side}: "
            f"bubble={float(row['bubble']['norm']):.8e} "
            f"contact={float(row['contact']['norm']):.8e} "
            f"total={float(row['total']['norm']):.8e} "
            f"alpha={_fmt_complex(fit['alpha'])} "
            f"alpha_res/req={float(fit['residual_over_target_norm']):.8e}"
        )
    print("ranked equal-time candidates against left missing_to_close")
    print("rank name diff/missing fit_alpha fit_res/missing")
    for idx, row in enumerate(payload["candidate_equal_time_vectors_ranked"][:12], start=1):
        fit = row["fit_to_target"]
        print(
            f"{idx:02d} {row['name']} "
            f"{float(row['difference_over_target_norm']):.8e} "
            f"{_fmt_complex(fit['alpha'])} "
            f"{float(fit['residual_over_target_norm']):.8e}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_normal_equal_time_ward_audit(
        args.output_dir,
        model_name=args.model,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
    )
    print(f"normal_equal_time_ward_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
