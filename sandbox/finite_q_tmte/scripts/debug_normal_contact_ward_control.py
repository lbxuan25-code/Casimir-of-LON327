#!/usr/bin/env python3
"""Debug-only normal-state Peierls contact Ward control CLI."""

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
from tmte.pipeline.normal_contact_ward_control import run_and_write_normal_contact_ward_control  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only normal-state Peierls contact Ward control.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _response_by_name(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["name"]): row for row in payload["responses"]}


def _print_summary(payload: dict[str, object]) -> None:
    print("normal_contact_ward_control summary")
    print("status:", payload["status"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    vertex = payload["vertex_identity"]
    print(
        "vertex_identity: "
        f"max_abs={float(vertex['max_abs_error_over_shifted_meshes']):.8e} "
        f"max_rel={float(vertex['max_rel_error_over_shifted_meshes']):.8e} "
        f"mean_rel={float(vertex['mean_rel_error_over_shifted_meshes']):.8e}"
    )
    rows = _response_by_name(payload)
    print("response residuals")
    for name in ["bubble", "contact", "total"]:
        row = rows[name]
        print(
            f"  {name}: "
            f"left={float(row['left_residual']['norm']):.8e} "
            f"right={float(row['right_residual']['norm']):.8e} "
            f"max/ref={float(row['max_residual_over_reference']):.8e}"
        )
    contact = payload["contact_formula"]
    left = contact["left_required_over_current_scalar_projection"]
    right = contact["right_required_over_current_scalar_projection"]
    print(
        "contact alpha required/current: "
        f"left={_fmt_complex(left['alpha_required_over_current'])} "
        f"left_proj_res/required={float(left['residual_over_required_norm']):.8e} "
        f"right={_fmt_complex(right['alpha_required_over_current'])} "
        f"right_proj_res/required={float(right['residual_over_required_norm']):.8e} "
        f"left_right_diff={float(contact['left_right_alpha_abs_diff']):.8e}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_normal_contact_ward_control(
        args.output_dir,
        model_name=args.model,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
    )
    print(f"normal_contact_ward_control.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
