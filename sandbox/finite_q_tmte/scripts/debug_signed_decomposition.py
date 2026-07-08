#!/usr/bin/env python3
"""Debug-only q-along-x signed complex decomposition diagnostic."""

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
from tmte.pipeline.signed_decomposition import ENTRY_SPECS, run_and_write_signed_decomposition  # noqa: E402


def _nonnegative_int(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return index


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("nk must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only q-along-x finite-q TM/TE signed decomposition.")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    def parts(value: complex) -> tuple[float, float]:
        number = complex(value)
        return float(number.real), float(number.imag)

    print("entry bubble.real bubble.imag contact.real contact.imag KSS.real KSS.imag schur.real schur.imag Keff.real Keff.imag")
    entries = payload["entries"]  # type: ignore[index]
    for entry_name, _, _ in ENTRY_SPECS:
        row = entries[entry_name]
        bubble = parts(row["K_SS_bubble"])
        contact = parts(row["K_SS_contact_scaled"])
        kss = parts(row["K_SS_scaled"])
        schur = parts(row["Schur_correction"])
        keff = parts(row["K_eff"])
        print(
            f"{entry_name} {bubble[0]:.6e} {bubble[1]:.6e} {contact[0]:.6e} {contact[1]:.6e} "
            f"{kss[0]:.6e} {kss[1]:.6e} {schur[0]:.6e} {schur[1]:.6e} {keff[0]:.6e} {keff[1]:.6e}"
        )
    ratios = payload["ratios"]  # type: ignore[index]
    schur = payload["schur"]  # type: ignore[index]
    print(f"gauge_row_norm {ratios['gauge_row_norm']}")
    print(f"gauge_col_norm {ratios['gauge_col_norm']}")
    print(f"gauge_gg_norm {ratios['gauge_gg_norm']}")
    print(f"physical_matrix_norm {ratios['physical_matrix_norm']}")
    print(f"gauge_over_physical {ratios['gauge_over_physical']}")
    print(f"gauge_over_tm_abs {ratios['gauge_over_tm_abs']}")
    print(f"gauge_gg_over_tm_abs {ratios['gauge_gg_over_tm_abs']}")
    print(f"schur_solve_method {schur['solve_method']}")
    print(f"schur_numerically_suspect {schur['numerically_suspect']}")
    print(f"etaeta_condition_number {schur['etaeta_condition_number']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_signed_decomposition(
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
    )
    print(f"signed_decomposition.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

