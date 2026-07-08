#!/usr/bin/env python3
"""Debug-only q-along-x eta-channel Schur ablation diagnostic."""

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
from tmte.pipeline.eta_channel_ablation import run_and_write_eta_channel_ablation  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only q-along-x finite-q TM/TE eta-channel Schur ablation.")
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


def _as_complex(value: object) -> complex:
    if isinstance(value, dict):
        return complex(float(value["real"]), float(value["imag"]))
    return complex(value)  # type: ignore[arg-type]


def _cond_text(value: object) -> str:
    return "none" if value is None else f"{float(value):.6e}"


def _print_summary(payload: dict[str, object]) -> None:
    print(
        "mode Schur_GG.real Schur_GG.imag Schur_GTM.real Schur_GTM.imag "
        "Keff_GG.real Keff_GG.imag Keff_GTM.real Keff_GTM.imag g/TM GG/TM cond method suspect"
    )
    for result in payload["mode_results"]:  # type: ignore[index]
        schur_entries = result["Schur_correction_entries"]
        eff_entries = result["K_eff_entries"]
        diagnostics = result["diagnostics"]
        schur_gg = _as_complex(schur_entries["GG"])
        schur_gtm = _as_complex(schur_entries["GTM"])
        eff_gg = _as_complex(eff_entries["GG"])
        eff_gtm = _as_complex(eff_entries["GTM"])
        print(
            f"{result['mode']} "
            f"{schur_gg.real:.6e} {schur_gg.imag:.6e} "
            f"{schur_gtm.real:.6e} {schur_gtm.imag:.6e} "
            f"{eff_gg.real:.6e} {eff_gg.imag:.6e} "
            f"{eff_gtm.real:.6e} {eff_gtm.imag:.6e} "
            f"{float(diagnostics['gauge_over_tm_abs']):.6e} "
            f"{float(diagnostics['gauge_gg_over_tm_abs']):.6e} "
            f"{_cond_text(diagnostics['etaeta_condition_number'])} "
            f"{diagnostics['schur_solve_method']} {diagnostics['schur_numerically_suspect']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_eta_channel_ablation(
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
    print(f"eta_channel_ablation.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
