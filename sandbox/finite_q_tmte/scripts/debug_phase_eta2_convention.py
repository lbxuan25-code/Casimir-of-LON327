#!/usr/bin/env python3
"""Debug-only q-along-x phase_eta2 convention-transform diagnostic."""

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
from tmte.pipeline.phase_eta2_convention import DEFAULT_TRANSFORMS, run_and_write_phase_eta2_convention  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only q-along-x finite-q TM/TE phase_eta2 convention diagnostic.")
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
    parser.add_argument("--transforms", nargs="+", default=list(DEFAULT_TRANSFORMS))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _as_complex(value: object) -> complex:
    if isinstance(value, dict):
        return complex(float(value["real"]), float(value["imag"]))
    return complex(value)  # type: ignore[arg-type]


def _print_summary(payload: dict[str, object]) -> None:
    print("transform Keff_GG.real Keff_GG.imag Keff_GTM.real Keff_GTM.imag g/TM GG/TM cond method suspect delta_g/TM")
    results = payload["transform_results"]  # type: ignore[index]
    best = None
    for result in results:
        eff = result["K_eff_entries"]
        diagnostics = result["diagnostics"]
        schur = result["schur"]
        eff_gg = _as_complex(eff["GG"])
        eff_gtm = _as_complex(eff["GTM"])
        if best is None or diagnostics["gauge_over_tm_abs"] < best["diagnostics"]["gauge_over_tm_abs"]:
            best = result
        print(
            f"{result['transform']} "
            f"{eff_gg.real:.6e} {eff_gg.imag:.6e} "
            f"{eff_gtm.real:.6e} {eff_gtm.imag:.6e} "
            f"{float(diagnostics['gauge_over_tm_abs']):.6e} "
            f"{float(diagnostics['gauge_gg_over_tm_abs']):.6e} "
            f"{float(diagnostics['etaeta_condition_number']):.6e} "
            f"{schur['solve_method']} {schur['numerically_suspect']} "
            f"{float(diagnostics['delta_gauge_over_tm_abs_vs_identity']):.6e}"
        )
    if best is not None:
        print(
            "best_debug_only_by_gauge_over_tm_abs "
            f"{best['transform']} {float(best['diagnostics']['gauge_over_tm_abs']):.6e}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_phase_eta2_convention(
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
        transforms=tuple(args.transforms),
    )
    print(f"phase_eta2_convention.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
