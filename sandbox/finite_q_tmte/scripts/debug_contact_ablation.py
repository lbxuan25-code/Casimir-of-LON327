#!/usr/bin/env python3
"""Debug-only q-along-x spatial-contact ablation diagnostic."""

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
from tmte.pipeline.contact_ablation import run_and_write_contact_ablation  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only q-along-x finite-q TM/TE contact ablation.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--contact-scales", nargs="+", type=float, default=[1.0, 0.0, -1.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    def real_part(value: object) -> float:
        return float(complex(value).real)

    print("scale bare_G_row bare_GG eff_G_row eff_GG K_TMTM K_TETE g/phys g/TM GG/TM cond method suspect")
    for row in payload.get("contact_scale_results", []):  # type: ignore[union-attr]
        bare = row["bare_diagnostics"]["K_SS_scaled"]  # type: ignore[index]
        eff = row["effective_diagnostics"]  # type: ignore[index]
        elements = row["selected_matrix_elements"]  # type: ignore[index]
        ratios = row["ratios"]  # type: ignore[index]
        schur = row["schur"]  # type: ignore[index]
        print(
            "{scale:g} {bare_g:.6e} {bare_gg:.6e} {eff_g:.6e} {eff_gg:.6e} {ktmtm:.6e} {ktete:.6e} "
            "{gphys:.6e} {gtm:.6e} {ggtm:.6e} {cond:.6e} {method} {suspect}".format(
                scale=row["contact_scale"],  # type: ignore[index]
                bare_g=bare["G_row_norm"],
                bare_gg=bare["GG_abs"],
                eff_g=eff["G_row_norm"],
                eff_gg=eff["GG_abs"],
                ktmtm=real_part(elements["K_TMTM"]),
                ktete=real_part(elements["K_TETE"]),
                gphys=ratios["gauge_over_physical"],
                gtm=ratios["gauge_over_tm_abs"],
                ggtm=ratios["gauge_gg_over_tm_abs"],
                cond=schur["etaeta_condition_number"],
                method=schur["solve_method"],
                suspect=schur["numerically_suspect"],
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_contact_ablation(
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
        contact_scales=tuple(args.contact_scales),
    )
    print(f"contact_ablation.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
