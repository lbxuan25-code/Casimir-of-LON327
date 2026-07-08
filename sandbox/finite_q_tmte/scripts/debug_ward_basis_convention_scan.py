#!/usr/bin/env python3
"""Debug-only Ward-basis convention fingerprint scan."""

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
from tmte.pipeline.ward_basis_convention_scan import default_candidate_names, run_and_write_ward_basis_convention_scan  # noqa: E402


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("nk must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only Ward-basis convention fingerprint scan; not a production fix.")
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
    parser.add_argument("--candidate", dest="candidates", action="append", choices=default_candidate_names(), default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _candidate_by_name(payload: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in payload["basis_candidates"] if item["basis_candidate"] == name)  # type: ignore[index]


def _ward_by_name(candidate: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in candidate["extended_ward_candidates"] if item["candidate"] == name)  # type: ignore[index]


def _print_summary(payload: dict[str, object]) -> None:
    print("basis_candidate gauge_row gauge_gg g/TM GG/TM K_GTM.real K_GTM.imag analytic_total fitted_total accepted_convention")
    for candidate in payload["basis_candidates"]:  # type: ignore[index]
        metrics = candidate["effective_metrics"]
        k_gtm = metrics["K_eff_GTM"]
        if isinstance(k_gtm, dict):
            k_gtm_value = complex(k_gtm["real"], k_gtm["imag"])
        else:
            k_gtm_value = complex(k_gtm)
        analytic = _ward_by_name(candidate, "analytic_same_negative")["norms"]
        fitted = _ward_by_name(candidate, "fitted_both_independent")["norms"]
        analytic_total = max(float(analytic["left_total_extended_norm"]), float(analytic["right_total_extended_norm"]))
        fitted_total = max(float(fitted["left_total_extended_norm"]), float(fitted["right_total_extended_norm"]))
        print(
            f"{candidate['basis_candidate']} "
            f"{float(metrics['gauge_row_norm']):.6e} "
            f"{float(metrics['gauge_gg_norm']):.6e} "
            f"{float(metrics['gauge_over_tm_abs']):.6e} "
            f"{float(metrics['gauge_gg_over_tm_abs']):.6e} "
            f"{k_gtm_value.real:.6e} {k_gtm_value.imag:.6e} "
            f"{analytic_total:.6e} {fitted_total:.6e} "
            f"{candidate['status']['accepted_convention']}"
        )
    status = payload["status"]  # type: ignore[index]
    print("status:")
    print(f"diagnostic_only_not_a_fix {status['diagnostic_only_not_a_fix']}")
    print(f"accepted_convention {status['accepted_convention']}")
    print(f"requires_analytic_derivation_before_convention_change {status['requires_analytic_derivation_before_convention_change']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_ward_basis_convention_scan(
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
        candidate_names=tuple(args.candidates) if args.candidates else default_candidate_names(),
    )
    print(f"ward_basis_convention_scan.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
