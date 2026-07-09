#!/usr/bin/env python3
"""Debug-only RHS-aware finite-q convergence scan CLI."""

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
from tmte.pipeline.rhs_aware_convergence_scan import (  # noqa: E402
    SHIFT_MODE_FRACTIONS,
    run_and_write_rhs_aware_convergence_scan,
)
from tmte.pipeline.rhs_aware_finite_q_validation import (  # noqa: E402
    DEFAULT_CONDITION_MAX,
    DEFAULT_RESIDUAL_TOL,
)
from tmte.pipeline.schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE  # noqa: E402


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only RHS-aware finite-q convergence scan.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairings", nargs="+", default=["spm", "dwave"])
    parser.add_argument("--matsubara-indices", "--n-values", dest="matsubara_indices", nargs="+", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q-values", nargs="+", type=_positive_float, required=True)
    parser.add_argument("--nk-values", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--shift-modes", nargs="+", choices=sorted(SHIFT_MODE_FRACTIONS), default=["noshift"])
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--residual-tol", type=float, default=DEFAULT_RESIDUAL_TOL)
    parser.add_argument("--condition-max", type=float, default=DEFAULT_CONDITION_MAX)
    parser.add_argument("--include-validation-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_rows(rows: list[dict[str, object]]) -> None:
    print("rows")
    print("closed pairing n q nk shift maxS maxEff etaProj legacyZero/Keff cond Keff R_eff/Keff")
    for row in rows:
        r_eff_over_keff = max(float(row["left_R_eff_over_K_eff_norm"]), float(row["right_R_eff_over_K_eff_norm"]))
        print(
            f"{'PASS' if row['rhs_aware_ward_closed'] else 'FAIL'} "
            f"pairing={row['pairing']} n={row['matsubara_index']} q={float(row['q_value']):.6g} "
            f"nk={row['nk']} shift={row['shift_mode']} "
            f"maxS={float(row['max_s_channel_residual_over_rhs_s']):.3e} "
            f"maxEff={float(row['max_effective_residual_over_reference']):.3e} "
            f"etaProj={float(row['max_eta_projection_over_rhs_s']):.3e} "
            f"legacyZero/Keff={float(row['max_legacy_zero_rhs_residual_over_k_eff_norm']):.3e} "
            f"cond={float(row['K_etaeta_condition_number']):.3e} "
            f"Keff={float(row['K_eff_norm']):.3e} "
            f"R_eff/Keff={r_eff_over_keff:.3e}"
        )


def _print_convergence(label: str, rows: list[dict[str, object]], limit: int = 20) -> None:
    print(label)
    if not rows:
        print("  <none>")
        return
    for row in rows[:limit]:
        if "nk_from" in row:
            prefix = f"pairing={row['pairing']} n={row['matsubara_index']} q={float(row['q_value']):.6g} shift={row['shift_mode']} nk={row['nk_from']}->{row['nk_to']}"
        else:
            prefix = f"pairing={row['pairing']} n={row['matsubara_index']} q={float(row['q_value']):.6g} nk={row['nk']} shift={row['shift_from']}->{row['shift_to']}"
        print(
            f"  {prefix} "
            f"dKeff={float(row['relative_change_K_eff_norm']):.3e} "
            f"dReff={max(float(row['relative_change_left_R_eff_norm']), float(row['relative_change_right_R_eff_norm'])):.3e} "
            f"dEtaProj={float(row['relative_change_eta_projection_over_rhs_s']):.3e} "
            f"dCond={float(row['relative_change_condition_number']):.3e}"
        )
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def _print_summary(payload: dict[str, object]) -> None:
    print("rhs_aware_convergence_scan summary")
    print("status:", payload["status"])
    print("scan_parameters:", payload["scan_parameters"])
    print("aggregate:", payload["aggregate"])
    _print_rows(payload["rows"])
    _print_convergence("nk_convergence", payload["nk_convergence"])
    _print_convergence("shift_convergence", payload["shift_convergence"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_rhs_aware_convergence_scan(
        args.output_dir,
        model_name=args.model,
        pairings=tuple(args.pairings),
        matsubara_indices=tuple(args.matsubara_indices),
        temperature_K=args.temperature_K,
        q_values=tuple(args.q_values),
        nk_values=tuple(args.nk_values),
        shift_modes=tuple(args.shift_modes),
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        contact_scale=args.contact_scale,
        candidate_name=args.candidate,
        residual_tol=args.residual_tol,
        condition_max=args.condition_max,
        include_validation_payloads=args.include_validation_payloads,
    )
    print(f"rhs_aware_convergence_scan.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
