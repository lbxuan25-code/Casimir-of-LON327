#!/usr/bin/env python3
"""Debug-only q-along-x collective Schur factor diagnostic."""

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
from tmte.pipeline.collective_schur_factors import run_and_write_collective_schur_factors  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only q-along-x finite-q TM/TE collective Schur factors.")
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


def _format_complex(value: object) -> str:
    number = _as_complex(value)
    return f"{number.real:.6e}{number.imag:+.6e}j"


def _print_matrix(name: str, matrix: object) -> None:
    print(name)
    rows = matrix.get("data", matrix) if isinstance(matrix, dict) else matrix  # type: ignore[union-attr]
    for row in rows:  # type: ignore[union-attr]
        print("  " + " ".join(_format_complex(value) for value in row))


def _print_summary(payload: dict[str, object]) -> None:
    matrices = payload["matrices"]  # type: ignore[index]
    _print_matrix("K_Seta", matrices["K_Seta"])  # type: ignore[index]
    _print_matrix("K_etaS", matrices["K_etaS"])  # type: ignore[index]
    _print_matrix("K_etaeta", matrices["K_etaeta"])  # type: ignore[index]
    solver_action = matrices["K_etaeta_inverse_or_solver_action"]  # type: ignore[index]
    _print_matrix("X = solve(K_etaeta, K_etaS)", solver_action["X"])  # type: ignore[index]

    collective_order = payload["collective_order"]  # type: ignore[index]
    first_label = collective_order[0]  # type: ignore[index]
    second_label = collective_order[1]  # type: ignore[index]
    print(f"entry total.real total.imag {first_label}.product.real {first_label}.product.imag {second_label}.product.real {second_label}.product.imag reconstruction_error")
    decomposition = payload["schur_factor_decomposition"]  # type: ignore[index]
    for entry_name in ("GG", "GTM", "TMG", "TMTM"):
        row = decomposition[entry_name]  # type: ignore[index]
        total = _as_complex(row["total"])
        contributions = row["contributions"]
        eta0 = _as_complex(contributions[0]["product"])
        eta1 = _as_complex(contributions[1]["product"])
        print(
            f"{entry_name} {total.real:.6e} {total.imag:.6e} "
            f"{eta0.real:.6e} {eta0.imag:.6e} {eta1.real:.6e} {eta1.imag:.6e} "
            f"{float(row['reconstruction_error']):.6e}"
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
    payload = run_and_write_collective_schur_factors(
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
    print(f"collective_schur_factors.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
