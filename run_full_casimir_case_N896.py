from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lno327.casimir.cli import execute_case


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run one canonical full Casimir case with an extended "
            "transverse-N ladder."
        )
    )

    result.add_argument("--case", required=True)
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/casimir/runs"),
    )
    result.add_argument("--resume", action="store_true")

    result.add_argument(
        "--pairing",
        required=True,
        choices=("spm", "dwave"),
    )
    result.add_argument("--temperature-K", type=float, default=10.0)
    result.add_argument("--separation-nm", type=float, default=20.0)
    result.add_argument("--angle-deg", type=float, required=True)

    result.add_argument("--workers", type=int, default=30)
    result.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "q", "context", "wave"),
        default="auto",
    )
    result.add_argument(
        "--memory-budget-gb",
        type=float,
        default=0.0,
    )
    result.add_argument(
        "--max-context-workers",
        type=int,
        default=1,
    )

    result.add_argument(
        "--N-candidates",
        nargs="+",
        type=int,
        default=(128, 192, 256, 384, 512, 640, 768, 896),
    )
    result.add_argument(
        "--required-consecutive-passes",
        type=int,
        default=2,
    )

    result.add_argument(
        "--matsubara-cutoffs",
        nargs="+",
        type=int,
        default=(1, 3, 7, 15, 31),
    )
    result.add_argument(
        "--outer-cutoffs-u",
        nargs="+",
        type=float,
        default=(6, 10, 14, 18, 24, 30, 36, 42),
    )

    result.add_argument("--rtol", type=float, default=5e-3)
    result.add_argument("--atol-J-m2", type=float, default=1e-12)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)

    result = execute_case(
        case=args.case,
        output_root=args.output_root,
        resume=args.resume,
        pairings=(args.pairing,),
        temperature_K=args.temperature_K,
        separation_nm=args.separation_nm,
        plate_angles_deg=(0.0, args.angle_deg),
        N_candidates=tuple(args.N_candidates),
        required_consecutive_passes=(
            args.required_consecutive_passes
        ),
        workers=args.workers,
        parallel_mode=args.parallel_mode,
        memory_budget_gb=args.memory_budget_gb,
        max_context_workers=args.max_context_workers,
        matsubara_cutoff_values=tuple(
            args.matsubara_cutoffs
        ),
        cutoff_u_values=tuple(args.outer_cutoffs_u),
        total_free_energy_rtol=args.rtol,
        total_free_energy_atol_J_m2=args.atol_J_m2,
    )

    compact = {
        "case": args.case,
        "pairing": args.pairing,
        "angle_deg": args.angle_deg,
        "status": result.status,
        "termination_reason": result.termination_reason,
        "matsubara_converged": result.matsubara_converged,
        "selected_matsubara_cutoff":
            result.selected_matsubara_cutoff,
        "production_casimir_allowed":
            result.production_casimir_allowed,
        "N_candidates": list(args.N_candidates),
        "workers": args.workers,
        "parallel_mode": args.parallel_mode,
        "memory_budget_gb": args.memory_budget_gb,
        "max_context_workers":
            args.max_context_workers,
        "provider_statistics":
            dict(result.provider_statistics),
    }

    print(json.dumps(compact, sort_keys=True, indent=2))

    return 0 if result.matsubara_converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
