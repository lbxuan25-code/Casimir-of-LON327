#!/usr/bin/env python3
"""Run finite-grid material Casimir energy/torque candidate data generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327.material_casimir_figures import (  # noqa: E402
    DEFAULT_DISTANCE_NM,
    DEFAULT_PAIRINGS,
    DEFAULT_THETA_DEG,
    DEFAULT_ZERO_MODE_OMEGA_EV,
    MaterialCasimirConfig,
    assemble_energy_data,
    atomic_write_json,
    interior_q_nodes_nm_inv,
    run_point_grid,
    save_material_casimir_outputs,
    uniform_phi_nodes_deg,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "material_casimir"


def _parse_float_tuple(values: list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "cache")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairings", nargs="+", choices=list(DEFAULT_PAIRINGS), default=list(DEFAULT_PAIRINGS))
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--N-Q", type=int, default=16)
    parser.add_argument("--N-phi", type=int, default=16)
    parser.add_argument("--Q-max-nm-inv", type=float, default=0.25)
    parser.add_argument("--theta-deg", nargs="+", type=float, default=list(DEFAULT_THETA_DEG))
    parser.add_argument("--distance-nm", nargs="+", type=float, default=list(DEFAULT_DISTANCE_NM))
    parser.add_argument("--zero-mode-omega-eV", nargs="+", type=float, default=list(DEFAULT_ZERO_MODE_OMEGA_EV))
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--eta-eV", type=float, default=1e-4)
    parser.add_argument("--bdg-nk", type=int, default=32)
    parser.add_argument("--delta0-eV", type=float, default=0.04)
    parser.add_argument("--dry-run-grid-only", action="store_true", help="Write a plan report without computing response.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MaterialCasimirConfig(
        n_max=args.n_max,
        N_Q=args.N_Q,
        N_phi=args.N_phi,
        Q_max_nm_inv=args.Q_max_nm_inv,
        theta_deg=_parse_float_tuple(args.theta_deg),
        distance_nm=_parse_float_tuple(args.distance_nm),
        zero_mode_omega_eV=_parse_float_tuple(args.zero_mode_omega_eV),
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        bdg_nk=args.bdg_nk,
        delta0_eV=args.delta0_eV,
    )
    for subdir in ("cache", "data", "figures"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    if args.dry_run_grid_only:
        q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
        phi_nodes = uniform_phi_nodes_deg(config.N_phi)
        plan = {
            "status": "DRY_RUN_GRID_ONLY",
            "pairings": list(args.pairings),
            "config": config.__dict__,
            "num_points": len(args.pairings) * (config.n_max + 1) * config.N_Q * config.N_phi,
            "Q_nm_inv_preview": q_nodes[: min(5, len(q_nodes))].tolist(),
            "phi_deg_preview": phi_nodes[: min(5, len(phi_nodes))].tolist(),
            "report_label": "finite-grid publication-style candidate result; not full convergence audit",
        }
        report_path = args.output_dir / "data" / "material_casimir_dry_run_plan.json"
        atomic_write_json(report_path, plan)
        print(f"Wrote {report_path}")
        return

    point_rows = run_point_grid(
        list(args.pairings),
        config,
        cache_dir=args.cache_dir,
        workers=args.workers,
        resume=args.resume,
        skip_existing=args.skip_existing,
        force_recompute=args.force_recompute,
    )
    energy_data = assemble_energy_data(list(args.pairings), config, point_rows)
    paths = save_material_casimir_outputs(args.output_dir, config, point_rows, energy_data)
    for label, path in paths.items():
        print(f"{label} = {path}")
    print("report = finite-grid publication-style candidate result; not full convergence audit")


if __name__ == "__main__":
    main()
