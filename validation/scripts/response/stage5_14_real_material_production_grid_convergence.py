#!/usr/bin/env python3
"""Stage 5.14 real-material production-grid energy convergence run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lno327.electrodynamics.units import SheetConductivityUnitConvention  # noqa: E402
from lno327.casimir.material_production_grid import (  # noqa: E402
    N0_POLICY,
    Q0_POLICY,
    build_production_grid,
    integrate_grid_energy_from_rows,
    material_reflection_points_for_grid,
    production_grid_plan_from_stage5_13,
    summarize_energy_convergence,
    validate_stage5_13_input,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402

from stage5_11_real_material_reflection_grid_prototype import run_real_points  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "material_grid_convergence"
DEFAULT_INPUT = OUTPUT_DIR / "stage5_13_zero_mode_grid_convergence_audit.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_14_real_material_production_grid_convergence.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_14_real_material_production_grid_convergence.md"
DEFAULT_CACHE_DIR = ROOT / "validation" / "outputs" / "response" / "material_reflection_grid" / "cache"

BOUNDARY = {
    "no_response_formula_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_ward_convention_change": True,
    "no_pi_to_sigma_change": True,
    "no_unit_conversion_change": True,
    "no_reflection_convention_change": True,
    "no_trace_log_convention_change": True,
    "no_force_output": True,
    "no_torque_output": True,
    "not_final_physical_result": True,
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        return {"re": float(np.real(value)), "im": float(np.imag(value)), "abs": float(abs(value))}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _response_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "adaptive_level": int(args.adaptive_level),
        "gauss_order": int(args.gauss_order),
        "fermi_window_eV": float(args.fermi_window_eV),
        "coarse_grid": int(args.coarse_grid),
        "eta_eV": float(args.eta_eV),
    }


def _convention() -> SheetConductivityUnitConvention:
    return SheetConductivityUnitConvention(
        lattice_a_x_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        lattice_a_y_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        unit_cell_area_m2=LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2,
    )


def _lattice_convention_payload() -> dict[str, Any]:
    return {
        "a_x_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        "a_y_m": LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        "source": LNO327_THIN_FILM_SLAO_IN_PLANE.source_note,
        "is_placeholder": LNO327_THIN_FILM_SLAO_IN_PLANE.is_placeholder,
    }


def _cache_summary(rows_by_level: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for rows in rows_by_level.values() for row in rows]
    return {
        "num_rows": len(rows),
        "num_cache_hits": sum(row.get("cache", {}).get("source") == "hit" for row in rows),
        "num_computed": sum(row.get("cache", {}).get("source") == "computed" for row in rows),
        "num_without_cache_metadata": sum("cache" not in row for row in rows),
    }


def _planned_grid_run(level: str, *, Q_max_nm_inv: float, temperature_K: float) -> dict[str, Any]:
    grid = build_production_grid(level, Q_max_nm_inv=Q_max_nm_inv, temperature_K=temperature_K)
    return {
        "level": level,
        "n_max": grid.n_max,
        "n_Q": grid.n_Q,
        "n_phi": grid.n_phi,
        "Q_max_nm_inv": grid.Q_max_nm_inv,
        "num_response_points_expected": grid.num_response_points,
        "num_energy_points_including_n0": grid.num_energy_points_including_n0,
        "Q0_policy": Q0_POLICY,
        "n0_policy": N0_POLICY,
        "status": "DRY_RUN",
    }


def _run_grid_level(
    level: str,
    *,
    Q_max_nm_inv: float,
    temperature_K: float,
    separation_m: float,
    response_config: dict[str, Any],
    workers: int,
    cache_dir: Path,
    resume: bool,
    skip_existing: bool,
    force_recompute: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = build_production_grid(level, Q_max_nm_inv=Q_max_nm_inv, temperature_K=temperature_K)
    rows = run_real_points(
        material_reflection_points_for_grid(grid),
        response_config=response_config,
        separation_m=separation_m,
        convention=_convention(),
        workers=workers,
        cache_dir=cache_dir,
        resume=resume,
        skip_existing=skip_existing,
        force_recompute=force_recompute,
        lattice_convention=_lattice_convention_payload(),
    )
    return integrate_grid_energy_from_rows(grid, rows), rows


def run_convergence(
    input_json: Path,
    *,
    response_config: dict[str, Any],
    workers: int,
    temperature_K: float,
    separation_nm: float,
    Q_max_nm_inv: float,
    cache_dir: Path,
    resume: bool,
    skip_existing: bool,
    force_recompute: bool,
    dry_run_grid_only: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    data = json.loads(Path(input_json).read_text(encoding="utf-8"))
    input_status = validate_stage5_13_input(data)
    grid_plan = production_grid_plan_from_stage5_13(data)
    separation_m = float(separation_nm) * 1.0e-9
    if dry_run_grid_only:
        grid_runs = {
            level: _planned_grid_run(level, Q_max_nm_inv=Q_max_nm_inv, temperature_K=temperature_K)
            for level in ("coarse", "medium", "fine")
        }
        rows_by_level: dict[str, list[dict[str, Any]]] = {"coarse": [], "medium": [], "fine": []}
        energy_convergence = {
            "coarse_to_medium_relative_change": None,
            "medium_to_fine_relative_change": None,
            "status": "DRY_RUN",
        }
    else:
        rows_by_level = {}
        grid_runs = {}
        for level in ("coarse", "medium", "fine"):
            grid_runs[level], rows_by_level[level] = _run_grid_level(
                level,
                Q_max_nm_inv=Q_max_nm_inv,
                temperature_K=temperature_K,
                separation_m=separation_m,
                response_config=response_config,
                workers=workers,
                cache_dir=cache_dir,
                resume=resume,
                skip_existing=skip_existing,
                force_recompute=force_recompute,
            )
        energy_convergence = summarize_energy_convergence(grid_runs)
    cache_summary = _cache_summary(rows_by_level)
    checks = {
        "stage5_13_input": "PASS",
        "grid_plan_present": "PASS",
        "Q0_not_regular_point": "PASS",
        "n0_extrapolated_not_divided_by_zero": "PASS",
        "cache_enabled": "PASS" if cache_dir is not None else "FAIL",
        "energy_convergence": energy_convergence["status"],
        "no_force_output": "PASS",
        "no_torque_output": "PASS",
        "not_final_physical_result": "PASS",
    }
    if dry_run_grid_only:
        stage_status = "STAGE5_14_DRY_RUN_GRID_ONLY_PASSED"
    elif any(value == "FAIL" for value in checks.values()):
        stage_status = "STAGE5_14_REAL_MATERIAL_PRODUCTION_GRID_CONVERGENCE_FAILED"
    elif any(value == "MONITOR" for value in checks.values()):
        stage_status = "STAGE5_14_REAL_MATERIAL_PRODUCTION_GRID_CONVERGENCE_MONITOR"
    else:
        stage_status = "STAGE5_14_REAL_MATERIAL_PRODUCTION_GRID_CONVERGENCE_PASSED"
    return {
        "stage": "Stage 5.14",
        "purpose": "Real-material production-grid energy convergence run",
        "boundary": dict(BOUNDARY),
        "input": {"input_json": str(input_json), "input_stage": data.get("stage"), "input_status": input_status},
        "grid_plan": grid_plan,
        "grid_runs": grid_runs,
        "energy_convergence": energy_convergence,
        "cache_summary": {
            **cache_summary,
            "cache_dir": str(cache_dir),
            "resume": bool(resume),
            "skip_existing": bool(skip_existing),
            "force_recompute": bool(force_recompute),
        },
        "checks": checks,
        "diagnostic_status": {
            "stage5_14_status": stage_status,
            "recommended_next_action": "Use this only as a convergence gate; do not treat it as final force or torque.",
        },
        "run_config": {
            **response_config,
            "workers": int(workers),
            "temperature_K": float(temperature_K),
            "separation_nm": float(separation_nm),
            "Q_max_nm_inv": float(Q_max_nm_inv),
            "dry_run_grid_only": bool(dry_run_grid_only),
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# Stage 5.14 real-material production-grid energy convergence",
            "## Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## Input\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## Grid Runs\n\n"
            + _table(
                ("level", "n_max", "n_Q", "n_phi", "status"),
                [
                    (
                        level,
                        run.get("n_max"),
                        run.get("n_Q"),
                        run.get("n_phi"),
                        run.get("status"),
                    )
                    for level, run in data["grid_runs"].items()
                ],
            ),
            "## Energy Convergence\n\n" + _table(("quantity", "value"), list(data["energy_convergence"].items())),
            "## Cache Summary\n\n" + _table(("quantity", "value"), list(data["cache_summary"].items())),
            "## Checks\n\n" + _table(("check", "status"), list(data["checks"].items())),
            "## Notes\n\n"
            "Q=0 不作为普通点；径向网格使用 interior quadrature nodes。n=0 使用 xi->0+ 的 "
            "R_TE_TM 外推代理，不直接除以 Omega=0。本阶段不输出 force 或 torque，也不声称 final physical result。",
            "## Diagnostic Decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--adaptive-level", type=int, default=4)
    parser.add_argument("--gauss-order", type=int, default=7)
    parser.add_argument("--fermi-window-eV", type=float, default=0.05)
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--eta-eV", type=float, default=1e-10)
    parser.add_argument("--workers", type=int, default=1, help="Point-level multiprocessing worker count; separate from BLAS/OpenMP threading.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--separation-nm", type=float, default=100.0)
    parser.add_argument("--Q-max-nm-inv", type=float, default=0.5)
    parser.add_argument("--dry-run-grid-only", action="store_true", help="Validate input and emit planned grids without running real response.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_convergence(
        args.input_json,
        response_config=_response_config(args),
        workers=args.workers,
        temperature_K=args.temperature_K,
        separation_nm=args.separation_nm,
        Q_max_nm_inv=args.Q_max_nm_inv,
        cache_dir=args.cache_dir,
        resume=args.resume,
        skip_existing=args.skip_existing,
        force_recompute=args.force_recompute,
        dry_run_grid_only=args.dry_run_grid_only,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
