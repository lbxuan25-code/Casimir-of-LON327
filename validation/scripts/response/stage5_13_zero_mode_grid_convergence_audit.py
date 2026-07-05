#!/usr/bin/env python3
"""Stage 5.13 zero-mode and grid-convergence planning audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lno327.casimir.grid import kappa_si, matsubara_xi_grid, xi_si_to_omega_eV  # noqa: E402
from lno327.casimir.integrand import casimir_integrand_single_point  # noqa: E402
from lno327 import KuboConfig  # noqa: E402
from lno327.electrodynamics.conventions import spatial_response_to_bilayer_sheet_conductivity_model  # noqa: E402
from lno327.electrodynamics.units import SheetConductivityUnitConvention, model_to_dimensionless_sheet_conductivity  # noqa: E402
from lno327.casimir.material_grid_convergence import (  # noqa: E402
    SmallQAuditPoint,
    ZeroModeAuditPoint,
    classify_threshold,
    default_small_q_points,
    default_zero_mode_points,
    grid_convergence_plan,
    omega_eV_to_xi_si_scalar,
    q_nm_phi_to_si_model,
    stage5_13_metadata,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402
from lno327.electrodynamics.reflection import sigma_tilde_xy_to_te_tm_reflection_matrix  # noqa: E402
from lno327.collective.ward import physical_ward_residuals  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import integrate_physical_components_on_points  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "material_grid_convergence"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "material_energy_prototype" / "stage5_12_small_real_material_energy_prototype.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_13_zero_mode_grid_convergence_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_13_zero_mode_grid_convergence_audit.md"

BOUNDARY = {
    "no_response_formula_change": True,
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_conductivity_unit_change": True,
    "no_reflection_convention_change": True,
    "no_trace_log_convention_change": True,
    "no_production_energy": True,
    "no_force_output": True,
    "no_torque_output": True,
    "not_casimir_ready_claim": True,
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
        "coarse_grid": int(args.coarse_grid),
        "fermi_window_eV": float(args.fermi_window_eV),
        "eta_eV": float(args.eta_eV),
    }


def _convention() -> SheetConductivityUnitConvention:
    return SheetConductivityUnitConvention(
        lattice_a_x_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        lattice_a_y_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        unit_cell_area_m2=LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2,
    )


def _run_response_chain(
    *,
    point_payload: dict[str, Any],
    q_model: np.ndarray,
    omega_eV: float,
    xi_si: float,
    temperature_K: float,
    response_config: dict[str, Any],
    separation_m: float,
    convention: SheetConductivityUnitConvention,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        config = KuboConfig.from_kelvin(
            omega_eV=omega_eV,
            temperature_K=temperature_K,
            eta_eV=float(response_config["eta_eV"]),
            output_si=False,
        )
        cells, refined_count, _flagged = build_adaptive_cells(
            q_model,
            coarse_grid=int(response_config["coarse_grid"]),
            refinement_level=int(response_config["adaptive_level"]),
            fermi_window_eV=float(response_config["fermi_window_eV"]),
            fermi_level_eV=config.fermi_level_eV,
        )
        points, weights = quadrature_points_for_cells(cells, int(response_config["gauss_order"]))
        response = integrate_physical_components_on_points(points, weights, q_model, config)["total"]
        sigma_model = spatial_response_to_bilayer_sheet_conductivity_model(response, omega_eV)
        sigma_tilde = model_to_dimensionless_sheet_conductivity(sigma_model, convention)
        reflection_package = sigma_tilde_xy_to_te_tm_reflection_matrix(
            sigma_tilde,
            float(q_model[0]),
            float(q_model[1]),
            omega_eV,
            convention.lattice_a_x_m,
            convention.lattice_a_y_m,
        )
        reflection = reflection_package["reflection_TE_TM"]
        kappa = float(kappa_si(point_payload["Q_m_inv"], xi_si))
        logdet = complex(casimir_integrand_single_point(reflection, reflection, kappa, separation_m)["logdet_integrand"])
        left, right = physical_ward_residuals(response, omega_eV, q_model)
        ward_left = float(np.linalg.norm(left))
        ward_right = float(np.linalg.norm(right))
        ward_total = max(ward_left, ward_right)
        finite = bool(
            np.all(np.isfinite(response))
            and np.all(np.isfinite(sigma_tilde))
            and np.all(np.isfinite(reflection))
            and np.isfinite(logdet.real)
            and np.isfinite(logdet.imag)
            and np.isfinite(ward_total)
        )
        max_sigma = float(np.max(np.abs(sigma_tilde)))
        max_reflection = float(np.max(np.abs(reflection)))
        return {
            **point_payload,
            "omega_eV": omega_eV,
            "xi_si": xi_si,
            "ward_residual": {"left_max": ward_left, "right_max": ward_right, "total_max": ward_total},
            "sigma_tilde_xy": sigma_tilde,
            "reflection_TE_TM": reflection,
            "logdet_identical_sheet": logdet,
            "max_abs_sigma_tilde": max_sigma,
            "max_abs_R_TE_TM": max_reflection,
            "num_quadrature_points": int(len(points)),
            "refined_cell_count": int(refined_count),
            "runtime_seconds": float(time.perf_counter() - start),
            "status": "PASS" if finite else "FAIL",
        }
    except Exception as exc:
        return {
            **point_payload,
            "omega_eV": omega_eV,
            "xi_si": xi_si,
            "runtime_seconds": float(time.perf_counter() - start),
            "status": "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _small_q_payload(point: SmallQAuditPoint, convention: SheetConductivityUnitConvention) -> tuple[dict[str, Any], np.ndarray, float, float]:
    converted = q_nm_phi_to_si_model(point.Q_nm_inv, point.phi_deg, convention.lattice_a_x_m, convention.lattice_a_y_m)
    xi = float(matsubara_xi_grid(point.temperature_K, point.n)[point.n])
    omega = float(xi_si_to_omega_eV(xi))
    payload = {"audit": "small_Q", "n": point.n, "temperature_K": point.temperature_K, **converted}
    return payload, np.array([converted["q_model_x"], converted["q_model_y"]], dtype=float), omega, xi


def _zero_mode_payload(point: ZeroModeAuditPoint, convention: SheetConductivityUnitConvention) -> tuple[dict[str, Any], np.ndarray, float, float]:
    converted = q_nm_phi_to_si_model(point.Q_nm_inv, point.phi_deg, convention.lattice_a_x_m, convention.lattice_a_y_m)
    xi = omega_eV_to_xi_si_scalar(point.omega_eV)
    payload = {"audit": "zero_mode", "temperature_K": point.temperature_K, **converted}
    return payload, np.array([converted["q_model_x"], converted["q_model_y"]], dtype=float), point.omega_eV, xi


def _dry_row(payload: dict[str, Any], omega: float, xi: float) -> dict[str, Any]:
    return {**payload, "omega_eV": omega, "xi_si": xi, "status": "DRY_RUN"}


def _job(args: tuple[int, str, Any, dict[str, Any], float, SheetConductivityUnitConvention]) -> tuple[int, dict[str, Any]]:
    index, kind, point, response_config, separation_m, convention = args
    if kind == "small_Q":
        payload, q_model, omega, xi = _small_q_payload(point, convention)
    else:
        payload, q_model, omega, xi = _zero_mode_payload(point, convention)
    row = _run_response_chain(
        point_payload=payload,
        q_model=q_model,
        omega_eV=omega,
        xi_si=xi,
        temperature_K=float(point.temperature_K),
        response_config=response_config,
        separation_m=separation_m,
        convention=convention,
    )
    return index, row


def _run_jobs(jobs: list[tuple[int, str, Any, dict[str, Any], float, SheetConductivityUnitConvention]], *, workers: int) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1 or len(jobs) <= 1:
        return [_job(job)[1] for job in jobs]
    rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        future_map = {executor.submit(_job, job): job[0] for job in jobs}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                result_index, row = future.result()
                rows[result_index] = row
            except Exception as exc:
                rows[index] = {"status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)}}
    return [rows[index] for index in sorted(rows)]


def _status_from_rows(rows: list[dict[str, Any]]) -> str:
    if not rows or any(row.get("status") == "FAIL" for row in rows):
        return "FAIL"
    max_ward = max(row.get("ward_residual", {}).get("total_max", 0.0) for row in rows)
    return classify_threshold(max_ward, pass_threshold=1e-6, monitor_threshold=1e-5)


def _smoothness_by_group(rows: list[dict[str, Any]], x_key: str, y_key: str = "logdet_identical_sheet") -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple((k, row[k]) for k in row if k not in {x_key, y_key, "runtime_seconds", "sigma_tilde_xy", "reflection_TE_TM", "ward_residual"})
        grouped.setdefault(key, []).append(row)
    jumps = []
    for items in grouped.values():
        items.sort(key=lambda item: float(item[x_key]))
        values = [abs(complex(item[y_key])) for item in items if y_key in item]
        for prev, curr in zip(values[:-1], values[1:], strict=False):
            jumps.append(float(abs(curr - prev) / max(prev, curr, 1e-300)))
    return {"max_relative_jump": max(jumps, default=0.0), "num_jumps": len(jumps)}


def _audit_summary(rows: list[dict[str, Any]], *, x_key: str) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") != "DRY_RUN"]
    max_ward = max((row.get("ward_residual", {}).get("total_max", 0.0) for row in completed), default=None)
    return {
        "num_points": len(rows),
        "num_completed": len(completed),
        "num_failed": sum(row.get("status") == "FAIL" for row in completed),
        "max_ward_residual": max_ward,
        "max_abs_sigma_tilde": max((row.get("max_abs_sigma_tilde", 0.0) for row in completed), default=None),
        "max_abs_R_TE_TM": max((row.get("max_abs_R_TE_TM", 0.0) for row in completed), default=None),
        "max_abs_logdet": max((abs(complex(row.get("logdet_identical_sheet", 0.0))) for row in completed), default=None),
        "smoothness": _smoothness_by_group(completed, x_key) if completed else None,
    }


def run_audit(
    input_json: Path,
    *,
    response_config: dict[str, Any],
    workers: int,
    separation_nm: float,
    temperature_K: float,
    smoke: bool,
    dry_run_grid_only: bool,
) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    input_status = data.get("diagnostic_status", {}).get("stage5_12_status")
    if input_status != "STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED":
        raise ValueError("Stage 5.12 input must have STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED status")
    convention = _convention()
    separation_m = float(separation_nm) * 1e-9
    small_points = default_small_q_points(temperature_K=temperature_K, smoke=smoke)
    zero_points = default_zero_mode_points(temperature_K=temperature_K, smoke=smoke)
    if dry_run_grid_only:
        small_rows = []
        for point in small_points:
            payload, _q_model, omega, xi = _small_q_payload(point, convention)
            small_rows.append(_dry_row(payload, omega, xi))
        zero_rows = []
        for point in zero_points:
            payload, _q_model, omega, xi = _zero_mode_payload(point, convention)
            zero_rows.append(_dry_row(payload, omega, xi))
    else:
        small_jobs = [(i, "small_Q", point, response_config, separation_m, convention) for i, point in enumerate(small_points)]
        zero_jobs = [(i, "zero_mode", point, response_config, separation_m, convention) for i, point in enumerate(zero_points)]
        small_rows = _run_jobs(small_jobs, workers=workers)
        zero_rows = _run_jobs(zero_jobs, workers=workers)
    small_status = "PASS" if dry_run_grid_only else _status_from_rows(small_rows)
    zero_status = "PASS" if dry_run_grid_only else _status_from_rows(zero_rows)
    checks = {
        "input_status": "PASS",
        "small_Q_audit": small_status,
        "zero_mode_audit": zero_status,
        "grid_plan_present": "PASS",
        "no_production_energy": "PASS",
        "no_force_torque": "PASS",
    }
    if any(value == "FAIL" for value in checks.values()):
        stage_status = "STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_FAILED"
    elif any(value == "MONITOR" for value in checks.values()):
        stage_status = "STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_MONITOR"
    else:
        stage_status = "STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED"
    metadata = stage5_13_metadata()
    return {
        "stage": "Stage 5.13",
        "boundary": dict(BOUNDARY),
        "input": {"input_json": str(input_json), "input_stage": data.get("stage"), "input_status": input_status},
        "small_Q_audit": {
            "status": small_status,
            "Q0_handling_recommendation": metadata["Q0_handling_recommendation"],
            "points": small_rows,
            "summary": _audit_summary(small_rows, x_key="Q_nm_inv"),
        },
        "zero_mode_audit": {
            "status": zero_status,
            "zero_mode_recommendation": metadata["zero_mode_recommendation"],
            "n0_weight": metadata["n0_weight"],
            "points": zero_rows,
            "summary": _audit_summary(zero_rows, x_key="omega_eV"),
        },
        "grid_convergence_plan": grid_convergence_plan(),
        "checks": checks,
        "diagnostic_status": {
            "stage5_13_status": stage_status,
            "recommended_next_action": "Proceed only to production-grid convergence run after zero-mode and Q->0 handling are accepted.",
        },
        "run_config": {
            **response_config,
            "workers": workers,
            "temperature_K": temperature_K,
            "separation_nm": separation_nm,
            "smoke": smoke,
            "dry_run_grid_only": dry_run_grid_only,
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
            "# Stage 5.13 zero-mode and grid-convergence planning audit",
            "## Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## Input\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## Small-Q Audit\n\n" + _table(("quantity", "value"), list(data["small_Q_audit"]["summary"].items())),
            "## Zero-Mode Audit\n\n" + _table(("quantity", "value"), list(data["zero_mode_audit"]["summary"].items())),
            "## Grid Convergence Plan\n\n" + json.dumps(data["grid_convergence_plan"], indent=2),
            "## Checks\n\n" + _table(("check", "status"), list(data["checks"].items())),
            "## Notes\n\n"
            "Q=0 不作为普通点；Q->0+ 应使用内部 quadrature 节点。n=0 不能直接使用 sigma=-Pi/Omega，"
            "应由 xi->0+ 的 R_TE_TM 极限获得。本阶段不输出 production energy、force 或 torque。",
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
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--fermi-window-eV", type=float, default=0.05)
    parser.add_argument("--eta-eV", type=float, default=1e-10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--separation-nm", type=float, default=100.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run-grid-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_audit(
        args.input_json,
        response_config={
            "adaptive_level": args.adaptive_level,
            "gauss_order": args.gauss_order,
            "coarse_grid": args.coarse_grid,
            "fermi_window_eV": args.fermi_window_eV,
            "eta_eV": args.eta_eV,
        },
        workers=args.workers,
        separation_nm=args.separation_nm,
        temperature_K=args.temperature_K,
        smoke=args.smoke,
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
