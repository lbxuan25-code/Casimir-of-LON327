#!/usr/bin/env python3
"""Stage 4.17 right Ward source-side sign/routing audit.

Diagnostic-only.  This script reuses the Stage 4.16 adaptive full-response
quadrature and compares right Ward source-side sign candidates.  It does not
modify the main response, bubble sign, direct contact, conductivity,
reflection, or Casimir code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import bosonic_matsubara_energy_eV  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import (  # noqa: E402
    Q_BASE,
    Q_SCALES,
    config_for_temperature,
    integrate_physical_components_on_points,
    project_spatial_components,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_17_right_ward_source_convention_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_17_right_ward_source_convention_audit.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
OUTPUT_SI = False
COARSE_GRID = 32
MAX_REFINEMENT_LEVEL = 4
GAUSS_ORDER = 3
FERMI_WINDOW_EV = 0.05
EPS = 1e-300

RIGHT_CANDIDATES = (
    ("R_right_plus_omega_plus_q", 1.0, 1.0),
    ("R_right_plus_omega_minus_q", 1.0, -1.0),
    ("R_right_minus_omega_plus_q", -1.0, 1.0),
    ("R_right_minus_omega_minus_q", -1.0, -1.0),
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        raise TypeError("complex values must be split before JSON serialization")
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _complex_vector_parts(vector: np.ndarray) -> dict[str, list[float]]:
    return {
        "real": [float(item.real) for item in vector],
        "imag": [float(item.imag) for item in vector],
        "abs": [float(abs(item)) for item in vector],
    }


def _right_residual(response: np.ndarray, omega_eV: float, q: np.ndarray, omega_sign: float, q_sign: float) -> np.ndarray:
    qx, qy = float(q[0]), float(q[1])
    return (
        omega_sign * 1j * omega_eV * response[:, 0]
        + q_sign * response[:, 1] * qx
        + q_sign * response[:, 2] * qy
    )


def _candidate_parts(candidate: str, residual: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    longitudinal, transverse = project_spatial_components(q, residual[1:])
    return {
        "candidate": candidate,
        "right_norm": float(np.linalg.norm(residual)),
        "right_density_observable_abs": float(abs(residual[0])),
        "right_spatial_observable_norm": float(np.linalg.norm(residual[1:])),
        "right_longitudinal_abs": float(abs(longitudinal)),
        "right_transverse_abs": float(abs(transverse)),
        "right_residual_vector": _complex_vector_parts(residual),
    }


def audit_row(q_scale: float, q: np.ndarray, *, coarse_grid: int, max_refinement_level: int, gauss_order: int, fermi_window_eV: float) -> dict[str, Any]:
    config = config_for_temperature()
    cells, refined_count, flagged_base = build_adaptive_cells(
        q,
        coarse_grid=int(coarse_grid),
        refinement_level=int(max_refinement_level),
        fermi_window_eV=float(fermi_window_eV),
        fermi_level_eV=config.fermi_level_eV,
    )
    points, weights = quadrature_points_for_cells(cells, int(gauss_order))
    components = integrate_physical_components_on_points(points, weights, q, config)
    response = components["total"]
    left, _right_plus_plus = physical_ward_residuals(response, config.omega_eV, q)
    left_long, left_trans = project_spatial_components(q, left[1:])
    candidates = []
    for name, omega_sign, q_sign in RIGHT_CANDIDATES:
        residual = _right_residual(response, config.omega_eV, q, omega_sign, q_sign)
        candidates.append(_candidate_parts(name, residual, q))
    best = min(candidates, key=lambda item: float(item["right_norm"]))
    plus_plus = next(item for item in candidates if item["candidate"] == "R_right_plus_omega_plus_q")
    predicted = next(item for item in candidates if item["candidate"] == "R_right_plus_omega_minus_q")
    return {
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "left_norm": float(np.linalg.norm(left)),
        "left_density_source_abs": float(abs(left[0])),
        "left_spatial_source_norm": float(np.linalg.norm(left[1:])),
        "left_longitudinal_abs": float(abs(left_long)),
        "left_transverse_abs": float(abs(left_trans)),
        "left_residual_vector": _complex_vector_parts(left),
        "right_candidates": candidates,
        "best_right_candidate": best["candidate"],
        "best_right_norm": best["right_norm"],
        "predicted_candidate_norm": predicted["right_norm"],
        "plus_plus_candidate_norm": plus_plus["right_norm"],
        "predicted_over_plus_plus_norm": float(predicted["right_norm"] / max(float(plus_plus["right_norm"]), EPS)),
        "num_cells_total": len(cells),
        "num_cells_refined": int(refined_count),
        "num_base_fermi_window_cells": int(flagged_base),
        "num_quadrature_points": len(points),
        "same_points_weights_for_all_candidates": True,
    }


def dominant_channel(rows: list[dict[str, Any]], candidate: str) -> str:
    candidate_rows = []
    for row in rows:
        candidate_rows.append(next(item for item in row["right_candidates"] if item["candidate"] == candidate))
    metrics = {
        "right_density_observable": max(float(item["right_density_observable_abs"]) for item in candidate_rows),
        "right_spatial_observable": max(float(item["right_spatial_observable_norm"]) for item in candidate_rows),
        "right_longitudinal": max(float(item["right_longitudinal_abs"]) for item in candidate_rows),
        "right_transverse": max(float(item["right_transverse_abs"]) for item in candidate_rows),
    }
    return max(metrics, key=metrics.get)


def classify(rows: list[dict[str, Any]]) -> dict[str, str]:
    best_candidates = [str(row["best_right_candidate"]) for row in rows]
    best_global = best_candidates[0] if best_candidates and all(item == best_candidates[0] for item in best_candidates) else "INCONSISTENT"
    predicted_best = best_global == "R_right_plus_omega_minus_q"
    ratios = [float(row["predicted_over_plus_plus_norm"]) for row in rows]
    predicted_norms = [float(row["predicted_candidate_norm"]) for row in rows]
    left_norms = [float(row["left_norm"]) for row in rows]
    improves_100 = predicted_best and ratios and all(ratio < 0.01 for ratio in ratios)
    max_predicted = max(predicted_norms) if predicted_norms else float("inf")
    left_scale = max(max(left_norms), EPS) if left_norms else EPS
    close_to_left = max_predicted / left_scale < 10.0
    if improves_100:
        source_status = "RIGHT_WARD_SOURCE_SIGN_CONFIRMED"
    elif best_global == "INCONSISTENT":
        source_status = "RIGHT_WARD_SOURCE_SIGN_INCONCLUSIVE"
    else:
        source_status = "RIGHT_WARD_NOT_EXPLAINED_BY_SOURCE_SIGN"
    if predicted_best and close_to_left and max_predicted < 1e-6:
        closure = "RIGHT_WARD_NUMERICALLY_CLOSED"
    elif predicted_best and (improves_100 or max_predicted < max(float(row["plus_plus_candidate_norm"]) for row in rows)):
        closure = "RIGHT_WARD_SIGN_CONFIRMED_BUT_NOT_CLOSED"
    else:
        closure = "RIGHT_WARD_NOT_EXPLAINED_BY_SOURCE_SIGN"
    if closure == "RIGHT_WARD_NUMERICALLY_CLOSED":
        likely = "RIGHT_WARD_DIAGNOSTIC_SIGN_CONVENTION"
        next_step = "Next: update diagnostic Ward residual convention docs/tests, then rerun full response validation without changing the response formula."
    elif closure == "RIGHT_WARD_SIGN_CONFIRMED_BUT_NOT_CLOSED":
        likely = "RIGHT_WARD_SIGN_PLUS_REMAINING_DENSITY_OR_ROUTING"
        next_step = "Next: audit finite-q density vertex embedding and source routing using the right Ward plus-omega/minus-q diagnostic convention."
    else:
        likely = "FINITE_Q_DENSITY_VERTEX_OR_SOURCE_ROUTING_NOT_SIGN_ONLY"
        next_step = "Next: audit finite-q density vertex embedding/source routing; source-side q sign alone does not explain the residual."
    return {
        "best_candidate_global": best_global,
        "right_source_sign_status": source_status,
        "closure_status": closure,
        "dominant_remaining_channel": dominant_channel(rows, "R_right_plus_omega_minus_q") if rows else "unknown",
        "likely_issue": likely,
        "next_step": next_step,
    }


def run_audit(
    *,
    coarse_grid: int = COARSE_GRID,
    max_refinement_level: int = MAX_REFINEMENT_LEVEL,
    gauss_order: int = GAUSS_ORDER,
    fermi_window_eV: float = FERMI_WINDOW_EV,
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
) -> dict[str, Any]:
    rows = [
        audit_row(
            float(q_scale),
            float(q_scale) * Q_BASE,
            coarse_grid=coarse_grid,
            max_refinement_level=max_refinement_level,
            gauss_order=gauss_order,
            fermi_window_eV=fermi_window_eV,
        )
        for q_scale in q_scales
    ]
    status = classify(rows)
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    return {
        "stage": "Stage 4.17",
        "purpose": "Right Ward source-side sign and routing audit",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "output_si": OUTPUT_SI,
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in q_scales],
            "coarse_grid": int(coarse_grid),
            "max_refinement_level": int(max_refinement_level),
            "gauss_order": int(gauss_order),
            "fermi_window_eV": float(fermi_window_eV),
            "predicted_best_candidate": "R_right_plus_omega_minus_q",
        },
        "results": rows,
        "diagnostic_status": status,
        "boundary": {
            "no_main_response_change": True,
            "no_bubble_sign_change": True,
            "no_direct_contact_change": True,
            "no_source_observable_change": True,
            "no_residual_tuning": True,
            "no_fitted_contact": True,
            "no_E_ET_added": True,
            "no_conductivity_reflection_casimir": True,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    status = data["diagnostic_status"]
    rows = data["results"]
    comparison_rows = []
    for row in rows:
        values = {item["candidate"]: item["right_norm"] for item in row["right_candidates"]}
        comparison_rows.append(
            (
                _fmt(float(row["q_scale"])),
                _fmt(float(row["left_norm"])),
                _fmt(float(values["R_right_plus_omega_plus_q"])),
                _fmt(float(values["R_right_plus_omega_minus_q"])),
                _fmt(float(values["R_right_minus_omega_plus_q"])),
                _fmt(float(values["R_right_minus_omega_minus_q"])),
                row["best_right_candidate"],
            )
        )
    return "\n\n".join(
        [
            "# Stage 4.17 Right Ward source-side convention audit",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no source/observable change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir",
            "## Analytic source-side Ward identity\n\n"
            "$$G_+^{-1}-G_-^{-1}=i\\Omega\\rho-q_iV_i.$$\n\n"
            "Because $P_0=\\rho$ and $P_i=V_i$, the natural right Ward source-side contraction is $i\\Omega\\Pi_{\\mu0}-q_i\\Pi_{\\mu i}$.",
            "## Candidate definitions\n\n"
            "Candidates: plus/plus, plus/minus, minus/plus, and minus/minus in the omega/q signs. The predicted candidate is `R_right_plus_omega_minus_q`.",
            "## Adaptive full-response setup\n\n"
            "Uses Stage 4.16 adaptive full-response quadrature with corrected Stage 4.13 bubble sign and unchanged direct contact.",
            "## Left Ward reference\n\n"
            + _table(("q_scale", "left_norm"), [(_fmt(float(row["q_scale"])), _fmt(float(row["left_norm"]))) for row in rows]),
            "## Right Ward candidate comparison\n\n"
            + _table(
                ("q_scale", "left", "++", "+-", "-+", "--", "best"),
                comparison_rows,
            ),
            "## Best candidate decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("best_candidate_global", status["best_candidate_global"]),
                    ("right_source_sign_status", status["right_source_sign_status"]),
                    ("closure_status", status["closure_status"]),
                    ("dominant_remaining_channel", status["dominant_remaining_channel"]),
                    ("likely_issue", status["likely_issue"]),
                ],
            ),
            "## Diagnostic decision\n\nDo not change the response formula from this diagnostic alone.",
            "## Next step\n\n" + status["next_step"],
        ]
    ) + "\n"


def main() -> None:
    data = run_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
