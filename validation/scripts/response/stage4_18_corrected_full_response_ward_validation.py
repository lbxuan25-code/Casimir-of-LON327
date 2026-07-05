#!/usr/bin/env python3
"""Stage 4.18 corrected full-response Ward residual validation.

Diagnostic-only.  This stage consolidates the corrected left/right Ward
residual convention found in Stage 4.17 and reruns the Stage 4.16 adaptive
full-response validation.  It does not modify the main response formula,
bubble prefactor sign, direct contact, source/observable split, conductivity,
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

from lno327 import bosonic_matsubara_energy_eV  # noqa: E402
from lno327.collective.ward import physical_ward_residuals, physical_ward_residuals_legacy  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import (  # noqa: E402
    Q_BASE,
    Q_SCALES,
    config_for_temperature,
    integrate_physical_components_on_points,
    project_spatial_components,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_18_corrected_full_response_ward_validation.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_18_corrected_full_response_ward_validation.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
OUTPUT_SI = False
COARSE_GRID = 32
MAX_REFINEMENT_LEVEL = 4
GAUSS_ORDER = 3
FERMI_WINDOW_EV = 0.05
EPS = 1e-300


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


def corrected_residual_row(
    *,
    q_scale: float,
    q: np.ndarray,
    response: np.ndarray,
    omega_eV: float,
    num_cells_total: int,
    num_cells_refined: int,
    num_quadrature_points: int,
) -> dict[str, Any]:
    left, right = physical_ward_residuals(response, omega_eV, q)
    _legacy_left, legacy_right = physical_ward_residuals_legacy(response, omega_eV, q)
    left_long, left_trans = project_spatial_components(q, left[1:])
    right_long, right_trans = project_spatial_components(q, right[1:])
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    legacy_right_norm = float(np.linalg.norm(legacy_right))
    return {
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "max_corrected_norm": max(left_norm, right_norm),
        "legacy_right_norm": legacy_right_norm,
        "legacy_over_corrected_norm": float(legacy_right_norm / max(max(left_norm, right_norm), EPS)),
        "left_density_source_abs": float(abs(left[0])),
        "left_spatial_source_norm": float(np.linalg.norm(left[1:])),
        "right_density_observable_abs": float(abs(right[0])),
        "right_spatial_observable_norm": float(np.linalg.norm(right[1:])),
        "left_longitudinal_abs": float(abs(left_long)),
        "left_transverse_abs": float(abs(left_trans)),
        "right_longitudinal_abs": float(abs(right_long)),
        "right_transverse_abs": float(abs(right_trans)),
        "left_residual": _complex_vector_parts(left),
        "right_residual": _complex_vector_parts(right),
        "legacy_right_residual": _complex_vector_parts(legacy_right),
        "num_cells_total": int(num_cells_total),
        "num_cells_refined": int(num_cells_refined),
        "num_quadrature_points": int(num_quadrature_points),
        "same_points_weights_for_bubble_and_direct": True,
    }


def validation_row(
    q_scale: float,
    q: np.ndarray,
    *,
    coarse_grid: int,
    max_refinement_level: int,
    gauss_order: int,
    fermi_window_eV: float,
) -> dict[str, Any]:
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
    row = corrected_residual_row(
        q_scale=q_scale,
        q=q,
        response=components["total"],
        omega_eV=config.omega_eV,
        num_cells_total=len(cells),
        num_cells_refined=refined_count,
        num_quadrature_points=len(points),
    )
    row["num_base_fermi_window_cells"] = int(flagged_base)
    return row


def legacy_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "q_scale": float(row["q_scale"]),
        "q_model": list(row["q_model"]),
        "corrected_right_norm": float(row["right_norm"]),
        "legacy_right_norm": float(row["legacy_right_norm"]),
        "legacy_over_corrected_norm": float(row["legacy_over_corrected_norm"]),
        "legacy_right_residual": row["legacy_right_residual"],
    }


def dominant_remaining_channel(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    metrics = {
        "left_density_source": max(float(row["left_density_source_abs"]) for row in rows),
        "left_spatial_source": max(float(row["left_spatial_source_norm"]) for row in rows),
        "right_density_observable": max(float(row["right_density_observable_abs"]) for row in rows),
        "right_spatial_observable": max(float(row["right_spatial_observable_norm"]) for row in rows),
        "left_longitudinal": max(float(row["left_longitudinal_abs"]) for row in rows),
        "left_transverse": max(float(row["left_transverse_abs"]) for row in rows),
        "right_longitudinal": max(float(row["right_longitudinal_abs"]) for row in rows),
        "right_transverse": max(float(row["right_transverse_abs"]) for row in rows),
    }
    return max(metrics, key=metrics.get)


def classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "corrected_ward_status": "CORRECTED_WARD_IMPROVED_BUT_NOT_CLOSED",
            "right_convention_status": "RIGHT_WARD_CONVENTION_FIX_FAILED",
            "max_corrected_norm": float("inf"),
            "max_legacy_right_norm": float("inf"),
            "dominant_remaining_channel": "unknown",
            "likely_issue": "RIGHT_RESIDUAL_NOT_EXPLAINED_BY_CONVENTION",
            "next_step": "Rerun the diagnostic with non-empty q scales.",
        }
    max_corrected = max(float(row["max_corrected_norm"]) for row in rows)
    max_legacy = max(float(row["legacy_right_norm"]) for row in rows)
    left_norms = [float(row["left_norm"]) for row in rows]
    right_norms = [float(row["right_norm"]) for row in rows]
    legacy_norms = [float(row["legacy_right_norm"]) for row in rows]
    same_order = all(
        right <= 10.0 * max(left, EPS) and left <= 10.0 * max(right, EPS)
        for left, right in zip(left_norms, right_norms, strict=True)
    )
    legacy_large = all(1e-3 <= legacy <= 1e-1 for legacy in legacy_norms)
    corrected_beats_legacy = all(right < legacy for right, legacy in zip(right_norms, legacy_norms, strict=True))
    if max_corrected < 1e-6:
        corrected_status = "CORRECTED_WARD_NUMERICALLY_CLOSED"
    elif corrected_beats_legacy:
        corrected_status = "CORRECTED_WARD_IMPROVED_BUT_NOT_CLOSED"
    else:
        corrected_status = "RIGHT_WARD_CONVENTION_FIX_FAILED"
    if same_order and legacy_large:
        right_status = "RIGHT_WARD_CONVENTION_FIX_VALIDATED"
    elif corrected_beats_legacy:
        right_status = "RIGHT_WARD_CONVENTION_FIX_PARTIALLY_VALIDATED"
    else:
        right_status = "RIGHT_WARD_CONVENTION_FIX_FAILED"
    if corrected_status == "CORRECTED_WARD_NUMERICALLY_CLOSED":
        likely = "PREVIOUS_RIGHT_RESIDUAL_WAS_DIAGNOSTIC_CONVENTION"
        next_step = "Next: Stage 4.19 multi-parameter robustness scan before any conductivity/reflection/Casimir use."
    elif corrected_status == "CORRECTED_WARD_IMPROVED_BUT_NOT_CLOSED":
        likely = "CORRECTED_CONVENTION_PLUS_REMAINING_NUMERICAL_OR_ROUTING_ERROR"
        next_step = "Next: refine the remaining channel and run Stage 4.19 robustness checks before any downstream use."
    else:
        likely = "RIGHT_RESIDUAL_NOT_EXPLAINED_BY_CONVENTION"
        next_step = "Next: audit source/observable routing and density/contact conventions again; do not change response formula from this result alone."
    return {
        "corrected_ward_status": corrected_status,
        "right_convention_status": right_status,
        "max_corrected_norm": float(max_corrected),
        "max_legacy_right_norm": float(max_legacy),
        "dominant_remaining_channel": dominant_remaining_channel(rows),
        "likely_issue": likely,
        "next_step": next_step,
    }


def run_validation(
    *,
    coarse_grid: int = COARSE_GRID,
    max_refinement_level: int = MAX_REFINEMENT_LEVEL,
    gauss_order: int = GAUSS_ORDER,
    fermi_window_eV: float = FERMI_WINDOW_EV,
    q_scales: tuple[float, ...] | list[float] = Q_SCALES,
) -> dict[str, Any]:
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    rows = [
        validation_row(
            float(q_scale),
            float(q_scale) * Q_BASE,
            coarse_grid=coarse_grid,
            max_refinement_level=max_refinement_level,
            gauss_order=gauss_order,
            fermi_window_eV=fermi_window_eV,
        )
        for q_scale in q_scales
    ]
    return {
        "stage": "Stage 4.18",
        "purpose": "Consolidate corrected left/right Ward residual convention and validate full response",
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
            "right_residual_convention": "iOmega Pi[mu,0] - qx Pi[mu,x] - qy Pi[mu,y]",
            "legacy_right_residual_convention": "iOmega Pi[mu,0] + qx Pi[mu,x] + qy Pi[mu,y]",
        },
        "corrected_validation_results": rows,
        "legacy_comparison_results": [legacy_comparison_row(row) for row in rows],
        "diagnostic_status": classify(rows),
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
    rows = data["corrected_validation_results"]
    status = data["diagnostic_status"]
    corrected_table = _table(
        (
            "q_scale",
            "left_norm",
            "right_norm",
            "max_corrected",
            "left_long",
            "left_trans",
            "right_long",
            "right_trans",
            "quad points",
        ),
        [
            (
                _fmt(float(row["q_scale"])),
                _fmt(float(row["left_norm"])),
                _fmt(float(row["right_norm"])),
                _fmt(float(row["max_corrected_norm"])),
                _fmt(float(row["left_longitudinal_abs"])),
                _fmt(float(row["left_transverse_abs"])),
                _fmt(float(row["right_longitudinal_abs"])),
                _fmt(float(row["right_transverse_abs"])),
                row["num_quadrature_points"],
            )
            for row in rows
        ],
    )
    legacy_table = _table(
        ("q_scale", "corrected right", "legacy right", "legacy/corrected"),
        [
            (
                _fmt(float(row["q_scale"])),
                _fmt(float(row["right_norm"])),
                _fmt(float(row["legacy_right_norm"])),
                _fmt(float(row["legacy_over_corrected_norm"])),
            )
            for row in rows
        ],
    )
    return "\n\n".join(
        [
            "# Stage 4.18 Corrected full response Ward validation",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no source/observable change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir",
            "## Corrected Ward residual convention\n\n"
            "$$R_L[\\nu]=i\\Omega\\Pi_{0\\nu}+q_x\\Pi_{x\\nu}+q_y\\Pi_{y\\nu},$$\n\n"
            "$$R_R[\\mu]=i\\Omega\\Pi_{\\mu0}-q_x\\Pi_{\\mu x}-q_y\\Pi_{\\mu y}.$$\n\n"
            "The legacy right residual $i\\Omega\\Pi_{\\mu0}+q_x\\Pi_{\\mu x}+q_y\\Pi_{\\mu y}$ is kept only as an old diagnostic comparison and is not a closure criterion.",
            "## Analytic derivation summary\n\n"
            "Stage 4.13 fixed the bubble sign. Stage 4.15 addressed the $C-K$ quadrature issue. Stage 4.17 found the right Ward diagnostic convention problem. Stage 4.18 does not alter the response formula; it only consolidates the residual diagnostic definition.\n\n"
            "The asymmetric left/right signs follow from $J_i=-V_i$ and $P_i=V_i$, together with\n\n"
            "$$G_+^{-1}-G_-^{-1}=i\\Omega\\rho-q_iV_i.$$",
            "## Adaptive full-response setup\n\n"
            "The validation reuses the Stage 4.16 adaptive Fermi-window points and weights. Bubble and direct contact use identical integration points and weights. The response remains $\\Pi_{\\mu\\nu}=\\Pi_{\\mu\\nu}^{bubble}+D_{\\mu\\nu}$ with corrected positive bubble prefactor and unchanged $D_{ij}=-\\langle M_{ij}\\rangle$.",
            "## Corrected left/right Ward residuals\n\n" + corrected_table,
            "## Legacy right residual comparison\n\n" + legacy_table,
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("corrected_ward_status", status["corrected_ward_status"]),
                    ("right_convention_status", status["right_convention_status"]),
                    ("max_corrected_norm", _fmt(float(status["max_corrected_norm"]))),
                    ("max_legacy_right_norm", _fmt(float(status["max_legacy_right_norm"]))),
                    ("dominant_remaining_channel", status["dominant_remaining_channel"]),
                    ("likely_issue", status["likely_issue"]),
                ],
            ),
            "## Next step\n\n"
            + status["next_step"]
            + "\n\nBefore conductivity/reflection/Casimir use, Stage 4.19 must perform a multi-parameter robustness scan.",
        ]
    ) + "\n"


def main() -> None:
    data = run_validation()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
