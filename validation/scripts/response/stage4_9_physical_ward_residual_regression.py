#!/usr/bin/env python3
"""Stage 4.9 Ward residual regression after the Stage 4.8 Kubo audit.

This script is diagnostic-only.  It does not tune residuals, change the bubble
formula, compute conductivity, or feed reflection/Casimir calculations.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.ward_response import (  # noqa: E402
    normal_density_current_response_imag_axis,
    normal_physical_density_current_response_imag_axis,
    physical_ward_residuals,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response"
JSON_OUTPUT = OUTPUT_DIR / "stage4_9_physical_ward_residual_regression.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_9_physical_ward_residual_regression.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
MESH_SIZE = 16
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
EPS = 1e-300

SLOPE_KEYS = (
    "max_norm",
    "left_norm",
    "right_norm",
    "left_spatial_source_norm",
    "right_spatial_observable_norm",
    "left_longitudinal_abs",
    "right_longitudinal_abs",
    "left_transverse_abs",
    "right_transverse_abs",
)


def to_jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-safe builtins and reject complex scalars."""

    if isinstance(value, complex | np.complexfloating):
        raise TypeError("complex values must be split into real/imag before JSON serialization")
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def project_spatial_components(q: np.ndarray, spatial: np.ndarray) -> tuple[complex, complex]:
    """Project a complex spatial vector onto q-longitudinal and transverse axes."""

    qnorm = float(np.linalg.norm(q))
    if qnorm <= 0.0:
        raise ValueError("q norm must be positive")
    qhat = q / qnorm
    that = np.array([-qhat[1], qhat[0]], dtype=float)
    longitudinal = qhat[0] * spatial[0] + qhat[1] * spatial[1]
    transverse = that[0] * spatial[0] + that[1] * spatial[1]
    return complex(longitudinal), complex(transverse)


def classify_status(smallest_q_max_error: float, max_norm_slope: float) -> str:
    """Classify the diagnostic status using the fixed Stage 4.9 rules."""

    if smallest_q_max_error < 1e-10:
        return "NUMERICALLY_CLOSED"
    if 0.75 <= max_norm_slope <= 1.25:
        return "ORDER_Q_RESIDUAL"
    if max_norm_slope > 1.25:
        return "ORDER_Q2_OR_BETTER_RESIDUAL"
    return "NON_SCALING_OR_UNCLEAR_RESIDUAL"


def fixed_next_step(status: str) -> str:
    """Return the fixed next-step sentence for a Stage 4.9 status."""

    if status == "NUMERICALLY_CLOSED":
        return "Next: document Ward closure and add a non-regression test. Do not enter conductivity/reflection/Casimir until the closure proof is written."
    if status == "ORDER_Q_RESIDUAL":
        return "Next: audit equal-time / commutator completion. Do not change bubble signs or introduce fitting coefficients."
    if status == "ORDER_Q2_OR_BETTER_RESIDUAL":
        return "Next: inspect whether remaining residual is numerical quadrature / finite-mesh error or an analytic O(q^2) term. Do not proceed to conductivity until this is classified."
    if status == "NON_SCALING_OR_UNCLEAR_RESIDUAL":
        return "Next: recheck scalar source convention, density vertex normalization, and finite-q reverse matrix element relation."
    raise ValueError(f"unknown status: {status}")


def _residual_row(response_label: str, pi: np.ndarray, omega_eV: float, q: np.ndarray, q_scale: float) -> dict[str, Any]:
    left, right = physical_ward_residuals(pi, omega_eV, q)
    pi_norm = float(np.linalg.norm(pi))
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    max_norm = max(left_norm, right_norm)
    scale = max(pi_norm, EPS)
    left_longitudinal, left_transverse = project_spatial_components(q, left[1:])
    right_longitudinal, right_transverse = project_spatial_components(q, right[1:])
    return {
        "response_label": response_label,
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "pi_norm": pi_norm,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "max_norm": max_norm,
        "left_error": float(left_norm / scale),
        "right_error": float(right_norm / scale),
        "max_error": float(max(left_norm / scale, right_norm / scale)),
        "left_real": [float(item.real) for item in left],
        "left_imag": [float(item.imag) for item in left],
        "right_real": [float(item.real) for item in right],
        "right_imag": [float(item.imag) for item in right],
        "left_density_source_abs": float(abs(left[0])),
        "left_spatial_source_norm": float(np.linalg.norm(left[1:])),
        "right_density_observable_abs": float(abs(right[0])),
        "right_spatial_observable_norm": float(np.linalg.norm(right[1:])),
        "left_longitudinal_abs": float(abs(left_longitudinal)),
        "left_transverse_abs": float(abs(left_transverse)),
        "right_longitudinal_abs": float(abs(right_longitudinal)),
        "right_transverse_abs": float(abs(right_transverse)),
    }


def _compute_slopes(rows: list[dict[str, Any]]) -> dict[str, float]:
    q_norms = np.array([float(row["q_norm"]) for row in rows], dtype=float)
    x = np.log(q_norms)
    slopes: dict[str, float] = {}
    for key in SLOPE_KEYS:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        y = np.log(np.maximum(values, EPS))
        slopes[key] = float(np.polyfit(x, y, 1)[0])
    return slopes


def _stage47_historical_response(mesh: np.ndarray, config: KuboConfig, q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    b_code = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="none",
    )
    pi_contact_plus = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="finite_q_peierls",
        contact_sign_convention="plus",
    )
    c_code = pi_contact_plus - b_code
    historical = np.array(b_code, copy=True)
    historical[0, 0] = b_code[0, 0]
    historical[0, 1:] = -b_code[0, 1:]
    historical[1:, 0] = -b_code[1:, 0]
    historical[1:, 1:] = b_code[1:, 1:] - c_code[1:, 1:]
    return historical


def run_regression(mesh_size: int = MESH_SIZE) -> dict[str, Any]:
    """Compute the fixed Stage 4.9 Ward residual regression."""

    if mesh_size < 12:
        raise ValueError("mesh_size must be at least 12")
    omega_eV = bosonic_matsubara_energy_eV(MATSUBARA_INDEX, TEMPERATURE_K)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=TEMPERATURE_K,
        eta_eV=ETA_EV,
        output_si=False,
    )
    mesh = uniform_bz_mesh(mesh_size)
    weights = k_weights(mesh)

    response_specs = {
        "stage48_physical_observable_source": {
            "description": "-<J_mu P_nu> plus direct term, with J=(rho,-Vx,-Vy) and P=(rho,Vx,Vy).",
            "results_by_q_scale": [],
        },
        "stage47_historical_observable_observable": {
            "description": "historical Stage 4.7 diagnostic only; observable-observable bubble assembled from old pieces.",
            "results_by_q_scale": [],
        },
    }

    for q_scale in Q_SCALES:
        q = float(q_scale) * Q_BASE
        pi_stage48 = normal_physical_density_current_response_imag_axis(mesh, config, q, weights)
        pi_stage47 = _stage47_historical_response(mesh, config, q, weights)
        response_specs["stage48_physical_observable_source"]["results_by_q_scale"].append(
            _residual_row("stage48_physical_observable_source", pi_stage48, omega_eV, q, float(q_scale))
        )
        response_specs["stage47_historical_observable_observable"]["results_by_q_scale"].append(
            _residual_row("stage47_historical_observable_observable", pi_stage47, omega_eV, q, float(q_scale))
        )

    for response in response_specs.values():
        rows = response["results_by_q_scale"]
        slopes = _compute_slopes(rows)
        response["slopes"] = slopes
        response["status"] = classify_status(float(rows[-1]["max_error"]), float(slopes["max_norm"]))

    stage48_smallest = float(response_specs["stage48_physical_observable_source"]["results_by_q_scale"][-1]["max_error"])
    stage47_smallest = float(response_specs["stage47_historical_observable_observable"]["results_by_q_scale"][-1]["max_error"])
    ratio = stage48_smallest / max(stage47_smallest, EPS)
    if ratio < 0.5:
        interpretation = "Stage 4.8 source/observable split substantially reduces the normalized Ward residual relative to historical Stage 4.7."
    elif ratio <= 2.0:
        interpretation = "Stage 4.8 source/observable split changes the residual but does not substantially reduce it."
    else:
        interpretation = "Stage 4.8 source/observable split increases the normalized Ward residual relative to historical Stage 4.7."

    return {
        "stage": "Stage 4.9",
        "purpose": "Ward residual regression after Stage 4.8 Kubo bubble audit",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "mesh_size": int(mesh_size),
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in Q_SCALES],
        },
        "responses": response_specs,
        "stage48_vs_stage47": {
            "stage48_smallest_max_error": stage48_smallest,
            "stage47_smallest_max_error": stage47_smallest,
            "ratio_stage48_over_stage47": float(ratio),
            "fixed_interpretation": interpretation,
        },
        "boundary": {
            "no_residual_tuning": True,
            "no_bubble_formula_change": True,
            "no_conductivity_reflection_casimir": True,
            "does_not_claim_ward_closure_unless_status_closed": True,
        },
    }


def _dominance_summary(row: dict[str, Any]) -> dict[str, str]:
    side = "left" if float(row["left_norm"]) >= float(row["right_norm"]) else "right"
    if side == "left":
        block = "density" if float(row["left_density_source_abs"]) >= float(row["left_spatial_source_norm"]) else "spatial"
        channel = (
            "longitudinal"
            if float(row["left_longitudinal_abs"]) >= float(row["left_transverse_abs"])
            else "transverse"
        )
    else:
        block = (
            "density"
            if float(row["right_density_observable_abs"]) >= float(row["right_spatial_observable_norm"])
            else "spatial"
        )
        channel = (
            "longitudinal"
            if float(row["right_longitudinal_abs"]) >= float(row["right_transverse_abs"])
            else "transverse"
        )
    return {"side": side, "block": block, "channel": channel}


def _format_float(value: float) -> str:
    return f"{value:.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    """Render a Markdown report for the Stage 4.9 regression."""

    stage48 = data["responses"]["stage48_physical_observable_source"]
    stage47 = data["responses"]["stage47_historical_observable_observable"]
    stage48_rows = stage48["results_by_q_scale"]
    stage47_rows = stage47["results_by_q_scale"]
    smallest_stage48 = stage48_rows[-1]
    dominance = _dominance_summary(smallest_stage48)
    next_step = fixed_next_step(str(stage48["status"]))

    def scaling_table(rows: list[dict[str, Any]]) -> str:
        return _table(
            ("q_scale", "q_norm", "left_error", "right_error", "max_error", "max_norm"),
            [
                (
                    _format_float(float(row["q_scale"])),
                    _format_float(float(row["q_norm"])),
                    _format_float(float(row["left_error"])),
                    _format_float(float(row["right_error"])),
                    _format_float(float(row["max_error"])),
                    _format_float(float(row["max_norm"])),
                )
                for row in rows
            ],
        )

    def block_table(rows: list[dict[str, Any]]) -> str:
        return _table(
            (
                "q_scale",
                "left_density_source_abs",
                "left_spatial_source_norm",
                "right_density_observable_abs",
                "right_spatial_observable_norm",
            ),
            [
                (
                    _format_float(float(row["q_scale"])),
                    _format_float(float(row["left_density_source_abs"])),
                    _format_float(float(row["left_spatial_source_norm"])),
                    _format_float(float(row["right_density_observable_abs"])),
                    _format_float(float(row["right_spatial_observable_norm"])),
                )
                for row in rows
            ],
        )

    def lt_table(rows: list[dict[str, Any]]) -> str:
        return _table(
            (
                "q_scale",
                "left_longitudinal_abs",
                "left_transverse_abs",
                "right_longitudinal_abs",
                "right_transverse_abs",
            ),
            [
                (
                    _format_float(float(row["q_scale"])),
                    _format_float(float(row["left_longitudinal_abs"])),
                    _format_float(float(row["left_transverse_abs"])),
                    _format_float(float(row["right_longitudinal_abs"])),
                    _format_float(float(row["right_transverse_abs"])),
                )
                for row in rows
            ],
        )

    def slope_table(response_name: str, slopes: dict[str, float]) -> str:
        return _table(
            ("response", "quantity", "slope"),
            [(response_name, key, _format_float(float(value))) for key, value in slopes.items()],
        )

    return "\n\n".join(
        [
            "# Stage 4.9 Ward residual regression after Kubo bubble audit",
            "## Boundary\n\n- no residual tuning\n- no bubble formula change\n- no conductivity / reflection / Casimir\n- no claim of Ward closure unless `NUMERICALLY_CLOSED`",
            "## Fixed Response Formula\n\n"
            "$J=(\\rho,-V_x,-V_y)$\n\n"
            "$P=(\\rho,V_x,V_y)$\n\n"
            "$\\Pi_{\\mu\\nu}^{4.8}=-\\langle J_\\mu P_\\nu\\rangle+\\left\\langle\\delta J_\\mu/\\delta a_\\nu\\right\\rangle$",
            f"## Configuration\n\nmesh_size = {data['config']['mesh_size']}; temperature_K = {data['config']['temperature_K']}; matsubara_index = {data['config']['matsubara_index']}; omega_eV = {_format_float(float(data['config']['omega_eV']))}; q_base = {data['config']['q_base']}",
            "## Stage 4.8 q-scaling\n\n" + scaling_table(stage48_rows),
            "## Historical Stage 4.7 q-scaling\n\nhistorical Stage 4.7 diagnostic only\n\n" + scaling_table(stage47_rows),
            "## Left/Right Residual Decomposition\n\n### Stage 4.8\n\n"
            + block_table(stage48_rows)
            + "\n\n### Historical Stage 4.7\n\n"
            + block_table(stage47_rows),
            "## Longitudinal/Transverse Decomposition\n\n### Stage 4.8\n\n"
            + lt_table(stage48_rows)
            + "\n\n### Historical Stage 4.7\n\n"
            + lt_table(stage47_rows),
            "## Slope Table\n\n"
            + slope_table("stage48_physical_observable_source", stage48["slopes"])
            + "\n\n"
            + slope_table("stage47_historical_observable_observable", stage47["slopes"]),
            "## Stage 4.8 vs Stage 4.7\n\n"
            + _table(
                ("stage48_smallest_max_error", "stage47_smallest_max_error", "ratio_stage48_over_stage47"),
                [
                    (
                        _format_float(float(data["stage48_vs_stage47"]["stage48_smallest_max_error"])),
                        _format_float(float(data["stage48_vs_stage47"]["stage47_smallest_max_error"])),
                        _format_float(float(data["stage48_vs_stage47"]["ratio_stage48_over_stage47"])),
                    )
                ],
            )
            + f"\n\n{data['stage48_vs_stage47']['fixed_interpretation']}",
            f"## Final Diagnostic Status\n\n- Stage 4.8: `{stage48['status']}`\n- Historical Stage 4.7: `{stage47['status']}`\n\nDominance at smallest q for Stage 4.8: {dominance['side']} / {dominance['block']} / {dominance['channel']}.",
            "## Next Step\n\n" + next_step,
        ]
    ) + "\n"


def main() -> None:
    data = run_regression()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
