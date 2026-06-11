#!/usr/bin/env python3
"""Stage 4.10 equal-time / commutator completion audit.

This is a diagnostic-only script.  It does not change the Kubo bubble formula,
the main response path, conductivity, reflection, or Casimir calculations.
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
from lno327.tb_fourier import (  # noqa: E402
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)
from lno327.ward_response import (  # noqa: E402
    normal_physical_density_current_response_components_imag_axis,
    normal_physical_density_current_response_imag_axis,
    physical_ward_residuals,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_10_equal_time_commutator_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_10_equal_time_commutator_audit.md"

TEMPERATURE_K = 30.0
MATSUBARA_INDEX = 1
ETA_EV = 1e-10
MESH_SIZE = 16
Q_BASE = np.array([0.02, 0.013], dtype=float)
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
SAMPLE_K_POINTS = ((0.17, -0.23), (0.41, 0.19), (-0.37, 0.29))
EPS = 1e-300

SLOPE_KEYS = (
    "total_max_norm",
    "left_total_norm",
    "right_total_norm",
    "left_missing_norm",
    "right_missing_norm",
    "left_total_longitudinal_abs",
    "right_total_longitudinal_abs",
    "left_missing_longitudinal_abs",
    "right_missing_longitudinal_abs",
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


def classify_second_order_identity(max_rel_error: float) -> str:
    return "MATCH" if max_rel_error < 1e-10 else "MISMATCH"


def classify_direct_completion(smallest_total_max_error: float, total_max_norm_slope: float) -> str:
    if smallest_total_max_error < 1e-10:
        return "DIRECT_TERM_NUMERICALLY_CLOSES_WARD"
    if 0.75 <= total_max_norm_slope <= 1.25:
        return "DIRECT_TERM_LEAVES_ORDER_Q_RESIDUAL"
    if total_max_norm_slope > 1.25:
        return "DIRECT_TERM_LEAVES_ORDER_Q2_OR_BETTER_RESIDUAL"
    return "DIRECT_TERM_LEAVES_NONSCALING_RESIDUAL"


def fixed_next_step(second_order_status: str, direct_status: str) -> str:
    if second_order_status == "MATCH" and direct_status == "DIRECT_TERM_LEAVES_ORDER_Q_RESIDUAL":
        return "Next: derive and evaluate the explicit equal-time commutator E_ET. After the Stage 4.13 bubble prefactor fix, do not tune signs further or fit contact coefficients."
    if second_order_status == "MISMATCH":
        return "Next: revisit finite-q Hamiltonian contact vertex M_ij before adding any commutator term."
    if direct_status == "DIRECT_TERM_NUMERICALLY_CLOSES_WARD":
        return "Next: document Ward closure and add a non-regression test. Do not proceed to conductivity/reflection/Casimir until closure proof is written."
    return "Next: inspect scalar source convention, density normalization, and finite-q matrix-element routing before adding any response correction."


def second_order_identity_sample(kx: float, ky: float, q: np.ndarray, direction_j: str) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    m_xj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", direction_j)
    m_yj = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "y", direction_j)
    lhs = qx * m_xj + qy * m_yj
    v_plus = peierls_hamiltonian_vector_vertex(
        kx + 0.5 * qx,
        ky + 0.5 * qy,
        qx,
        qy,
        direction_j,
    )
    v_minus = peierls_hamiltonian_vector_vertex(
        kx - 0.5 * qx,
        ky - 0.5 * qy,
        qx,
        qy,
        direction_j,
    )
    rhs = v_plus - v_minus
    abs_error = float(np.linalg.norm(lhs - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    rel_error = abs_error / max(rhs_norm, EPS)
    return {
        "kx": float(kx),
        "ky": float(ky),
        "q_model": [qx, qy],
        "direction_j": direction_j,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def _split_complex_vector(vector: np.ndarray) -> dict[str, list[float]]:
    return {
        "real": [float(item.real) for item in vector],
        "imag": [float(item.imag) for item in vector],
    }


def _residual_parts(pi: np.ndarray, omega_eV: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return physical_ward_residuals(pi, omega_eV, q)


def _lt_abs(q: np.ndarray, vector: np.ndarray) -> tuple[float, float]:
    longitudinal, transverse = project_spatial_components(q, vector[1:])
    return float(abs(longitudinal)), float(abs(transverse))


def residual_decomposition_row(q_scale: float, q: np.ndarray, omega_eV: float, components: dict[str, np.ndarray]) -> dict[str, Any]:
    left_bubble, right_bubble = _residual_parts(components["bubble"], omega_eV, q)
    left_direct, right_direct = _residual_parts(components["direct"], omega_eV, q)
    left_total, right_total = _residual_parts(components["total"], omega_eV, q)
    left_missing = -(left_bubble + left_direct)
    right_missing = -(right_bubble + right_direct)
    left_bubble_long, left_bubble_trans = _lt_abs(q, left_bubble)
    left_direct_long, left_direct_trans = _lt_abs(q, left_direct)
    left_total_long, left_total_trans = _lt_abs(q, left_total)
    left_missing_long, left_missing_trans = _lt_abs(q, left_missing)
    right_bubble_long, right_bubble_trans = _lt_abs(q, right_bubble)
    right_direct_long, right_direct_trans = _lt_abs(q, right_direct)
    right_total_long, right_total_trans = _lt_abs(q, right_total)
    right_missing_long, right_missing_trans = _lt_abs(q, right_missing)
    pi_norm = float(np.linalg.norm(components["total"]))
    left_total_norm = float(np.linalg.norm(left_total))
    right_total_norm = float(np.linalg.norm(right_total))
    total_max_norm = max(left_total_norm, right_total_norm)
    return {
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "pi_total_norm": pi_norm,
        "total_max_error": float(total_max_norm / max(pi_norm, EPS)),
        "total_max_norm": total_max_norm,
        "left_bubble_norm": float(np.linalg.norm(left_bubble)),
        "left_direct_norm": float(np.linalg.norm(left_direct)),
        "left_total_norm": left_total_norm,
        "left_missing_norm": float(np.linalg.norm(left_missing)),
        "right_bubble_norm": float(np.linalg.norm(right_bubble)),
        "right_direct_norm": float(np.linalg.norm(right_direct)),
        "right_total_norm": right_total_norm,
        "right_missing_norm": float(np.linalg.norm(right_missing)),
        "left_bubble_spatial_source": _split_complex_vector(left_bubble[1:]),
        "left_direct_spatial_source": _split_complex_vector(left_direct[1:]),
        "left_total_spatial_source": _split_complex_vector(left_total[1:]),
        "left_missing_spatial_source": _split_complex_vector(left_missing[1:]),
        "left_bubble_longitudinal_abs": left_bubble_long,
        "left_direct_longitudinal_abs": left_direct_long,
        "left_total_longitudinal_abs": left_total_long,
        "left_missing_longitudinal_abs": left_missing_long,
        "left_bubble_transverse_abs": left_bubble_trans,
        "left_direct_transverse_abs": left_direct_trans,
        "left_total_transverse_abs": left_total_trans,
        "left_missing_transverse_abs": left_missing_trans,
        "right_bubble_longitudinal_abs": right_bubble_long,
        "right_direct_longitudinal_abs": right_direct_long,
        "right_total_longitudinal_abs": right_total_long,
        "right_missing_longitudinal_abs": right_missing_long,
        "right_bubble_transverse_abs": right_bubble_trans,
        "right_direct_transverse_abs": right_direct_trans,
        "right_total_transverse_abs": right_total_trans,
        "right_missing_transverse_abs": right_missing_trans,
    }


def compute_slopes(rows: list[dict[str, Any]]) -> dict[str, float]:
    q_norms = np.array([float(row["q_norm"]) for row in rows], dtype=float)
    x = np.log(q_norms)
    slopes: dict[str, float] = {}
    for key in SLOPE_KEYS:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        slopes[key] = float(np.polyfit(x, np.log(np.maximum(values, EPS)), 1)[0])
    return slopes


def run_audit(mesh_size: int = MESH_SIZE) -> dict[str, Any]:
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

    identity_samples = []
    for q_scale in Q_SCALES:
        q = float(q_scale) * Q_BASE
        for kx, ky in SAMPLE_K_POINTS:
            for direction_j in ("x", "y"):
                identity_samples.append(second_order_identity_sample(kx, ky, q, direction_j))
    max_abs = max(float(item["abs_error"]) for item in identity_samples)
    max_rel = max(float(item["rel_error"]) for item in identity_samples)
    second_order_status = classify_second_order_identity(max_rel)

    rows = []
    for q_scale in Q_SCALES:
        q = float(q_scale) * Q_BASE
        components = normal_physical_density_current_response_components_imag_axis(mesh, config, q, weights)
        total = normal_physical_density_current_response_imag_axis(mesh, config, q, weights)
        if not np.allclose(components["total"], total):
            raise RuntimeError("components['total'] does not match normal_physical_density_current_response_imag_axis")
        rows.append(residual_decomposition_row(float(q_scale), q, omega_eV, components))
    slopes = compute_slopes(rows)
    direct_status = classify_direct_completion(float(rows[-1]["total_max_error"]), float(slopes["total_max_norm"]))
    next_step = fixed_next_step(second_order_status, direct_status)

    return {
        "stage": "Stage 4.10",
        "purpose": "Equal-time / commutator completion audit",
        "config": {
            "temperature_K": TEMPERATURE_K,
            "matsubara_index": MATSUBARA_INDEX,
            "omega_eV": float(omega_eV),
            "eta_eV": ETA_EV,
            "mesh_size": int(mesh_size),
            "q_base": [float(Q_BASE[0]), float(Q_BASE[1])],
            "q_scales": [float(item) for item in Q_SCALES],
        },
        "second_order_peierls_identity": {
            "status": second_order_status,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "samples": identity_samples,
        },
        "residual_decomposition": {
            "results_by_q_scale": rows,
            "slopes": slopes,
        },
        "diagnostic_status": {
            "second_order_identity_status": second_order_status,
            "direct_completion_status": direct_status,
            "next_step": next_step,
        },
        "boundary": {
            "no_residual_tuning": True,
            "no_bubble_formula_change": True,
            "no_main_response_change": True,
            "no_conductivity_reflection_casimir": True,
            "does_not_claim_ward_closure_unless_closed": True,
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
    rows = data["residual_decomposition"]["results_by_q_scale"]
    slopes = data["residual_decomposition"]["slopes"]
    status = data["diagnostic_status"]
    residual_table = _table(
        (
            "q_scale",
            "left_bubble_norm",
            "left_direct_norm",
            "left_total_norm",
            "left_missing_norm",
            "right_total_norm",
            "right_missing_norm",
        ),
        [
            (
                _fmt(float(row["q_scale"])),
                _fmt(float(row["left_bubble_norm"])),
                _fmt(float(row["left_direct_norm"])),
                _fmt(float(row["left_total_norm"])),
                _fmt(float(row["left_missing_norm"])),
                _fmt(float(row["right_total_norm"])),
                _fmt(float(row["right_missing_norm"])),
            )
            for row in rows
        ],
    )
    lt_table = _table(
        (
            "q_scale",
            "left_total_longitudinal_abs",
            "left_total_transverse_abs",
            "left_missing_longitudinal_abs",
            "left_missing_transverse_abs",
            "right_total_longitudinal_abs",
            "right_missing_longitudinal_abs",
        ),
        [
            (
                _fmt(float(row["q_scale"])),
                _fmt(float(row["left_total_longitudinal_abs"])),
                _fmt(float(row["left_total_transverse_abs"])),
                _fmt(float(row["left_missing_longitudinal_abs"])),
                _fmt(float(row["left_missing_transverse_abs"])),
                _fmt(float(row["right_total_longitudinal_abs"])),
                _fmt(float(row["right_missing_longitudinal_abs"])),
            )
            for row in rows
        ],
    )
    slope_table = _table(("quantity", "slope"), [(key, _fmt(float(value))) for key, value in slopes.items()])
    conclusion_table = _table(
        ("item", "analytic_status", "code_or_diagnostic_status", "conclusion"),
        [
            ("V_i vertex-level Ward identity", "derived in Stage 4.1B", "covered by existing vertex tests", "MATCH"),
            (
                "second-order Peierls identity q_i M_ij = Delta V_j",
                "derived from hopping formulas",
                data["second_order_peierls_identity"]["status"],
                data["second_order_peierls_identity"]["status"],
            ),
            ("direct derivative term D_ij=-<M_ij>", "derived", "included in direct component", "MATCH"),
            ("residual after bubble + direct", "nonzero allowed before ET audit", "Stage 4.10 result", "UNRESOLVED"),
            ("need for E_ET", "not proven zero", status["direct_completion_status"], "UNRESOLVED"),
        ],
    )
    return "\n\n".join(
        [
            "# Stage 4.10 Equal-time / commutator completion audit",
            "## Boundary\n\n- does not modify bubble factor\n- no residual tuning\n- no conductivity / reflection / Casimir\n- no Ward closure claim\n- only audits equal-time / commutator completion",
            "## Fixed response formula\n\n"
            "$V_i=\\delta H/\\delta A_i$, $M_{ij}=\\delta^2H/\\delta A_i\\delta A_j$.\n\n"
            "$J=(\\rho,-V_x,-V_y)$, $P=(\\rho,V_x,V_y)$.\n\n"
            "$\\Pi_{\\mu\\nu}=-\\langle J_\\mu P_\\nu\\rangle+\\langle\\delta J_\\mu/\\delta a_\\nu\\rangle+E_{\\mu\\nu}^{ET}$.\n\n"
            "Current code includes $D_{ij}=-\\langle M_{ij}\\rangle$ but no explicit $E^{ET}$ term.",
            "## Ward identity output directory\n\n`validation/outputs/response/ward_identity/`",
            "## Second-order Peierls identity check\n\n"
            f"status = `{data['second_order_peierls_identity']['status']}`; "
            f"max_abs_error = {_fmt(float(data['second_order_peierls_identity']['max_abs_error']))}; "
            f"max_rel_error = {_fmt(float(data['second_order_peierls_identity']['max_rel_error']))}.",
            "## Bubble/direct/total/missing residual decomposition\n\n"
            "$R^{missing}=-(R^{bubble}+R^{direct})$.\n\n"
            + residual_table,
            "## Longitudinal/transverse decomposition\n\n" + lt_table,
            "## q-scaling slopes\n\n" + slope_table,
            "## Conclusion table\n\n" + conclusion_table,
            "## Diagnostic status\n\n"
            f"- second_order_identity_status = `{status['second_order_identity_status']}`\n"
            f"- direct_completion_status = `{status['direct_completion_status']}`\n"
            f"- direct term numerically closes Ward residual: `{status['direct_completion_status'] == 'DIRECT_TERM_NUMERICALLY_CLOSES_WARD'}`\n"
            f"- remaining residual order: `{status['direct_completion_status']}`",
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
