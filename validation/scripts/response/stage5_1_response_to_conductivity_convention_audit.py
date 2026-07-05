#!/usr/bin/env python3
"""Stage 5.1 response-to-conductivity convention audit.

Diagnostic-only.  This script audits code paths and candidate conventions for
converting the normal-state physical response spatial block Pi_ij to
conductivity sigma_ij on the imaginary axis.  It does not modify the response
formula, conductivity production path, reflection, or Casimir code.
"""

from __future__ import annotations

import argparse
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

from lno327 import KuboConfig, bosonic_matsubara_energy_eV  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import integrate_physical_components_on_points  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
JSON_OUTPUT = OUTPUT_DIR / "stage5_1_response_to_conductivity_convention_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_1_response_to_conductivity_convention_audit.md"

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_reflection_casimir": True,
    "not_casimir_ready_claim": True,
}

KEYWORDS = (
    "conductivity",
    "sigma",
    "optical",
    "Kubo",
    "response",
    "current",
    "Pi",
    "Matsubara",
    "imag_axis",
    "output_si",
    "SI",
    "eV",
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        return {"real": float(np.real(value)), "imag": float(np.imag(value)), "abs": float(abs(value))}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _repo_text_files() -> list[Path]:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "validation" / "scripts", ROOT / "docs", ROOT / "tests")
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".txt"}:
                files.append(path)
    return files


def audit_existing_code() -> dict[str, list[dict[str, Any]]]:
    related: list[dict[str, Any]] = []
    helpers: list[dict[str, Any]] = []
    si_helpers: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    for path in sorted(_repo_text_files()):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [keyword for keyword in KEYWORDS if keyword.lower() in text.lower() or keyword.lower() in rel.lower()]
        if not hits:
            continue
        entry = {
            "path": rel,
            "keyword_hits": sorted(set(hits)),
            "has_conductivity_function": "conductivity" in text.lower() and "def " in text,
            "has_response_to_conductivity_conversion": "model_response_to_sheet_conductivity" in text
            or "sigma_like_response" in text
            or "response_to_conductivity" in text,
            "has_si_conversion": "E2_OVER_HBAR" in text or "output_si" in text or "SIGMA0" in text,
            "has_imag_axis_output": "imag_axis" in text or "sigma(i" in text,
            "mentions_downstream_consumer": "reflection" in text.lower() or "casimir" in text.lower(),
        }
        related.append(entry)
        if entry["has_conductivity_function"]:
            helpers.append(entry)
        if entry["has_si_conversion"]:
            si_helpers.append(entry)
        if entry["mentions_downstream_consumer"]:
            consumers.append(entry)
    return {
        "conductivity_related_files": related,
        "existing_conductivity_helpers": helpers,
        "existing_si_conversion_helpers": si_helpers,
        "reflection_or_casimir_consumers": consumers,
    }


def spatial_response_to_conductivity(response: np.ndarray, omega_eV: float, convention: str) -> np.ndarray:
    """Diagnostic wrapper converting Pi_ij to sigma_ij for candidate conventions."""

    matrix = np.asarray(response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    if omega_eV == 0.0:
        raise ValueError("omega_eV must be nonzero for response-to-conductivity conversion")
    spatial = matrix[1:3, 1:3]
    if convention == "A_plus_xi":
        return spatial / omega_eV
    if convention == "B_minus_xi":
        return -spatial / omega_eV
    if convention == "C_iOmega":
        return spatial / (1j * omega_eV)
    raise ValueError("unknown convention")


def candidate_conventions() -> list[dict[str, str]]:
    return [
        {
            "id": "A_plus_xi",
            "electric_field_relation": "E_j(i xi)=+xi A_j(i xi)",
            "formula": "sigma_ij(i xi)=Pi_ij(i xi)/xi",
            "status": "candidate",
        },
        {
            "id": "B_minus_xi",
            "electric_field_relation": "E_j(i xi)=-xi A_j(i xi)",
            "formula": "sigma_ij(i xi)=-Pi_ij(i xi)/xi",
            "status": "candidate",
        },
        {
            "id": "C_iOmega",
            "electric_field_relation": "E_j(iOmega)=iOmega A_j(iOmega)",
            "formula": "sigma_ij(iOmega)=Pi_ij(iOmega)/(iOmega)",
            "status": "candidate",
        },
    ]


def unit_audit_table() -> list[dict[str, str]]:
    return [
        {
            "quantity": "finite-q physical response",
            "symbol": "Pi_ij",
            "code_variable": "response[1:3,1:3]",
            "current_unit": "dimensionless model response from normalized BZ weights",
            "target_unit": "conductivity kernel before SI sheet scaling",
            "status": "INFERRED",
        },
        {
            "quantity": "Matsubara energy",
            "symbol": "hbar xi_n",
            "code_variable": "omega_eV",
            "current_unit": "eV",
            "target_unit": "eV for diagnostic conversion; rad/s for SI electrodynamics if needed",
            "status": "KNOWN",
        },
        {
            "quantity": "model conductivity",
            "symbol": "sigma_model",
            "code_variable": "spatial_response_to_conductivity(...)",
            "current_unit": "Pi_model / eV for candidates A/B or Pi_model/(i eV) for C",
            "target_unit": "model sheet conductivity convention",
            "status": "AMBIGUOUS",
        },
        {
            "quantity": "SI sheet conductivity scaling",
            "symbol": "sigma_sheet",
            "code_variable": "model_response_to_sheet_conductivity",
            "current_unit": "model response",
            "target_unit": "S",
            "status": "INFERRED",
        },
        {
            "quantity": "2D versus 3D conductivity",
            "symbol": "sigma_2D / sigma_3D",
            "code_variable": "SheetConductivityConvention",
            "current_unit": "2D sheet convention",
            "target_unit": "reflection input needs sheet conductivity or dimensionless sheet normalization",
            "status": "KNOWN",
        },
        {
            "quantity": "explicit lattice geometry",
            "symbol": "a_parallel, layer spacing",
            "code_variable": "lattice_constant_m, unit_cell_area_m2",
            "current_unit": "optional and inactive by default",
            "target_unit": "needed only for future explicit 3D/bulk normalization",
            "status": "MISSING",
        },
    ]


def _matrix_metrics(matrix: np.ndarray) -> dict[str, Any]:
    diagonal_scale = 0.5 * (abs(matrix[0, 0]) + abs(matrix[1, 1]))
    offdiag_norm = float(np.linalg.norm([matrix[0, 1], matrix[1, 0]]))
    return {
        "xx": matrix[0, 0],
        "yy": matrix[1, 1],
        "xy": matrix[0, 1],
        "yx": matrix[1, 0],
        "diagonal_scale": float(diagonal_scale),
        "offdiag_norm": offdiag_norm,
        "relative_offdiag": float(offdiag_norm / max(diagonal_scale, 1e-300)),
        "finite": bool(np.all(np.isfinite(matrix))),
    }


def lightweight_sanity_check(*, quick: bool) -> dict[str, Any]:
    temperature_K = 30.0
    matsubara_indices = [1, 2, 4]
    q = np.array([0.02, 0.013], dtype=float)
    coarse_grid = 8 if quick else 32
    adaptive_level = 1 if quick else 4
    gauss_order = 2 if quick else 5
    fermi_window_eV = 0.05
    rows: list[dict[str, Any]] = []
    for matsubara_index in matsubara_indices:
        omega_eV = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
        config = KuboConfig.from_kelvin(
            omega_eV=omega_eV,
            temperature_K=temperature_K,
            eta_eV=1e-10,
            output_si=False,
        )
        cells, refined_count, _flagged = build_adaptive_cells(
            q,
            coarse_grid=coarse_grid,
            refinement_level=adaptive_level,
            fermi_window_eV=fermi_window_eV,
            fermi_level_eV=config.fermi_level_eV,
        )
        points, weights = quadrature_points_for_cells(cells, gauss_order)
        response = integrate_physical_components_on_points(points, weights, q, config)["total"]
        spatial = response[1:3, 1:3]
        converted = {
            convention["id"]: _matrix_metrics(
                spatial_response_to_conductivity(response, omega_eV, convention["id"])
            )
            for convention in candidate_conventions()
        }
        rows.append(
            {
                "temperature_K": temperature_K,
                "matsubara_index": matsubara_index,
                "omega_eV": omega_eV,
                "q_model": [float(q[0]), float(q[1])],
                "coarse_grid": coarse_grid,
                "adaptive_level": adaptive_level,
                "gauss_order": gauss_order,
                "fermi_window_eV": fermi_window_eV,
                "num_cells_total": len(cells),
                "num_cells_refined": refined_count,
                "num_quadrature_points": len(points),
                "spatial_response_metrics": _matrix_metrics(spatial),
                "candidate_sigma_metrics": converted,
            }
        )
    diag_values = [float(row["candidate_sigma_metrics"]["A_plus_xi"]["diagonal_scale"]) for row in rows]
    smooth = all(np.isfinite(diag_values)) and max(diag_values) / max(min(diag_values), 1e-300) < 1e6
    return {
        "quick_mode": bool(quick),
        "rows": rows,
        "all_spatial_blocks_finite": all(row["spatial_response_metrics"]["finite"] for row in rows),
        "all_candidate_sigmas_finite": all(
            metrics["finite"]
            for row in rows
            for metrics in row["candidate_sigma_metrics"].values()
        ),
        "diagonal_scale_smooth_order_of_magnitude": bool(smooth),
    }


def selected_convention() -> dict[str, str]:
    return {
        "status": "AMBIGUOUS",
        "formula": "CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE",
        "reason": (
            "Existing local Kubo code computes sigma(i xi) directly, and response_conventions "
            "documents model-response-to-sheet normalization, but the finite-q physical "
            "Pi_ij to sigma_ij Euclidean E/A sign is not uniquely fixed by code alone."
        ),
    }


def diagnostic_status(selected: dict[str, str]) -> dict[str, str]:
    convention_status = (
        "CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE"
        if selected["status"] == "AMBIGUOUS"
        else "CONDUCTIVITY_CONVENTION_SELECTED"
    )
    unit_status = "UNIT_CHAIN_AMBIGUOUS"
    next_action = (
        "Decide the Euclidean E/A convention and sheet-vs-bulk normalization before "
        "Stage 5.2 numerical conductivity sanity."
    )
    return {
        "conductivity_convention_status": convention_status,
        "unit_status": unit_status,
        "recommended_next_action": "NOT_READY_NEEDS_CONVENTION_DECISION: " + next_action,
    }


def run_audit(*, quick: bool = False) -> dict[str, Any]:
    selected = selected_convention()
    return {
        "stage": "Stage 5.1",
        "purpose": "Response-to-conductivity convention audit",
        "boundary": dict(BOUNDARY),
        "existing_code_audit": audit_existing_code(),
        "response_convention": {
            "observable_J": ["rho", "-Vx", "-Vy"],
            "source_P": ["rho", "Vx", "Vy"],
            "spatial_block_interpretation": "delta<j_i>/delta A_j",
            "response_kernel": "Pi_munu = delta< J_mu >/delta a_nu",
            "source_fields": ["phi", "A_x", "A_y"],
        },
        "candidate_conductivity_conventions": candidate_conventions(),
        "selected_convention": selected,
        "unit_audit": unit_audit_table(),
        "lightweight_sanity_check": lightweight_sanity_check(quick=quick),
        "diagnostic_status": diagnostic_status(selected),
    }


def _fmt_complex(value: Any) -> str:
    if isinstance(value, dict) and {"real", "imag"} <= set(value):
        return f"{float(value['real']):.6e}{float(value['imag']):+.6e}j"
    if isinstance(value, complex | np.complexfloating):
        return f"{value.real:.6e}{value.imag:+.6e}j"
    return str(value)


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    existing = data["existing_code_audit"]
    unit_rows = [
        (
            row["quantity"],
            row["symbol"],
            row["code_variable"],
            row["current_unit"],
            row["target_unit"],
            row["status"],
        )
        for row in data["unit_audit"]
    ]
    sanity_rows = [
        (
            row["matsubara_index"],
            f"{float(row['omega_eV']):.6e}",
            _fmt_complex(to_jsonable(row["spatial_response_metrics"]["xx"])),
            _fmt_complex(to_jsonable(row["spatial_response_metrics"]["yy"])),
            f"{float(row['spatial_response_metrics']['relative_offdiag']):.6e}",
            row["num_quadrature_points"],
        )
        for row in data["lightweight_sanity_check"]["rows"]
    ]
    return "\n\n".join(
        [
            "# Stage 5.1 Response-to-conductivity convention audit",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no source/observable change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no reflection / Casimir\n"
            "- no Casimir-ready claim",
            "## Existing conductivity-related code paths\n\n"
            + _table(
                ("category", "count"),
                [
                    ("conductivity_related_files", len(existing["conductivity_related_files"])),
                    ("existing_conductivity_helpers", len(existing["existing_conductivity_helpers"])),
                    ("existing_si_conversion_helpers", len(existing["existing_si_conversion_helpers"])),
                    ("reflection_or_casimir_consumers", len(existing["reflection_or_casimir_consumers"])),
                ],
            ),
            "## Response convention\n\n"
            "$$\\Pi_{\\mu\\nu}=\\frac{\\delta\\langle J_\\mu\\rangle}{\\delta a_\\nu},\\quad a_\\nu=(\\phi,A_x,A_y).$$\n\n"
            "$$J=(\\rho,j_x,j_y)=(\\rho,-V_x,-V_y),\\qquad P=(\\rho,V_x,V_y).$$\n\n"
            "The spatial block is interpreted as $\\Pi_{ij}=\\delta\\langle j_i\\rangle/\\delta A_j$.",
            "## Candidate conductivity conventions\n\n"
            + _table(
                ("id", "E/A relation", "formula"),
                [
                    (row["id"], row["electric_field_relation"], row["formula"])
                    for row in data["candidate_conductivity_conventions"]
                ],
            ),
            "## Selected or ambiguous convention\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("status", data["selected_convention"]["status"]),
                    ("formula", data["selected_convention"]["formula"]),
                    ("reason", data["selected_convention"]["reason"]),
                ],
            ),
            "## Unit audit\n\n" + _table(("quantity", "symbol", "code variable", "current unit", "target unit", "status"), unit_rows),
            "## Lightweight sanity check\n\n"
            + _table(("n", "omega_eV", "Pi_xx", "Pi_yy", "relative_offdiag", "quad points"), sanity_rows),
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("conductivity_convention_status", data["diagnostic_status"]["conductivity_convention_status"]),
                    ("unit_status", data["diagnostic_status"]["unit_status"]),
                    ("recommended_next_action", data["diagnostic_status"]["recommended_next_action"]),
                ],
            ),
            "## Recommended next step\n\nConfirm the Euclidean electric-field/vector-potential convention and the sheet-vs-bulk normalization before Stage 5.2 numerical conductivity sanity. This audit is not reflection/Casimir input.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_audit(quick=args.quick)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
