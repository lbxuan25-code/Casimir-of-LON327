#!/usr/bin/env python3
"""Stage 5.1b bilayer sheet conductivity convention validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.electrodynamics.conventions import (  # noqa: E402
    bilayer_sheet_conductivity_convention_metadata,
    spatial_response_to_bilayer_sheet_conductivity_model,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
JSON_OUTPUT = OUTPUT_DIR / "stage5_1b_bilayer_sheet_conductivity_convention.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_1b_bilayer_sheet_conductivity_convention.md"

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


def synthetic_response() -> tuple[np.ndarray, float]:
    response = np.zeros((3, 3), dtype=complex)
    response[1, 1] = -0.3
    response[2, 2] = -0.4
    response[1, 2] = 0.01
    response[2, 1] = 0.02
    return response, 0.02


def run_validation(*, quick: bool = False) -> dict[str, Any]:
    response, omega_eV = synthetic_response()
    sigma = spatial_response_to_bilayer_sheet_conductivity_model(response, omega_eV)
    expected = np.array([[15.0, -0.5], [-1.0, 20.0]], dtype=complex)
    synthetic_pass = bool(np.allclose(sigma, expected))
    metadata = bilayer_sheet_conductivity_convention_metadata()
    return {
        "stage": "Stage 5.1b",
        "purpose": "Fix response-to-bilayer-sheet-conductivity convention",
        "boundary": dict(BOUNDARY),
        "selected_convention": {
            "electric_field_relation": metadata["electric_field_relation"],
            "response_to_conductivity_formula": metadata["model_formula"],
            "frequency_variable": "omega_eV = hbar xi",
            "normalization": metadata["normalization"],
            "si_scaling_applied": metadata["si_scaling_applied"],
            "bulk_3d_conductivity": False,
            "single_layer_conductivity": False,
        },
        "synthetic_check": {
            "quick_mode": bool(quick),
            "input_pi_spatial": response[1:3, 1:3],
            "omega_eV": omega_eV,
            "output_sigma_model": sigma,
            "expected_sigma_model": expected,
            "diagonal_positive_for_negative_pi": bool(sigma[0, 0].real > 0.0 and sigma[1, 1].real > 0.0),
            "status": "PASS" if synthetic_pass else "FAIL",
        },
        "diagnostic_status": {
            "conductivity_convention_status": "CONVENTION_FIXED",
            "normalization_status": "BILAYER_SHEET_MODEL_FIXED",
            "recommended_next_action": (
                "Proceed to Stage 5.2 numerical conductivity sanity scan; "
                "still do not enter reflection/Casimir."
            ),
        },
    }


def _format_matrix(matrix: Any) -> str:
    rows = []
    for row in matrix:
        rows.append(
            "["
            + ", ".join(
                f"{complex(value).real:.6g}{complex(value).imag:+.6g}j"
                if abs(complex(value).imag) > 0.0
                else f"{complex(value).real:.6g}"
                for value in row
            )
            + "]"
        )
    return "[" + ", ".join(rows) + "]"


def render_markdown(data: dict[str, Any]) -> str:
    selected = data["selected_convention"]
    check = data["synthetic_check"]
    boundary_lines = "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items())
    selected_lines = "\n".join(f"- {key}: {value}" for key, value in selected.items())
    diagnostic_lines = "\n".join(f"- {key}: {value}" for key, value in data["diagnostic_status"].items())
    return "\n\n".join(
        [
            "# Stage 5.1b Bilayer sheet conductivity convention",
            "## Boundary\n\n" + boundary_lines,
            "## Selected convention\n\n" + selected_lines,
            "## Analytic derivation\n\n"
            "With real-time convention $f(t)\\sim e^{-i\\omega t}$, "
            "$E_j(\\omega)=i\\omega A_j(\\omega)$ in transverse/optical gauge. "
            "Analytic continuation $\\omega\\to i\\xi$ gives $E_j(i\\xi)=-\\xi A_j(i\\xi)$. "
            "Since $j_i=\\Pi_{ij}A_j=\\sigma_{ij}E_j$, the model convention is "
            "$\\sigma^{model}_{ij}(i\\Omega)=-\\Pi_{ij}(i\\Omega)/\\Omega_{eV}$.",
            "## Bilayer-normalized 2D sheet interpretation\n\n"
            "The response is computed from the full bilayer Hamiltonian, including interlayer hopping, "
            "hybridization, bonding/antibonding structure, and bilayer pairing information. "
            "The output is the in-plane sheet response of one bilayer unit, not a 3D bulk conductivity "
            "and not a single-layer conductivity. Final SI sheet scaling is not applied here.",
            "## Synthetic check\n\n"
            f"- input_pi_spatial: {_format_matrix(check['input_pi_spatial'])}\n"
            f"- omega_eV: {check['omega_eV']}\n"
            f"- output_sigma_model: {_format_matrix(check['output_sigma_model'])}\n"
            f"- diagonal_positive_for_negative_pi: {check['diagonal_positive_for_negative_pi']}\n"
            f"- status: {check['status']}",
            "## Diagnostic decision\n\n" + diagnostic_lines,
            "## Next step\n\n"
            "Proceed to Stage 5.2 numerical conductivity sanity scan; still do not enter reflection/Casimir.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_validation(quick=args.quick)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
