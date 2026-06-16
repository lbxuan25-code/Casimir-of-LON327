#!/usr/bin/env python3
"""Stage 5.4a synthetic validation for conductivity unit conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.conductivity_units import (  # noqa: E402
    SheetConductivityUnitConvention,
    conductivity_unit_conversion_metadata,
    e2_over_hbar_siemens,
    four_pi_alpha,
    model_to_dimensionless_sheet_conductivity,
    model_to_si_sheet_conductivity,
    sheet_geometry_factor_tensor,
    si_sheet_to_dimensionless_conductivity,
    vacuum_impedance_ohm,
    z0_e2_over_hbar,
)
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
JSON_OUTPUT = OUTPUT_DIR / "stage5_4a_conductivity_unit_conversion.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_4a_conductivity_unit_conversion.md"

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
    "no_heavy_response_run": True,
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


def run_validation(
    convention: SheetConductivityUnitConvention,
    *,
    material_lattice_config: Any = LNO327_THIN_FILM_SLAO_IN_PLANE,
) -> dict[str, Any]:
    sigma_model = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    square_si = model_to_si_sheet_conductivity(sigma_model, convention)
    square_tilde = model_to_dimensionless_sheet_conductivity(sigma_model, convention)
    square_si_pass = bool(np.allclose(square_si, e2_over_hbar_siemens() * sigma_model))
    square_tilde_pass = bool(np.allclose(square_tilde, four_pi_alpha() * sigma_model, rtol=1e-10, atol=1e-15))

    a = convention.lattice_a_y_m
    rectangular = SheetConductivityUnitConvention(lattice_a_x_m=2.0 * a, lattice_a_y_m=a, unit_cell_area_m2=2.0 * a * a)
    rectangular_tensor = sheet_geometry_factor_tensor(rectangular)
    rectangular_pass = bool(np.allclose(rectangular_tensor, np.array([[2.0, 1.0], [1.0, 0.5]], dtype=float)))

    prefactor_error = abs(z0_e2_over_hbar() - four_pi_alpha())
    prefactor_pass = bool(prefactor_error / max(abs(four_pi_alpha()), 1e-300) < 1e-10)
    metadata = conductivity_unit_conversion_metadata(convention)
    checks = {
        "square_lattice_identity": "PASS" if square_si_pass and square_tilde_pass else "FAIL",
        "rectangular_lattice_geometry": "PASS" if rectangular_pass else "FAIL",
        "dimensionless_prefactor_consistency": "PASS" if prefactor_pass else "FAIL",
    }
    all_pass = all(value == "PASS" for value in checks.values())
    return {
        "stage": "Stage 5.4a",
        "purpose": "Conductivity SI sheet unit conversion and dimensionless sheet conductivity validation",
        "boundary": dict(BOUNDARY),
        "conductivity_convention": {
            "input": "sigma_model_ij",
            "input_formula": "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV",
            "output_si": "sigma_SI_sheet_ij",
            "output_dimensionless": "sigma_tilde_ij = Z0 * sigma_SI_sheet_ij",
            "normalization": convention.normalization,
            "bulk_3d_conductivity": False,
            "single_layer_conductivity": False,
        },
        "unit_conversion": {
            "formula_model_to_si": metadata["formula_model_to_si"],
            "formula_si_to_dimensionless": metadata["formula_si_to_dimensionless"],
            "formula_model_to_dimensionless": metadata["formula_model_to_dimensionless"],
            "e2_over_hbar_S": e2_over_hbar_siemens(),
            "vacuum_impedance_ohm": vacuum_impedance_ohm(),
            "z0_e2_over_hbar": z0_e2_over_hbar(),
            "four_pi_alpha": four_pi_alpha(),
            "z0_e2_over_hbar_minus_four_pi_alpha_abs": prefactor_error,
        },
        "geometry": {
            "material_lattice_config": material_lattice_config.name,
            "is_placeholder": bool(material_lattice_config.is_placeholder),
            "source_note": material_lattice_config.source_note,
            "lattice_a_x_m": convention.lattice_a_x_m,
            "lattice_a_y_m": convention.lattice_a_y_m,
            "unit_cell_area_m2": convention.unit_cell_area_m2,
            "geometry_tensor": sheet_geometry_factor_tensor(convention),
            "lattice_note": "3.754 Angstrom is the current thin-film working default; sample-specific constants may override this config.",
        },
        "synthetic_checks": checks,
        "synthetic_outputs": {
            "sigma_model": sigma_model,
            "sigma_SI_sheet": square_si,
            "sigma_tilde": square_tilde,
            "rectangular_geometry_tensor": rectangular_tensor,
        },
        "diagnostic_status": {
            "stage5_4a_status": "STAGE5_4A_CONDUCTIVITY_UNIT_CONVERSION_PASSED" if all_pass else "STAGE5_4A_CONDUCTIVITY_UNIT_CONVERSION_FAILED",
            "recommended_next_action": (
                "Proceed to Stage 5.4b to convert validated model conductivity outputs to SI sheet "
                "and sigma_tilde data; still do not run reflection/Casimir."
                if all_pass
                else "Fix unit conversion helpers before Stage 5.4b."
            ),
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
            "# Stage 5.4a 电导单位转换验证",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input conductivity convention\n\n" + _table(("quantity", "value"), list(data["conductivity_convention"].items())),
            "## 3. Analytic unit-chain derivation\n\n"
            "$A_i^{model}=ea_iA_i^{SI}/\\hbar$，因此 "
            "$\\sigma^{SI,sheet}_{ij}=(e^2/\\hbar)(a_i a_j/A_{cell})\\sigma^{model}_{ij}$。",
            "## 4. Geometry tensor\n\n" + _table(("quantity", "value"), list(data["geometry"].items())),
            "## 5. SI sheet conductivity conversion\n\n"
            "这是 one-bilayer sheet response，不是 bulk 3D conductivity，也不是 single-layer conductivity。",
            "## 6. Dimensionless sheet conductivity\n\n"
            "$\\tilde\\sigma_{ij}=Z_0\\sigma^{SI,sheet}_{ij}$。$\\tilde\\sigma$ 不是新的材料模型参数，而是 dimensionless sheet conductivity / dimensionless sheet admittance；不再使用 $g$ 作为符号。",
            "## 7. Synthetic square-lattice check\n\n"
            f"status: {data['synthetic_checks']['square_lattice_identity']}",
            "## 8. Synthetic rectangular-lattice check\n\n"
            f"status: {data['synthetic_checks']['rectangular_lattice_geometry']}",
            "## 9. Dimensionless prefactor check\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("Z0 e^2/hbar", data["unit_conversion"]["z0_e2_over_hbar"]),
                    ("4 pi alpha", data["unit_conversion"]["four_pi_alpha"]),
                    ("abs error", data["unit_conversion"]["z0_e2_over_hbar_minus_four_pi_alpha_abs"]),
                    ("status", data["synthetic_checks"]["dimensionless_prefactor_consistency"]),
                ],
            ),
            "## 10. Diagnostic decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 11. Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " 本阶段仍未进入 reflection/Casimir，也尚未做真实材料 lattice constants 的最终配置管理。",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice-a-x-m", type=float, default=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m)
    parser.add_argument("--lattice-a-y-m", type=float, default=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m)
    parser.add_argument("--unit-cell-area-m2", type=float, default=LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convention = SheetConductivityUnitConvention(
        lattice_a_x_m=args.lattice_a_x_m,
        lattice_a_y_m=args.lattice_a_y_m,
        unit_cell_area_m2=args.unit_cell_area_m2,
    )
    data = run_validation(convention)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
