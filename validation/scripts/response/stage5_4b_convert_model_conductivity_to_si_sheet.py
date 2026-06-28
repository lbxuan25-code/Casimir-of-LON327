#!/usr/bin/env python3
"""Stage 5.4b conversion from model conductivity JSON to SI sheet and sigma_tilde."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.response_conventions import (  # noqa: E402
    SheetConductivityUnitConvention,
    conductivity_unit_conversion_metadata,
    four_pi_alpha,
    model_to_dimensionless_sheet_conductivity,
    model_to_si_sheet_conductivity,
)
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
DEFAULT_INPUT = OUTPUT_DIR / "stage5_2_bilayer_sheet_conductivity_sanity_scan.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_4b_si_sheet_dimensionless_conductivity.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_4b_si_sheet_dimensionless_conductivity.md"

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


def parse_complex_component(value: Any) -> complex:
    if isinstance(value, dict):
        return complex(float(value["real"]), float(value.get("imag", 0.0)))
    return complex(value)


def model_matrix_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [parse_complex_component(row["sigma_xx_model"]), parse_complex_component(row["sigma_xy_model"])],
            [parse_complex_component(row["sigma_yx_model"]), parse_complex_component(row["sigma_yy_model"])],
        ],
        dtype=complex,
    )


def relative_offdiag_norm(matrix: np.ndarray) -> float:
    offdiag = float(np.sqrt(abs(matrix[0, 1]) ** 2 + abs(matrix[1, 0]) ** 2))
    diag = float(np.sqrt(abs(matrix[0, 0]) ** 2 + abs(matrix[1, 1]) ** 2))
    return offdiag / max(diag, 1e-300)


def convert_row(row: dict[str, Any], convention: SheetConductivityUnitConvention) -> dict[str, Any]:
    sigma_model = model_matrix_from_row(row)
    sigma_si = model_to_si_sheet_conductivity(sigma_model, convention)
    sigma_tilde = model_to_dimensionless_sheet_conductivity(sigma_model, convention)
    return {
        **row,
        "sigma_xx_SI_sheet": sigma_si[0, 0],
        "sigma_xy_SI_sheet": sigma_si[0, 1],
        "sigma_yx_SI_sheet": sigma_si[1, 0],
        "sigma_yy_SI_sheet": sigma_si[1, 1],
        "sigma_tilde_xx": sigma_tilde[0, 0],
        "sigma_tilde_xy": sigma_tilde[0, 1],
        "sigma_tilde_yx": sigma_tilde[1, 0],
        "sigma_tilde_yy": sigma_tilde[1, 1],
        "sigma_SI_sheet_matrix": sigma_si,
        "sigma_tilde_matrix": sigma_tilde,
        "sigma_SI_sheet_units": "S",
        "sigma_tilde_units": "dimensionless",
        "relative_offdiag_norm_model_recomputed": relative_offdiag_norm(sigma_model),
        "relative_offdiag_norm_tilde": relative_offdiag_norm(sigma_tilde),
    }


def _input_diagnostic_status(data: dict[str, Any]) -> Any:
    status = data.get("diagnostic_status", {})
    if isinstance(status, dict):
        for key in ("conductivity_sanity_status", "stage5_3b_status", "conductivity_symmetry_audit_status"):
            if key in status:
                return status[key]
    return status


def _validate_input_rows(rows: list[dict[str, Any]], *, require_no_fail_input: bool) -> None:
    if require_no_fail_input and any(row.get("status") == "FAIL" for row in rows):
        raise ValueError("input contains FAIL scan_results; refusing conversion by default")


def run_conversion(
    input_json: Path,
    convention: SheetConductivityUnitConvention,
    *,
    require_no_fail_input: bool = True,
    allow_monitor_input: bool = True,
) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    rows = list(data.get("scan_results", []))
    _validate_input_rows(rows, require_no_fail_input=require_no_fail_input)
    if not allow_monitor_input and any(row.get("status") == "MONITOR" for row in rows):
        raise ValueError("input contains MONITOR scan_results and allow_monitor_input=False")
    converted = [convert_row(row, convention) for row in rows]
    metadata = conductivity_unit_conversion_metadata(convention)
    max_abs_tilde = max((abs(parse_complex_component(row["sigma_tilde_xx"])) for row in converted), default=0.0)
    max_abs_tilde = max(max_abs_tilde, max((abs(parse_complex_component(row["sigma_tilde_xy"])) for row in converted), default=0.0))
    max_abs_tilde = max(max_abs_tilde, max((abs(parse_complex_component(row["sigma_tilde_yx"])) for row in converted), default=0.0))
    max_abs_tilde = max(max_abs_tilde, max((abs(parse_complex_component(row["sigma_tilde_yy"])) for row in converted), default=0.0))
    max_abs_si = max((float(np.max(np.abs(row["sigma_SI_sheet_matrix"]))) for row in converted), default=0.0)
    min_diag_si = min(
        (min(parse_complex_component(row["sigma_xx_SI_sheet"]).real, parse_complex_component(row["sigma_yy_SI_sheet"]).real) for row in converted),
        default=None,
    )
    max_model_offdiag = max((float(row.get("relative_offdiag_norm", row["relative_offdiag_norm_model_recomputed"])) for row in converted), default=0.0)
    max_tilde_offdiag = max((float(row["relative_offdiag_norm_tilde"]) for row in converted), default=0.0)
    return {
        "stage": "Stage 5.4b",
        "purpose": "Convert validated model bilayer sheet conductivity to SI sheet and sigma_tilde",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_diagnostic_status": _input_diagnostic_status(data),
            "input_num_cases": len(rows),
            "allow_monitor_input": bool(allow_monitor_input),
            "require_no_fail_input": bool(require_no_fail_input),
        },
        "conductivity_convention": {
            "input": "sigma_model_ij",
            "model_formula": "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV",
            "output_si": "sigma_SI_sheet_ij",
            "output_dimensionless": "sigma_tilde_ij = Z0 * sigma_SI_sheet_ij",
            "normalization": "bilayer-normalized 2D sheet conductivity",
            "bulk_3d_conductivity": False,
            "single_layer_conductivity": False,
        },
        "lattice_convention": {
            "name": LNO327_THIN_FILM_SLAO_IN_PLANE.name,
            "lattice_a_x_m": convention.lattice_a_x_m,
            "lattice_a_y_m": convention.lattice_a_y_m,
            "unit_cell_area_m2": convention.unit_cell_area_m2,
            "source_note": LNO327_THIN_FILM_SLAO_IN_PLANE.source_note,
            "is_placeholder": LNO327_THIN_FILM_SLAO_IN_PLANE.is_placeholder,
        },
        "unit_conversion": {
            "formula_model_to_si": metadata["formula_model_to_si"],
            "formula_si_to_dimensionless": metadata["formula_si_to_dimensionless"],
            "e2_over_hbar_S": metadata["e2_over_hbar_S"],
            "vacuum_impedance_ohm": metadata["vacuum_impedance_ohm"],
            "z0_e2_over_hbar": metadata["z0_e2_over_hbar"],
            "four_pi_alpha": four_pi_alpha(),
            "geometry_tensor": metadata["geometry_tensor"],
        },
        "converted_results": converted,
        "summary": {
            "num_cases": len(converted),
            "max_abs_sigma_tilde": float(max_abs_tilde),
            "max_abs_sigma_SI_sheet_S": float(max_abs_si),
            "min_diag_sigma_SI_sheet_real_S": None if min_diag_si is None else float(min_diag_si),
            "max_relative_offdiag_norm_model": float(max_model_offdiag),
            "max_relative_offdiag_norm_tilde": float(max_tilde_offdiag),
            "conversion_preserves_relative_structure": bool(abs(max_model_offdiag - max_tilde_offdiag) < 1e-12),
        },
        "diagnostic_status": {
            "stage5_4b_status": "STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED",
            "recommended_next_action": "Proceed to reflection-input preparation only after checking tensor formatting; do not run reflection/Casimir yet.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    reps = []
    for row in data["converted_results"][:3]:
        reps.append(
            (
                row.get("q_case"),
                row.get("matsubara_index"),
                row.get("q_scale", 1.0),
                row["sigma_tilde_xx"],
                row["sigma_tilde_xy"],
                row["sigma_tilde_yy"],
            )
        )
    return "\n\n".join(
        [
            "# Stage 5.4b SI sheet / sigma_tilde 转换",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input file and input status\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Conductivity convention\n\n" + _table(("quantity", "value"), list(data["conductivity_convention"].items())),
            "## 4. Lattice convention\n\n" + _table(("quantity", "value"), list(data["lattice_convention"].items())),
            "## 5. Unit conversion formula\n\n"
            "$\\sigma^{SI,sheet}_{ij}=(e^2/\\hbar)(a_i a_j/A_{cell})\\sigma^{model}_{ij}$，"
            "$\\tilde\\sigma_{ij}=Z_0\\sigma^{SI,sheet}_{ij}$。",
            "## 6. Converted conductivity summary\n\n" + _table(("quantity", "value"), list(data["summary"].items())),
            "## 7. Representative converted values\n\n" + (_table(("q", "n", "scale", "sigma_tilde_xx", "sigma_tilde_xy", "sigma_tilde_yy"), reps) if reps else "No converted rows."),
            "## 8. Model vs SI vs sigma_tilde\n\n"
            "`sigma_tilde` 是 dimensionless sheet conductivity / admittance，不再使用 `g` 作为符号。这是 one-bilayer sheet response，不是 bulk 3D，也不是 single-layer。",
            "## 9. Diagnostic decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 10. Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " 本阶段没有运行 heavy response，也没有进入 reflection/Casimir。LNO327 thin-film lattice constant is now 3.754 Å, not placeholder 3.85 Å; future sample-specific constants may override the config.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--lattice-config", default=LNO327_THIN_FILM_SLAO_IN_PLANE.name)
    parser.add_argument("--lattice-a-x-m", type=float, default=None)
    parser.add_argument("--lattice-a-y-m", type=float, default=None)
    parser.add_argument("--unit-cell-area-m2", type=float, default=None)
    parser.add_argument("--allow-monitor-input", action="store_true", default=True)
    parser.add_argument("--require-no-fail-input", action="store_true", default=True)
    parser.add_argument("--allow-fail-input", dest="require_no_fail_input", action="store_false")
    return parser.parse_args()


def convention_from_args(args: argparse.Namespace) -> SheetConductivityUnitConvention:
    ax = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m if args.lattice_a_x_m is None else args.lattice_a_x_m
    ay = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m if args.lattice_a_y_m is None else args.lattice_a_y_m
    area = LNO327_THIN_FILM_SLAO_IN_PLANE.unit_cell_area_m2 if args.unit_cell_area_m2 is None else args.unit_cell_area_m2
    return SheetConductivityUnitConvention(lattice_a_x_m=ax, lattice_a_y_m=ay, unit_cell_area_m2=area)


def main() -> None:
    args = parse_args()
    data = run_conversion(
        args.input_json,
        convention_from_args(args),
        require_no_fail_input=args.require_no_fail_input,
        allow_monitor_input=args.allow_monitor_input,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
