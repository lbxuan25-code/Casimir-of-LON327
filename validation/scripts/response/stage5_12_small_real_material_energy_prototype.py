#!/usr/bin/env python3
"""Stage 5.12 small real-material energy-integration prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.casimir.material_energy import (  # noqa: E402
    REQUIRED_WARNINGS,
    integrate_small_real_material_energy_prototype,
    load_stage5_11_reflection_grid,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "material_energy_prototype"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "material_reflection_grid" / "stage5_11c_real_material_reflection_grid_full36_order7_workers8.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_12_small_real_material_energy_prototype.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_12_small_real_material_energy_prototype.md"

BOUNDARY = {
    "no_response_rerun": True,
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "uses_stage5_11c_reflection_grid": True,
    "small_grid_only": True,
    "not_production_energy": True,
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


def _checks(scan: list[dict[str, Any]]) -> dict[str, str]:
    values = [complex(item["F_proto_over_area_J_m2"]) for item in scan]
    finite = all(np.isfinite(value.real) and np.isfinite(value.imag) for value in values)
    imag_ok = all(abs(value.imag) <= 1e-20 or abs(value.imag) <= 1e-8 * max(abs(value.real), 1e-300) for value in values)
    abs_values = [abs(value) for value in values]
    distance_ok = len(abs_values) < 3 or (abs_values[0] > abs_values[1] > abs_values[2])
    negative_ok = all(value.real < 0.0 for value in values)
    warnings = set().union(*(set(item["warnings"]) for item in scan))
    return {
        "input_status": "PASS",
        "finite_values": "PASS" if finite else "FAIL",
        "imaginary_part": "PASS" if imag_ok else "MONITOR",
        "distance_trend": "PASS" if distance_ok else "MONITOR",
        "negative_sign_sanity": "PASS" if negative_ok else "MONITOR",
        "warnings_present": "PASS" if set(REQUIRED_WARNINGS.values()).issubset(warnings) else "FAIL",
    }


def run_prototype(input_json: Path, *, separation_nm_values: list[float], allow_monitor: bool) -> dict[str, Any]:
    data = load_stage5_11_reflection_grid(input_json, allow_monitor=allow_monitor, require_no_fail=True)
    rows = data["point_results"]
    scan = []
    for separation_nm in separation_nm_values:
        result = integrate_small_real_material_energy_prototype(
            data,
            separation_m=float(separation_nm) * 1e-9,
            allow_monitor=allow_monitor,
        )
        scan.append({"separation_nm": float(separation_nm), **result})
    checks = _checks(scan)
    if any(value == "FAIL" for value in checks.values()):
        status = "STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_FAILED"
    elif any(value == "MONITOR" for value in checks.values()):
        status = "STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_MONITOR"
    else:
        status = "STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED"
    return {
        "stage": "Stage 5.12",
        "purpose": "Small real-material Casimir energy-integration prototype",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": data.get("diagnostic_status", {}).get("stage5_11_status"),
        },
        "scope": {
            "small_grid_only": True,
            "not_production_energy": True,
            "no_force": True,
            "no_torque": True,
            "n0_excluded": True,
            "zero_mode_not_included": True,
            "matsubara_grid_incomplete": True,
            "angular_grid_sparse": True,
            "Q_grid_sparse": True,
        },
        "grid_summary": {
            "n_values": sorted({int(row["n"]) for row in rows}),
            "Q_nm_inv_values": sorted({float(row["Q_nm_inv"]) for row in rows}),
            "phi_deg_values": sorted({float(row["phi_deg"]) for row in rows}),
            "num_points_used": len(rows),
        },
        "separation_scan": scan,
        "checks": checks,
        "diagnostic_status": {
            "stage5_12_status": status,
            "recommended_next_action": "Proceed to material-grid convergence planning; do not interpret prototype energy as physical prediction.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    scan_rows = [
        (item["separation_nm"], item["F_proto_over_area_J_m2"], item["imag_part_J_m2"], item["num_points_used"])
        for item in data["separation_scan"]
    ]
    first = data["separation_scan"][0] if data["separation_scan"] else {}
    return "\n\n".join(
        [
            "# Stage 5.12 small real-material energy prototype",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Scope and limitations\n\n" + _table(("quantity", "value"), list(data["scope"].items())),
            "## 4. Grid summary\n\n" + _table(("quantity", "value"), list(data["grid_summary"].items())),
            "## 5. Energy prototype formula\n\n"
            "`F_proto/A = k_B*T*sum_n' sum_Q,phi W_Qphi logdet[I-exp(-2*kappa*d) R R]`。这是稀疏 prototype quadrature。",
            "## 6. Separation scan\n\n" + _table(("d_nm", "F_proto/A", "imag", "points"), scan_rows),
            "## 7. Partial contributions by n\n\n" + str(first.get("partial_by_n", {})),
            "## 8. Partial contributions by Q\n\n" + str(first.get("partial_by_Q", {})),
            "## 9. Partial contributions by phi\n\n" + str(first.get("partial_by_phi", {})),
            "## 10. Checks\n\n" + _table(("check", "status"), list(data["checks"].items())),
            "## 11. What this is not\n\n"
            "这不是 production Casimir energy；缺少 n=0，Matsubara/Q/phi 网格极稀疏，因此不是物理预测。不输出 force，也不输出 torque。",
            "## 12. Diagnostic decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 13. Recommended next step\n\n" + data["diagnostic_status"]["recommended_next_action"],
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--separation-nm-values", nargs="+", type=float, default=[50.0, 100.0, 200.0])
    parser.add_argument("--allow-monitor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_prototype(args.input_json, separation_nm_values=args.separation_nm_values, allow_monitor=args.allow_monitor)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
