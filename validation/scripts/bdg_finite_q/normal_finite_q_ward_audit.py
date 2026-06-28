#!/usr/bin/env python3
"""Normal-state finite-q Ward residual audit for diagnostic output only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.tb_fourier import peierls_vertex_ward_residual  # noqa: E402
from lno327.ward_response import (  # noqa: E402
    normal_physical_density_current_response_components_imag_axis,
    physical_ward_residuals,
)

WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
RESPONSE_NAMES = ("bubble", "direct", "total")


def _complex_vector_components(vector: np.ndarray) -> list[dict[str, float | str]]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (3,):
        raise ValueError("Ward residual vector must have shape (3,)")
    return [
        {
            "component": label,
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }
        for label, value in zip(WARD_COMPONENT_LABELS, array, strict=True)
    ]


def _response_residual_row(response_name: str, matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    left, right = physical_ward_residuals(matrix, omega_eV, q)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return {
        "response_name": response_name,
        "residual_kind": "response_level",
        "residual_component_labels": list(WARD_COMPONENT_LABELS),
        "left_ward_residual_vector": _complex_vector_components(left),
        "right_ward_residual_vector": _complex_vector_components(right),
        "left_ward_residual_norm": left_norm,
        "right_ward_residual_norm": right_norm,
        "max_ward_residual_norm": float(max(left_norm, right_norm)),
        "valid_for_casimir_input": False,
    }


def _operator_level_rows(points: np.ndarray, q: np.ndarray) -> list[dict[str, Any]]:
    qx, qy = float(q[0]), float(q[1])
    rows: list[dict[str, Any]] = []
    for kx_value, ky_value in points:
        kx = float(kx_value)
        ky = float(ky_value)
        abs_error, rel_error, lhs_norm, rhs_norm = peierls_vertex_ward_residual(kx, ky, qx, qy)
        rows.append(
            {
                "residual_kind": "operator_level",
                "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                "k_model": [kx, ky],
                "absolute_error_norm": float(abs_error),
                "relative_error_norm": float(rel_error),
                "lhs_norm": float(lhs_norm),
                "rhs_norm": float(rhs_norm),
            }
        )
    return rows


def run_normal_finite_q_ward_audit(
    *,
    omega_eV: float = 0.01,
    q_values: tuple[float, ...] = (0.005, 0.01, 0.02),
    nk: int = 3,
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
) -> dict[str, Any]:
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    q_reports: list[dict[str, Any]] = []
    for q_value in q_values:
        q = np.array([float(q_value), 0.0], dtype=float)
        components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
        response_rows = [
            _response_residual_row(response_name, components[response_name], config.omega_eV, q)
            for response_name in RESPONSE_NAMES
        ]
        operator_rows = _operator_level_rows(points, q)
        q_reports.append(
            {
                "q_model": [float(q[0]), float(q[1])],
                "q_norm": float(np.linalg.norm(q)),
                "response_level_residuals": response_rows,
                "operator_level_peierls_ward": {
                    "residual_kind": "operator_level",
                    "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                    "max_absolute_error_norm": float(max(row["absolute_error_norm"] for row in operator_rows)),
                    "max_relative_error_norm": float(max(row["relative_error_norm"] for row in operator_rows)),
                    "per_k_residuals": operator_rows,
                },
            }
        )
    return {
        "audit_name": "normal_finite_q_ward_audit",
        "scope": "diagnostic_only_normal_state_finite_q_ward_residuals",
        "omega_eV": float(config.omega_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "nk": int(nk),
        "mesh_size": int(points.shape[0]),
        "q_values": [float(value) for value in q_values],
        "component_labels": list(WARD_COMPONENT_LABELS),
        "response_level_residuals_explain": (
            "bubble/direct/total are normal-state response-level residuals from physical_ward_residuals"
        ),
        "operator_level_residuals_explain": (
            "Peierls vertex identity is checked before response assembly and is distinct from response-level residuals"
        ),
        "q_reports": q_reports,
        "ward_identity_closed": False,
        "valid_for_casimir_input": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 normal-state finite-q Ward residual 诊断。")
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--nk", type=int, default=3)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    payload = run_normal_finite_q_ward_audit(
        omega_eV=args.omega,
        q_values=tuple(args.q_values),
        nk=args.nk,
        temperature_K=args.temperature_K,
        eta_eV=args.eta,
    )
    if args.json_output is not None:
        _write_json(args.json_output, payload)
    print(
        "normal finite-q Ward audit prepared: "
        f"q_values={payload['q_values']}, valid_for_casimir_input={payload['valid_for_casimir_input']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
