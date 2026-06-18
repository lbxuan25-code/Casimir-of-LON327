"""Shared helpers for BdG finite-q response validation audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.bdg_finite_q_response import bdg_finite_q_response_imag_axis  # noqa: E402
from lno327.bdg_response import bdg_superconducting_response_imag_axis  # noqa: E402
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.conductivity_conventions import spatial_response_to_bilayer_sheet_conductivity_model  # noqa: E402
from lno327.conductivity_units import SheetConductivityUnitConvention, model_to_dimensionless_sheet_conductivity  # noqa: E402
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.reflection_input import sigma_tilde_xy_to_te_tm_reflection_matrix, symmetric_antisymmetric_offdiag  # noqa: E402
from lno327.ward_response import normal_physical_density_current_response_imag_axis, physical_ward_residuals  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"


def parser(description: str) -> argparse.ArgumentParser:
    argp = argparse.ArgumentParser(description=description)
    argp.add_argument("--quick", action="store_true")
    return argp


def grid(quick: bool) -> tuple[np.ndarray, np.ndarray]:
    n = 4 if quick else 8
    points = uniform_bz_mesh(n)
    return points, k_weights(points)


def config(omega_eV: float, *, temperature_K: float = 10.0) -> KuboConfig:
    return KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=temperature_K, eta_eV=1e-8, output_si=False)


def cjson(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return cjson(value.tolist())
    if isinstance(value, np.generic):
        return cjson(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): cjson(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cjson(v) for v in value]
    return value


def status_from_failures(failures: list[str], monitors: list[str] | None = None) -> str:
    if failures:
        return "FAILED"
    if monitors:
        return "MONITOR"
    return "PASSED"


def write_report(name: str, payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{name}.json"
    md_path = OUTPUT_DIR / f"{name}.md"
    json_path.write_text(json.dumps(cjson(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {name}",
        "",
        f"- status: {payload.get('status')}",
        f"- quick: {payload.get('quick')}",
        f"- cases: {len(payload.get('cases', []))}",
    ]
    for key in ("summary", "failures", "monitors"):
        if key in payload:
            lines.append(f"- {key}: {cjson(payload[key])}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, float | bool]:
    return {
        "all_finite": bool(np.all(np.isfinite(matrix))),
        "hermiticity_abs": float(np.max(np.abs(matrix - matrix.conjugate().T))),
        "max_abs": float(np.max(np.abs(matrix))),
    }


def ward_norms(response: np.ndarray, omega_eV: float, q_model: np.ndarray) -> dict[str, float]:
    left, right = physical_ward_residuals(response, omega_eV, q_model)
    return {
        "left_norm": float(np.linalg.norm(left)),
        "right_norm": float(np.linalg.norm(right)),
        "max_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
    }


def response_case(pairing: str, omega: float, q: np.ndarray, delta0: float, quick: bool, *, phase: bool = True):
    points, weights = grid(quick)
    cfg = config(omega)
    return bdg_finite_q_response_imag_axis(
        pairing,
        omega,
        q,
        points,
        weights,
        cfg,
        PairingAmplitudes(delta0_eV=delta0),
        include_phase_correction=phase,
    )


def normal_response(omega: float, q: np.ndarray, quick: bool) -> np.ndarray:
    points, weights = grid(quick)
    return normal_physical_density_current_response_imag_axis(points, config(omega), q, weights)


def local_bdg_response_as_pi(pairing: str, omega: float, delta0: float, quick: bool) -> np.ndarray:
    points, weights = grid(quick)
    cfg = config(omega)
    local = bdg_superconducting_response_imag_axis(
        points,
        cfg,
        pairing,  # type: ignore[arg-type]
        PairingAmplitudes(delta0_eV=delta0),
        weights,
    )
    response = np.zeros((3, 3), dtype=complex)
    response[1:3, 1:3] = -omega * local.sigma_like_response
    return response


def reflection_package(response: np.ndarray, omega: float, q: np.ndarray) -> dict[str, Any]:
    structure = LNO327_THIN_FILM_SLAO_IN_PLANE
    convention = SheetConductivityUnitConvention(
        lattice_a_x_m=structure.lattice_a_x_m,
        lattice_a_y_m=structure.lattice_a_y_m,
        unit_cell_area_m2=structure.unit_cell_area_m2,
    )
    sigma_model = spatial_response_to_bilayer_sheet_conductivity_model(response, omega)
    sigma_tilde = model_to_dimensionless_sheet_conductivity(sigma_model, convention)
    refl = sigma_tilde_xy_to_te_tm_reflection_matrix(
        sigma_tilde,
        float(q[0]),
        float(q[1]),
        omega,
        convention.lattice_a_x_m,
        convention.lattice_a_y_m,
    )
    return {
        "sigma_model": sigma_model,
        "sigma_tilde": sigma_tilde,
        "reflection_TE_TM": refl["reflection_TE_TM"],
        "offdiag": symmetric_antisymmetric_offdiag(sigma_model),
    }
