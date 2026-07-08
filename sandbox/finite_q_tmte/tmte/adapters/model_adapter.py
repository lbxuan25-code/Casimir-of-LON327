"""Model construction adapter for finite-q TM/TE sandbox scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from validation.lib.finite_q_validation_models import available_finite_q_validation_models, get_finite_q_validation_model

from ..theory.conventions import require_xi_matches_omega


@dataclass(frozen=True)
class ModelScanInputs:
    model: object
    spec: object
    ansatz: object
    pairing_params: object
    k_points: np.ndarray
    weights: np.ndarray
    config: KuboConfig


def available_models() -> tuple[str, ...]:
    return tuple(available_finite_q_validation_models())


def build_model_scan_inputs(
    *,
    model_name: str,
    pairing_name: str,
    xi_eV: float,
    nk: int,
    omega_eV: float | None = None,
    delta0_eV: float | None = None,
    phase_vertex: str = "bond_endpoint_gauge",
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
    k_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> ModelScanInputs:
    """Build existing validation model objects behind a sandbox adapter."""

    selected_omega = float(xi_eV) if omega_eV is None else float(omega_eV)
    require_xi_matches_omega(xi_eV, selected_omega)
    model = get_finite_q_validation_model(model_name)
    model.require_pairing(pairing_name)
    ansatz = model.build_ansatz(pairing_name, phase_vertex=phase_vertex)
    pairing_params = model.build_pairing_params(delta0_eV)
    points = uniform_bz_mesh(nk) if k_points is None else np.asarray(k_points, dtype=float)
    mesh_weights = k_weights(points) if weights is None else np.asarray(weights, dtype=float)
    config = KuboConfig.from_kelvin(omega_eV=selected_omega, temperature_K=float(temperature_K), eta_eV=float(eta_eV), output_si=False)
    return ModelScanInputs(model=model, spec=model.spec, ansatz=ansatz, pairing_params=pairing_params, k_points=points, weights=mesh_weights, config=config)


def shifted_uniform_bz_mesh(nk: int, shift_fraction_x: float, shift_fraction_y: float) -> np.ndarray:
    """Return a shifted midpoint mesh over [-pi, pi) x [-pi, pi)."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    spacing = 2.0 * np.pi / int(nk)
    kx_values = -np.pi + (np.arange(nk) + 0.5 + float(shift_fraction_x)) * spacing
    ky_values = -np.pi + (np.arange(nk) + 0.5 + float(shift_fraction_y)) * spacing
    kx_values = ((kx_values + np.pi) % (2.0 * np.pi)) - np.pi
    ky_values = ((ky_values + np.pi) % (2.0 * np.pi)) - np.pi
    return np.array([(kx, ky) for kx in kx_values for ky in ky_values], dtype=float)


def weights_for_points(points: np.ndarray) -> np.ndarray:
    return k_weights(points)
