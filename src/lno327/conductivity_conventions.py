"""Explicit response-to-conductivity convention helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def spatial_response_to_bilayer_sheet_conductivity_model(
    response: np.ndarray,
    omega_eV: float,
) -> np.ndarray:
    """
    Convert the spatial block of the finite-q physical response matrix
    to bilayer-normalized model sheet conductivity on the imaginary axis.

    Convention:
        Pi_ij = delta <j_i> / delta A_j
        E_j(i xi) = - xi A_j(i xi)
        sigma_ij(i xi) = - Pi_ij(i xi) / xi

    Code frequency variable:
        omega_eV = hbar * xi, positive Matsubara energy in eV.

    Model-level conversion:
        sigma_model_ij(iOmega) = - response[1:3, 1:3] / omega_eV

    This returns a model-level bilayer sheet conductivity tensor.
    It does not yet apply final SI sheet-conductivity scaling.
    It does not create a 3D bulk conductivity.
    """

    matrix = np.asarray(response)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    if omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive")
    pi_spatial = matrix[1:3, 1:3]
    return -pi_spatial / omega_eV


def bilayer_sheet_conductivity_convention_metadata() -> dict[str, Any]:
    """Return metadata for the Stage 5.1b bilayer sheet convention."""

    return {
        "response_interpretation": "Pi_ij = delta<j_i>/delta A_j",
        "electric_field_relation": "E_j(i xi) = - xi A_j(i xi)",
        "model_formula": "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV",
        "normalization": "bilayer-normalized 2D sheet conductivity",
        "not_bulk_3d": True,
        "not_single_layer": True,
        "si_scaling_applied": False,
    }
