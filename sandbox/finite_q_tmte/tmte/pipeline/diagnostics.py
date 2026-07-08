"""Diagnostics for Schur-corrected target-basis response blocks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..theory.basis import physical_indices


def response_diagnostics(
    *,
    k_eff: np.ndarray,
    schur_correction: np.ndarray,
    etaeta_condition_number: float,
    source_order: tuple[str, ...],
) -> dict[str, Any]:
    """Return gauge and physical block diagnostics."""

    eff = np.asarray(k_eff, dtype=complex)
    g_index = source_order.index("G")
    phys = physical_indices(source_order)
    k_tmte = eff[np.ix_(phys, phys)]
    return {
        "gauge_row_norm": float(np.linalg.norm(eff[g_index, :])),
        "gauge_col_norm": float(np.linalg.norm(eff[:, g_index])),
        "gauge_gg_norm": float(abs(eff[g_index, g_index])),
        "etaeta_condition_number": float(etaeta_condition_number),
        "physical_matrix_norm": float(np.linalg.norm(k_tmte)),
        "schur_correction_norm": float(np.linalg.norm(schur_correction)),
        "valid_for_casimir_input": False,
    }

