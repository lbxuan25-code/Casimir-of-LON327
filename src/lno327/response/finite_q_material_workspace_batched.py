"""Batched midpoint material workspace for models with vectorized BdG support.

This module changes only execution order.  It builds the same q-independent
midpoint eigensystems, occupations, and collective counterterm as
``precompute_finite_q_material_workspace_from_model_ansatz`` while evaluating all
microscopic k points through the model's batch capabilities.
"""

from __future__ import annotations

import numpy as np

from lno327.response.finite_q_bdg import (
    _DefaultFiniteQOptions,
    _check_options,
    _pairing_params_from_inputs,
    require_peierls_finite_q_support,
)
from lno327.response.finite_q_optimized import FiniteQMaterialWorkspace
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs

_COLLECTIVE_CHANNELS = 2


def supports_batched_finite_q_material_workspace(
    spec: object,
    ansatz: object,
) -> bool:
    """Return whether midpoint BdG construction can run over all k at once."""

    return bool(
        callable(getattr(spec, "bdg_hamiltonian_from_pairing_batch", None))
        and callable(getattr(ansatz, "mean_pairing_batch", None))
    )


def precompute_finite_q_material_workspace_batched(
    spec: object,
    ansatz: object,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: object,
    pairing_params: object | None = None,
    options: object | None = None,
) -> FiniteQMaterialWorkspace:
    """Build the q-independent material workspace with batched eigensystems."""

    opts = options or _DefaultFiniteQOptions()
    _check_options(opts)
    _, points, weights = validate_finite_q_inputs(
        np.zeros(2, dtype=float),
        k_points,
        k_weights,
        config,
    )
    if opts.current_vertex == "peierls":
        require_peierls_finite_q_support(spec)
    if not supports_batched_finite_q_material_workspace(spec, ansatz):
        raise ValueError("model/ansatz do not support the batched material workspace")

    amp = _pairing_params_from_inputs(spec, pairing_params)
    collective_mode = opts.collective_mode
    disabled = None
    if (
        float(getattr(amp, "delta0_eV", 0.0)) == 0.0
        and collective_mode == "amplitude_phase"
    ):
        collective_mode = "none"
        disabled = "delta0=0 normal limit"
    if collective_mode != "amplitude_phase":
        raise ValueError(
            "optimized production workspace requires collective_mode='amplitude_phase' "
            "with nonzero delta0"
        )

    pairing_batch = np.asarray(
        ansatz.mean_pairing_batch(points, amp),
        dtype=complex,
    )
    hamiltonians = np.asarray(
        spec.bdg_hamiltonian_from_pairing_batch(points, pairing_batch),
        dtype=complex,
    )
    if hamiltonians.ndim != 3 or hamiltonians.shape[0] != points.shape[0]:
        raise ValueError(
            "batched BdG Hamiltonians must have shape (nk, nb, nb)"
        )
    if hamiltonians.shape[1] != hamiltonians.shape[2]:
        raise ValueError("batched BdG Hamiltonians must be square")

    midpoint_energies, midpoint_states = np.linalg.eigh(hamiltonians)
    midpoint_occupations = fermi_function(
        midpoint_energies,
        config.fermi_level_eV,
        config.temperature_eV,
    )

    counterterm = np.zeros(
        (_COLLECTIVE_CHANNELS, _COLLECTIVE_CHANNELS),
        dtype=complex,
    )
    if opts.collective_counterterm == "goldstone_gap_equation":
        counterterm = np.asarray(
            ansatz.hs_counterterm(config, points, weights, amp),
            dtype=complex,
        )

    return FiniteQMaterialWorkspace(
        spec=spec,
        ansatz=ansatz,
        k_points=points,
        k_weights=weights,
        config=config,
        pairing_params=amp,
        options=opts,
        collective_mode=collective_mode,
        collective_mode_disabled_reason=disabled,
        midpoint_energies=midpoint_energies,
        midpoint_states=midpoint_states,
        midpoint_occupations=midpoint_occupations,
        collective_counterterm_matrix=counterterm,
        metadata={
            "workspace_kind": "finite_q_material_vectorized",
            "material_workspace_implementation": "batched_model_capability",
            "q_independent": True,
            "midpoint_eigensystem_count": int(points.shape[0]),
            "goldstone_counterterm_cached_once": True,
            "unified_channel_order": ("rho", "Jx", "Jy", "eta1", "eta2"),
            "production_collective_mode": "amplitude_phase",
        },
    )


__all__ = [
    "precompute_finite_q_material_workspace_batched",
    "supports_batched_finite_q_material_workspace",
]
