"""q-independent material cache for the vectorized finite-q production path."""

from __future__ import annotations

import numpy as np

from lno327.response.config import KuboConfig
from lno327.response.finite_q_bdg import (
    _DefaultFiniteQOptions,
    _check_options,
    _pairing_params_from_inputs,
    bdg_eigensystem_from_model_pairing,
    require_peierls_finite_q_support,
)
from lno327.response.finite_q_optimized import FiniteQMaterialWorkspace
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs

_COLLECTIVE_CHANNELS = 2


def _static_factor_matrix(
    energies: np.ndarray,
    occupations: np.ndarray,
    config: KuboConfig,
) -> np.ndarray:
    em = np.asarray(energies, dtype=float)[:, None]
    en = np.asarray(energies, dtype=float)[None, :]
    fm = np.asarray(occupations, dtype=float)[:, None]
    fn = np.asarray(occupations, dtype=float)[None, :]
    delta_e = em - en
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (fm - fn) / delta_e

    degenerate = np.abs(delta_e) < float(config.eta_eV)
    if np.any(degenerate):
        shifted = em - float(config.fermi_level_eV)
        if config.temperature_eV <= 0.0:
            width = max(float(config.eta_eV), 1e-12)
            derivative = -width / (np.pi * (shifted * shifted + width * width))
        else:
            x = np.clip(
                shifted / (2.0 * float(config.temperature_eV)), -350.0, 350.0
            )
            derivative = -1.0 / (
                4.0 * float(config.temperature_eV) * np.cosh(x) ** 2
            )
        factor = np.where(degenerate, derivative, factor)
    return np.asarray(factor, dtype=complex)


def _goldstone_counterterm_from_cached_midpoint(
    ansatz: object,
    pairing_params: object,
    points: np.ndarray,
    weights: np.ndarray,
    energies: np.ndarray,
    states: np.ndarray,
    occupations: np.ndarray,
    config: KuboConfig,
) -> np.ndarray:
    eta2_bubble = 0.0 + 0.0j
    for index, (weight, (kx_value, ky_value)) in enumerate(
        zip(weights, points, strict=True)
    ):
        collective = tuple(
            ansatz.collective_vertices(
                float(kx_value),
                float(ky_value),
                0.0,
                0.0,
                pairing_params,
            )
        )
        if len(collective) != _COLLECTIVE_CHANNELS:
            raise ValueError("optimized workspace requires exactly two collective channels")
        eta2_band = states[index].conjugate().T @ collective[1] @ states[index]
        eta2_bubble += 0.5 * float(weight) * np.sum(
            _static_factor_matrix(energies[index], occupations[index], config)
            * eta2_band
            * np.conjugate(eta2_band)
        )
    return -complex(eta2_bubble) * np.eye(_COLLECTIVE_CHANNELS, dtype=complex)


def precompute_finite_q_material_workspace_from_model_ansatz(
    spec: object,
    ansatz: object,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: object | None = None,
    options: object | None = None,
) -> FiniteQMaterialWorkspace:
    """Build the q-independent cache with no duplicate midpoint eigensystem pass."""

    opts = options or _DefaultFiniteQOptions()
    _check_options(opts)
    if opts.current_vertex != "peierls":
        raise ValueError("optimized production workspace requires current_vertex='peierls'")
    if opts.collective_mode != "amplitude_phase":
        raise ValueError(
            "optimized production workspace requires collective_mode='amplitude_phase'"
        )
    require_peierls_finite_q_support(spec)
    _, points, weights = validate_finite_q_inputs(
        np.zeros(2, dtype=float), k_points, k_weights, config
    )
    amp = _pairing_params_from_inputs(spec, pairing_params)
    if float(getattr(amp, "delta0_eV", 0.0)) <= 0.0:
        raise ValueError("optimized production workspace requires nonzero delta0")

    energy_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    occupation_rows: list[np.ndarray] = []
    for kx_value, ky_value in points:
        kx, ky = float(kx_value), float(ky_value)
        bands = bdg_eigensystem_from_model_pairing(
            spec,
            kx,
            ky,
            ansatz.mean_pairing(kx, ky, amp),
        )
        energy_rows.append(np.asarray(bands.energies, dtype=float))
        state_rows.append(np.asarray(bands.states, dtype=complex))
        occupation_rows.append(
            np.asarray(
                fermi_function(
                    bands.energies,
                    config.fermi_level_eV,
                    config.temperature_eV,
                ),
                dtype=float,
            )
        )

    midpoint_energies = np.stack(energy_rows, axis=0)
    midpoint_states = np.stack(state_rows, axis=0)
    midpoint_occupations = np.stack(occupation_rows, axis=0)
    counterterm = np.zeros((_COLLECTIVE_CHANNELS, _COLLECTIVE_CHANNELS), dtype=complex)
    if opts.collective_counterterm == "goldstone_gap_equation":
        counterterm = _goldstone_counterterm_from_cached_midpoint(
            ansatz,
            amp,
            points,
            weights,
            midpoint_energies,
            midpoint_states,
            midpoint_occupations,
            config,
        )

    return FiniteQMaterialWorkspace(
        spec=spec,
        ansatz=ansatz,
        k_points=points,
        k_weights=weights,
        config=config,
        pairing_params=amp,
        options=opts,
        collective_mode="amplitude_phase",
        collective_mode_disabled_reason=None,
        midpoint_energies=midpoint_energies,
        midpoint_states=midpoint_states,
        midpoint_occupations=midpoint_occupations,
        collective_counterterm_matrix=counterterm,
        metadata={
            "workspace_kind": "finite_q_material_vectorized",
            "q_independent": True,
            "midpoint_eigensystem_count": int(points.shape[0]),
            "duplicate_midpoint_eigensystem_passes": 0,
            "goldstone_counterterm_cached_once": True,
            "goldstone_counterterm_from_cached_midpoint_bands": True,
            "unified_channel_order": ("rho", "Jx", "Jy", "eta1", "eta2"),
            "production_collective_mode": "amplitude_phase",
            "production_current_vertex": "peierls",
        },
    )


__all__ = ["precompute_finite_q_material_workspace_from_model_ansatz"]
