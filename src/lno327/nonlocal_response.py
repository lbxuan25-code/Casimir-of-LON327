"""Nonlocal sheet-response interface placeholders for future finite-q work."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .pairing import PairingAmplitudes
from .response_interface import LocalSheetResponse, ResponseKind, local_response_imag_axis

NonlocalResponseMode = Literal["local_fallback", "finite_q_placeholder"]


@dataclass(frozen=True)
class MomentumTransfer:
    """In-plane momentum transfer used by future nonlocal response functions."""

    q_parallel: float
    phi: float


@dataclass(frozen=True)
class NonlocalSheetResponse:
    """Sheet response tagged with nonlocal-resolution status."""

    kind: ResponseKind
    omega_eV: float
    momentum: MomentumTransfer
    matrix: np.ndarray
    mode: NonlocalResponseMode
    nonlocal_resolved: bool
    local_fallback_used: bool
    unit_label: str
    source: str
    valid_for_casimir_input: bool
    notes: tuple[str, ...]


def nonlocal_response_imag_axis(
    kind: ResponseKind,
    omega_eV: float,
    q_parallel: float,
    phi: float,
    mode: NonlocalResponseMode,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    temperature_K: float,
    eta_eV: float = 1e-4,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> NonlocalSheetResponse:
    """Return a tagged nonlocal response, or an explicit local fallback.

    ``finite_q_placeholder`` intentionally raises ``NotImplementedError`` so
    future finite-q BdG and normal-state response implementations have a clear
    API target without being silently replaced by local physics.
    """

    if q_parallel < 0.0:
        raise ValueError("q_parallel must be non-negative")
    if mode == "finite_q_placeholder":
        raise NotImplementedError("finite-q sheet response is not implemented yet")
    if mode != "local_fallback":
        raise ValueError("mode must be 'local_fallback' or 'finite_q_placeholder'")

    local = local_response_imag_axis(
        kind,
        omega_eV,
        k_points,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=pairing_params,
        k_weights=k_weights,
    )
    return NonlocalSheetResponse(
        kind=kind,
        omega_eV=omega_eV,
        momentum=MomentumTransfer(q_parallel=q_parallel, phi=phi),
        matrix=local.matrix,
        mode=mode,
        nonlocal_resolved=False,
        local_fallback_used=True,
        unit_label=local.unit_label,
        source=local.source,
        valid_for_casimir_input=False,
        notes=local.notes
        + (
            "nonlocal q_parallel response unresolved",
            f"local_fallback used at q_parallel={q_parallel}",
        ),
    )
