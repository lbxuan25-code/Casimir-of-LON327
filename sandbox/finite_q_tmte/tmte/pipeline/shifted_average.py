"""Shifted-mesh average helpers for direct target-basis blocks."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks
from .block_builder import EffectiveTargetResponse, build_effective_from_blocks


def shift_pairs_from_fractions(shift_fractions: Sequence[float]) -> tuple[tuple[float, float], ...]:
    shifts = tuple(float(value) for value in shift_fractions)
    if not shifts:
        raise ValueError("shift_fractions must not be empty")
    return tuple((sx, sy) for sx in shifts for sy in shifts)


def _mean_matrix(blocks: Sequence[TargetBareBlocks], field: str) -> np.ndarray:
    values = [np.asarray(getattr(block, field), dtype=complex) for block in blocks]
    return sum(values, np.zeros_like(values[0], dtype=complex)) / len(values)


def average_bare_blocks_then_schur(blocks: Sequence[TargetBareBlocks]) -> EffectiveTargetResponse:
    """Average bare blocks first, then apply Schur as the main result."""

    if not blocks:
        raise ValueError("blocks must not be empty")
    first = blocks[0]
    averaged = replace(
        first,
        k_ss_bubble=_mean_matrix(blocks, "k_ss_bubble"),
        k_ss_contact=_mean_matrix(blocks, "k_ss_contact"),
        k_ss=_mean_matrix(blocks, "k_ss"),
        k_seta=_mean_matrix(blocks, "k_seta"),
        k_etas=_mean_matrix(blocks, "k_etas"),
        k_etaeta_bubble=_mean_matrix(blocks, "k_etaeta_bubble"),
        k_etaeta_counterterm=_mean_matrix(blocks, "k_etaeta_counterterm"),
        k_etaeta=_mean_matrix(blocks, "k_etaeta"),
        metadata={
            **first.metadata,
            "shifted_average": True,
            "average_order": "average_blocks_then_schur",
            "num_shifted_meshes": len(blocks),
            "valid_for_casimir_input": False,
        },
    )
    return build_effective_from_blocks(averaged)


def per_shift_summaries(responses: Sequence[EffectiveTargetResponse], shift_pairs: Sequence[tuple[float, float]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for response, shift in zip(responses, shift_pairs, strict=True):
        summaries.append(
            {
                "shift": [float(shift[0]), float(shift[1])],
                "K_SS_norm": float(np.linalg.norm(response.bare_blocks.k_ss)),
                "K_TMTE_eff_norm": float(np.linalg.norm(response.k_tmte_eff)),
                "etaeta_condition_number": float(response.schur.etaeta_condition_number),
                "valid_for_casimir_input": False,
            }
        )
    return summaries

