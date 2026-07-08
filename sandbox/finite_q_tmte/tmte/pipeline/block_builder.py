"""Build effective target-basis response objects from bare blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..theory.basis import physical_indices
from ..theory.schur import TargetSchurResult, schur_effective
from .diagnostics import response_diagnostics


@dataclass(frozen=True)
class EffectiveTargetResponse:
    bare_blocks: TargetBareBlocks
    schur: TargetSchurResult
    k_tmte_eff: np.ndarray
    diagnostics: dict[str, Any]


def build_effective_from_blocks(blocks: TargetBareBlocks) -> EffectiveTargetResponse:
    """Apply Schur to target-basis bare blocks and slice K_TMTE_eff."""

    schur = schur_effective(blocks.k_ss, blocks.k_seta, blocks.k_etaeta, blocks.k_etas)
    phys = physical_indices(blocks.source_order)
    k_tmte = schur.effective[np.ix_(phys, phys)]
    diagnostics = response_diagnostics(
        k_eff=schur.effective,
        schur_correction=schur.correction,
        etaeta_condition_number=schur.etaeta_condition_number,
        schur_solve_method=schur.solve_method,
        schur_numerically_suspect=schur.numerically_suspect,
        source_order=blocks.source_order,
    )
    return EffectiveTargetResponse(bare_blocks=blocks, schur=schur, k_tmte_eff=k_tmte, diagnostics=diagnostics)


def compute_effective_target_response(**kwargs: Any) -> EffectiveTargetResponse:
    """Compute bare target blocks through the adapter, then Schur them."""

    return build_effective_from_blocks(compute_target_bare_blocks(**kwargs))
