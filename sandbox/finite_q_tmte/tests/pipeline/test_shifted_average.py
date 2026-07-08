from __future__ import annotations

import numpy as np

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.shifted_average import average_bare_blocks_then_schur
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.schur import schur_effective


def _blocks(scale: float) -> TargetBareBlocks:
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi_eV=0.01)
    k_ss = scale * np.eye(3, dtype=complex)
    k_seta = scale * np.asarray([[1.0, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=complex)
    k_etas = scale * np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=complex)
    k_etaeta = np.asarray([[2.0 + scale, 0.1], [0.2, 3.0 + scale]], dtype=complex)
    return TargetBareBlocks(
        source_order=("G", "TM", "TE"),
        conventions=conventions,
        k_ss_bubble=k_ss,
        k_ss_contact=np.zeros_like(k_ss),
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta_bubble=k_etaeta,
        k_etaeta_counterterm=np.zeros_like(k_etaeta),
        k_etaeta=k_etaeta,
        metadata={},
    )


def test_shifted_average_averages_blocks_before_schur():
    blocks = [_blocks(1.0), _blocks(2.0)]
    response = average_bare_blocks_then_schur(blocks)
    mean_ss = 0.5 * (blocks[0].k_ss + blocks[1].k_ss)
    mean_seta = 0.5 * (blocks[0].k_seta + blocks[1].k_seta)
    mean_etas = 0.5 * (blocks[0].k_etas + blocks[1].k_etas)
    mean_etaeta = 0.5 * (blocks[0].k_etaeta + blocks[1].k_etaeta)
    expected = schur_effective(mean_ss, mean_seta, mean_etaeta, mean_etas).effective
    per_shift_average = 0.5 * (
        schur_effective(blocks[0].k_ss, blocks[0].k_seta, blocks[0].k_etaeta, blocks[0].k_etas).effective
        + schur_effective(blocks[1].k_ss, blocks[1].k_seta, blocks[1].k_etaeta, blocks[1].k_etas).effective
    )
    np.testing.assert_allclose(response.schur.effective, expected)
    assert not np.allclose(response.schur.effective, per_shift_average)
