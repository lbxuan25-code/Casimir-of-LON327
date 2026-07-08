from __future__ import annotations

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.diagnostics import response_diagnostics


def test_gauge_diagnostics_reject_physical_only_source_order():
    with pytest.raises(ValueError, match="diagnostic source order"):
        response_diagnostics(
            k_eff=np.eye(2, dtype=complex),
            schur_correction=np.zeros((2, 2), dtype=complex),
            etaeta_condition_number=1.0,
            schur_solve_method="solve",
            schur_numerically_suspect=False,
            source_order=("TM", "TE"),
        )
