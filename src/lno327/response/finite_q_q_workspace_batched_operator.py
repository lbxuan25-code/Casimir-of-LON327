"""Compatibility wrapper for operator-enabled batched q workspaces.

The physical implementation lives only in ``finite_q_q_workspace_batched``.
Keeping this import surface avoids breaking historical callers while eliminating
the duplicated shifted-Hamiltonian, vertex, direct-term and Ward-RHS body.
"""
from __future__ import annotations

import numpy as np

from lno327.response.finite_q_optimized import FiniteQMaterialWorkspace, FiniteQQWorkspace
from lno327.response.finite_q_q_workspace_batched import (
    precompute_finite_q_q_workspace_batched,
)


def precompute_finite_q_q_workspace_batched_operator(
    material: FiniteQMaterialWorkspace,
    q_model: np.ndarray,
) -> FiniteQQWorkspace:
    return precompute_finite_q_q_workspace_batched(
        material,
        q_model,
        operator_diagnostics=True,
    )


__all__ = ["precompute_finite_q_q_workspace_batched_operator"]
