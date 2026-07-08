from __future__ import annotations

import numpy as np

from sandbox.finite_q_tmte.tmte.theory.basis import aligned_source_vectors, component_source_vectors
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions


def test_basis_vectors_use_fixed_te_sign():
    conventions = finite_q_conventions(np.asarray([0.0, 0.2]), xi=0.01)
    aligned = aligned_source_vectors(conventions)
    component = component_source_vectors(conventions)
    np.testing.assert_allclose(aligned["TE"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(component["TE"], [0.0, -1.0, 0.0])

