from __future__ import annotations

import numpy as np

from sandbox.finite_q_tmte.tmte.theory.contacts import project_spatial_contact, spatial_contact_vectors
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions


def test_contact_projection_includes_offdiagonal_signs():
    conventions = finite_q_conventions(np.asarray([0.0, 0.2]), xi_eV=0.01)
    d = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    projected = project_spatial_contact(d, conventions)
    vectors = spatial_contact_vectors(conventions)
    order = ("G", "TM", "TE")
    expected = np.asarray([[vectors[a] @ d @ vectors[b] for b in order] for a in order], dtype=complex)
    np.testing.assert_allclose(projected, expected)
    assert projected[1, 2] == vectors["TM"] @ d @ vectors["TE"]
