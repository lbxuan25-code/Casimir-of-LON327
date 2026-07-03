import numpy as np

import lno327.models.lno327_four_orbital.spec as four_spec_module
import lno327.models.symmetry_bdg_2band.spec as two_spec_module
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


def _assert_spec_cache_behavior(spec):
    terms_a = spec.hopping_terms()
    terms_b = spec.hopping_terms()
    assert terms_a is terms_b

    vector_default = spec.peierls_hamiltonian_vector_vertex(0.21, -0.34, 0.17, -0.09, "x")
    assert spec.hopping_terms() is terms_a
    vector_explicit = spec.peierls_hamiltonian_vector_vertex(
        0.21,
        -0.34,
        0.17,
        -0.09,
        "x",
        hopping_terms=terms_a,
    )
    np.testing.assert_allclose(vector_default, vector_explicit)

    contact_default = spec.peierls_hamiltonian_contact_vertex(0.21, -0.34, 0.17, -0.09, "x", "y")
    assert spec.hopping_terms() is terms_a
    contact_explicit = spec.peierls_hamiltonian_contact_vertex(
        0.21,
        -0.34,
        0.17,
        -0.09,
        "x",
        "y",
        hopping_terms=terms_a,
    )
    np.testing.assert_allclose(contact_default, contact_explicit)


def test_four_orbital_spec_caches_hopping_terms():
    _assert_spec_cache_behavior(LNO327FourOrbitalSpec())


def test_two_band_spec_caches_hopping_terms():
    _assert_spec_cache_behavior(SymmetryBdG2BandSpec())


def test_four_orbital_spec_vertices_do_not_regenerate_hopping_terms(monkeypatch):
    spec = LNO327FourOrbitalSpec()

    def fail(_params):
        raise AssertionError("hopping terms regenerated")

    monkeypatch.setattr(four_spec_module, "normal_state_hopping_terms", fail)
    spec.peierls_hamiltonian_vector_vertex(0.21, -0.34, 0.17, -0.09, "x")
    spec.peierls_hamiltonian_contact_vertex(0.21, -0.34, 0.17, -0.09, "x", "y")


def test_two_band_spec_vertices_do_not_regenerate_hopping_terms(monkeypatch):
    spec = SymmetryBdG2BandSpec()

    def fail(_params):
        raise AssertionError("hopping terms regenerated")

    monkeypatch.setattr(two_spec_module, "normal_state_hopping_terms", fail)
    spec.peierls_hamiltonian_vector_vertex(0.21, -0.34, 0.17, -0.09, "x")
    spec.peierls_hamiltonian_contact_vertex(0.21, -0.34, 0.17, -0.09, "x", "y")
