from __future__ import annotations

from validation.lib.commensurate_periodic import CommensuratePeriodicGrid


def test_even_integer_shift_keeps_half_q_endpoints_on_same_sublattice():
    grid = CommensuratePeriodicGrid(nk=8, mx=2, my=-4, max_points=64)

    assert grid.translation_permutation_exact
    assert grid.half_translation_permutation_exact
    assert grid.half_translation_sublattice_offset == (0.0, 0.0)


def test_odd_integer_component_moves_half_q_endpoints_to_complementary_sublattice():
    grid = CommensuratePeriodicGrid(nk=8, mx=3, my=2, max_points=64)

    assert grid.translation_permutation_exact
    assert not grid.half_translation_permutation_exact
    assert grid.half_translation_sublattice_offset == (0.5, 0.0)


def test_two_odd_components_shift_both_half_q_sublattices():
    grid = CommensuratePeriodicGrid(nk=8, mx=-1, my=3, max_points=64)

    assert not grid.half_translation_permutation_exact
    assert grid.half_translation_sublattice_offset == (0.5, 0.5)
