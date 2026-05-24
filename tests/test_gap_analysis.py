import numpy as np
import pytest

from lno327 import (
    PairingAmplitudes,
    band_gap_projection,
    fermi_surface_points,
    gap_statistics_by_band,
    gap_statistics_on_fermi_surface,
)


def test_band_gap_projection_shape():
    gaps = band_gap_projection(0.2, -0.5, "spm", PairingAmplitudes(delta0_eV=0.04))

    assert gaps.shape == (4,)
    assert gaps.dtype == complex


def test_spm_and_dwave_output_finite_gap_data():
    for kind in ("spm", "dwave"):
        stats = gap_statistics_on_fermi_surface(
            kind,
            PairingAmplitudes(delta0_eV=0.04),
            nk=12,
            energy_tolerance_eV=0.15,
        )

        assert stats.gap_complex.shape == stats.gap_abs.shape
        assert stats.gap_complex.size > 0
        assert np.isfinite(stats.gap_complex).all()
        assert np.isfinite(stats.gap_abs).all()
        assert np.isfinite(stats.mean_abs_gap)


def test_dwave_form_factor_zero_has_small_projected_gap():
    gaps = band_gap_projection(np.pi / 2.0, np.pi / 2.0, "dwave", PairingAmplitudes(delta0_eV=0.04))

    np.testing.assert_allclose(gaps, np.zeros(4), atol=1e-14)


def test_fermi_surface_points_rejects_nonpositive_energy_tolerance():
    with pytest.raises(ValueError, match="energy_tolerance_eV must be positive"):
        fermi_surface_points(8, 0.0)


def test_band_resolved_summary_counts_sum_to_fermi_surface_points():
    stats = gap_statistics_on_fermi_surface(
        "spm",
        PairingAmplitudes(delta0_eV=0.04),
        nk=12,
        energy_tolerance_eV=0.15,
    )
    summary = gap_statistics_by_band(stats)

    assert sum(item["count"] for item in summary.values()) == stats.gap_abs.size


def test_larger_node_tolerance_does_not_reduce_node_count():
    small_tolerance = gap_statistics_on_fermi_surface(
        "dwave",
        PairingAmplitudes(delta0_eV=0.04),
        nk=12,
        energy_tolerance_eV=0.15,
        node_tolerance_eV=1e-4,
    )
    large_tolerance = gap_statistics_on_fermi_surface(
        "dwave",
        PairingAmplitudes(delta0_eV=0.04),
        nk=12,
        energy_tolerance_eV=0.15,
        node_tolerance_eV=1e-2,
    )

    assert large_tolerance.approximate_nodes >= small_tolerance.approximate_nodes
