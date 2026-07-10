from __future__ import annotations

import numpy as np
import pytest

from lno327.constants import KB_EV_PER_K
from validation.run_positive_matsubara_point_scan import (
    _annotate_convergence,
    _run_task,
    matsubara_energy_eV,
)


def test_matsubara_energy_uses_positive_bosonic_frequency():
    value = matsubara_energy_eV(3, 10.0)
    assert value == pytest.approx(2.0 * np.pi * 3.0 * KB_EV_PER_K * 10.0)
    with pytest.raises(ValueError, match="positive Matsubara index"):
        matsubara_energy_eV(0, 10.0)


def test_positive_matsubara_point_scan_batches_frequencies_and_reports_contracts():
    rows = _run_task(
        {
            "nk": 2,
            "pairing": "spm",
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "degeneracy": 1.0,
            "separation_nm": 20.0,
            "ward_tolerance": 1e-6,
            "ward_absolute_tolerance": 1e-12,
            "condition_max": 1e12,
            "matsubara_indices": (1, 2),
        }
    )

    assert len(rows) == 2
    assert [row["matsubara_index"] for row in rows] == [1, 2]
    assert rows[1]["xi_eV"] == pytest.approx(2.0 * rows[0]["xi_eV"])
    for row in rows:
        assert row["nk"] == 2
        assert row["num_k_points"] == 4
        assert row["xi_eV"] > 0.0
        assert row["schur_inverse_method"] == "inv"
        assert np.isfinite(row["schur_condition_number"])
        assert row["sheet_finite"] is True
        assert np.isfinite(row["sigma_tilde_frobenius_norm"])
        assert isinstance(row["single_point_pipeline_passed"], bool)
        assert isinstance(row["reflection_error"], str)
        assert isinstance(row["logdet_error"], str)

    _annotate_convergence(rows)
    for row in rows:
        assert row["convergence_reference_nk"] == 2
        assert row["relative_sigma_tilde_to_reference"] == pytest.approx(0.0)
        if row["reflection_constructed"]:
            assert row["relative_reflection_to_reference"] == pytest.approx(0.0)
        if row["logdet_passed"]:
            assert row["absolute_logdet_to_reference"] == pytest.approx(0.0)
            assert row["relative_logdet_to_reference"] == pytest.approx(0.0)
