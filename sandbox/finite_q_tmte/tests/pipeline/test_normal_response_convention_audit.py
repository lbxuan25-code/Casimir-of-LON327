from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.normal_response_convention_audit import (
    DEFAULT_CANDIDATE_NAMES,
    SCHEMA_VERSION,
    band_vertex,
    candidate_specs_from_names,
    full_grid_candidate_specs,
    kubo_factor,
    scalar_projection,
    ward_vectors,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_normal_response_convention_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_normal_response_convention_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_and_default_candidates():
    assert SCHEMA_VERSION == "finite_q_tmte_normal_response_convention_audit_v1"
    assert "baseline_current" in DEFAULT_CANDIDATE_NAMES
    candidates = candidate_specs_from_names()
    assert candidates[0]["name"] == "baseline_current"


def test_candidate_specs_from_names_selects_requested_order():
    candidates = candidate_specs_from_names(["kubo_fully_reversed", "baseline_current"])
    assert [row["name"] for row in candidates] == ["kubo_fully_reversed", "baseline_current"]


def test_candidate_specs_from_names_rejects_unknown():
    with pytest.raises(ValueError):
        candidate_specs_from_names(["missing"])


def test_full_grid_candidate_specs_nonempty_and_named():
    rows = full_grid_candidate_specs()
    assert rows
    assert rows[0]["name"].startswith("grid_")


def test_ward_vectors_standard_and_alternates():
    q = np.asarray([0.3, -0.4])
    left, right = ward_vectors(0.2, q, "standard")
    np.testing.assert_allclose(left, [0.2j, 0.3, -0.4])
    np.testing.assert_allclose(right, [0.2j, -0.3, 0.4])
    left_alt, right_alt = ward_vectors(0.2, q, "right_spatial_plus")
    np.testing.assert_allclose(left_alt, left)
    np.testing.assert_allclose(right_alt, [0.2j, 0.3, -0.4])


def test_kubo_factor_conventions():
    kwargs = dict(energy_minus=1.0, energy_plus=1.5, occupation_minus=0.8, occupation_plus=0.2, xi_eV=0.1)
    current = kubo_factor(**kwargs, convention="minus_plus")
    denom_flipped = kubo_factor(**kwargs, convention="denominator_flipped")
    fully_reversed = kubo_factor(**kwargs, convention="fully_reversed")
    np.testing.assert_allclose(current, 0.6 / (0.1j - 0.5))
    np.testing.assert_allclose(denom_flipped, 0.6 / (0.1j + 0.5))
    np.testing.assert_allclose(fully_reversed, -0.6 / (0.1j + 0.5))


def test_band_vertex_orientations_identity_states():
    states_minus = np.eye(2, dtype=complex)
    states_plus = np.eye(2, dtype=complex)
    vertex = np.asarray([[1.0, 2.0 + 1.0j], [3.0 - 0.5j, 4.0]], dtype=complex)
    np.testing.assert_allclose(band_vertex(states_minus, vertex, states_plus, "forward_minus_plus"), vertex.T)
    np.testing.assert_allclose(band_vertex(states_minus, vertex, states_plus, "direct_minus_plus"), vertex)


def test_scalar_projection_exact_parallel_vector():
    current = np.asarray([1.0, 2.0j, -1.0], dtype=complex)
    alpha = -0.2 + 0.3j
    report = scalar_projection(alpha * current, current)
    np.testing.assert_allclose(report["alpha_required_over_current"], alpha)
    assert report["residual_norm"] < 1e-12


def test_normal_response_convention_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--matsubara-index",
                "1",
                "--q",
                "0.02",
                "--nk",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_normal_response_convention_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--matsubara-index",
                "-1",
                "--q",
                "0.02",
                "--nk",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )
