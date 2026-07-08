from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.ward_basis_convention_scan import (
    default_candidate_names,
    primitive_blocks_from_baseline,
    scan_candidate,
    target_blocks_from_primitive,
    target_transform_matrix,
    ward_basis_convention_scan_payload,
)
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.frequency import frequency_payload


def _fake_blocks() -> TargetBareBlocks:
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi_eV=0.01)
    k_ss_bubble = np.asarray(
        [
            [1.0 + 0.2j, 2.0 - 0.3j, 0.5 + 0.1j],
            [3.0 + 0.4j, 4.0 + 0.0j, 0.7],
            [0.2 - 0.1j, 0.3, 5.0],
        ],
        dtype=complex,
    )
    k_ss_contact = np.asarray(
        [
            [0.1 - 0.05j, 0.2 + 0.03j, 0.0],
            [0.4 - 0.02j, 0.1, 0.0],
            [0.0, 0.0, 0.05],
        ],
        dtype=complex,
    )
    k_ss = k_ss_bubble + k_ss_contact
    k_seta = np.asarray([[0.2 + 0.1j, 0.4 - 0.2j], [0.1, 0.3 + 0.4j], [0.5j, 0.2]], dtype=complex)
    k_etas = np.asarray([[0.7 - 0.1j, 0.2 + 0.5j, 0.1], [0.3 + 0.2j, 0.8, 0.4 - 0.1j]], dtype=complex)
    k_etaeta = np.asarray([[2.0 + 0.0j, 0.25 - 0.1j], [0.4 + 0.2j, 3.0 + 0.0j]], dtype=complex)
    return TargetBareBlocks(
        source_order=("G", "TM", "TE"),
        conventions=conventions,
        k_ss_bubble=k_ss_bubble,
        k_ss_contact=k_ss_contact,
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta_bubble=k_etaeta.copy(),
        k_etaeta_counterterm=np.zeros_like(k_etaeta),
        k_etaeta=k_etaeta,
        metadata={},
    )


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_ward_basis_convention_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_ward_basis_convention_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_candidates_include_only_diagnostic_names():
    names = default_candidate_names()
    assert "baseline_real" in names
    assert "temporal_i_plus" in names
    assert "temporal_i_minus" in names
    assert all(isinstance(name, str) for name in names)


def test_candidate_transform_shapes_and_nonzero_determinants():
    for name in default_candidate_names():
        matrix = target_transform_matrix(name, xi_eV=0.01, q_norm=0.2)
        assert matrix.shape == (3, 3)
        assert abs(np.linalg.det(matrix)) > 0.0


def test_baseline_primitive_roundtrip_reconstructs_target_blocks():
    blocks = _fake_blocks()
    primitive = primitive_blocks_from_baseline(blocks)
    transform = target_transform_matrix("baseline_real", xi_eV=blocks.conventions.g0, q_norm=blocks.conventions.gL)
    reconstructed = target_blocks_from_primitive(blocks, primitive, transform)
    np.testing.assert_allclose(reconstructed.k_ss, blocks.k_ss, atol=1e-13)
    np.testing.assert_allclose(reconstructed.k_ss_bubble, blocks.k_ss_bubble, atol=1e-13)
    np.testing.assert_allclose(reconstructed.k_ss_contact, blocks.k_ss_contact, atol=1e-13)
    np.testing.assert_allclose(reconstructed.k_seta, blocks.k_seta, atol=1e-13)
    np.testing.assert_allclose(reconstructed.k_etas, blocks.k_etas, atol=1e-13)


def test_scan_candidate_is_not_accepted_and_reports_phase_fingerprint():
    blocks = _fake_blocks()
    primitive = primitive_blocks_from_baseline(blocks)
    result = scan_candidate(
        name="baseline_real",
        baseline_blocks=blocks,
        primitive=primitive,
        delta0_eV=0.1,
        collective_order=("amplitude_eta1", "phase_eta2"),
        ratio_eps=1e-30,
    )
    assert result["status"]["diagnostic_only_not_a_fix"] is True
    assert result["status"]["accepted_convention"] is False
    assert result["status"]["requires_analytic_derivation_before_convention_change"] is True
    assert result["valid_for_casimir_input"] is False
    assert result["phase_eta2_primitive_fingerprint"]["primitive_order"] == ["A0", "L", "T"]
    assert result["phase_eta2_primitive_fingerprint"]["target_order"] == ["G", "TM", "TE"]
    assert "extended_ward_candidates" in result


def test_payload_is_diagnostic_only_and_has_no_winner():
    payload = ward_basis_convention_scan_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"q_value": 0.02, "nk": 5},
        collective_order=("amplitude_eta1", "phase_eta2"),
        raw_ansatz_channel_names=("eta1", "eta2"),
        primitive_metadata={"valid_for_casimir_input": False},
        candidates=[],
    )
    assert payload["schema_version"] == "finite_q_tmte_ward_basis_convention_scan_v1"
    assert payload["status"]["diagnostic_only_not_a_fix"] is True
    assert payload["status"]["accepted_convention"] is False
    assert payload["status"]["requires_analytic_derivation_before_convention_change"] is True
    assert "winner" not in payload
    assert payload["valid_for_casimir_input"] is False


def test_ward_basis_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
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


def test_ward_basis_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
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
