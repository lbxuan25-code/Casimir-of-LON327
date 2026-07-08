from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.phase_eta2_convention import (
    apply_phase_eta2_transform,
    phase_eta2_convention_payload,
    phase_transform_results,
)
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.frequency import frequency_payload


def _fake_blocks() -> TargetBareBlocks:
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi_eV=0.01)
    k_ss_bubble = np.asarray(
        [[1.0 + 1.0j, 2.0 - 3.0j, 4.0 + 0.5j], [5.0 + 2.0j, 6.0 + 0.0j, 7.0], [8.0 - 1.0j, 9.0, 10.0]],
        dtype=complex,
    )
    k_ss_contact = np.asarray(
        [[0.5 - 0.25j, 0.25 + 0.5j, 0.0], [0.1, 0.2, 0.0], [0.0, 0.0, 0.3]],
        dtype=complex,
    )
    k_seta = np.asarray([[1.0 + 0.5j, 2.0 - 0.25j], [3.0 - 1.0j, 4.0 + 0.5j], [0.5, 1.0j]], dtype=complex)
    k_etas = np.asarray([[1.0, 2.0 + 1.0j, 3.0], [4.0 - 1.0j, 5.0, 0.0 + 6.0j]], dtype=complex)
    k_etaeta = np.asarray([[2.0, 0.5], [0.25, 3.0]], dtype=complex)
    return TargetBareBlocks(
        source_order=("G", "TM", "TE"),
        conventions=conventions,
        k_ss_bubble=k_ss_bubble,
        k_ss_contact=k_ss_contact,
        k_ss=k_ss_bubble + k_ss_contact,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta_bubble=k_etaeta,
        k_etaeta_counterterm=np.zeros_like(k_etaeta),
        k_etaeta=k_etaeta,
        metadata={},
    )


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_phase_eta2_convention.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_phase_eta2_convention_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_identity_transform_leaves_collective_blocks_unchanged():
    blocks = _fake_blocks()
    seta, etas, etaeta, changed = apply_phase_eta2_transform(transform="identity", k_seta=blocks.k_seta, k_etas=blocks.k_etas, k_etaeta=blocks.k_etaeta)
    np.testing.assert_allclose(seta, blocks.k_seta)
    np.testing.assert_allclose(etas, blocks.k_etas)
    np.testing.assert_allclose(etaeta, blocks.k_etaeta)
    assert changed == []


def test_phase_eta2_seta_sign_flip_only_flips_seta_column_one():
    blocks = _fake_blocks()
    seta, etas, etaeta, changed = apply_phase_eta2_transform(transform="phase_eta2_seta_sign_flip", k_seta=blocks.k_seta, k_etas=blocks.k_etas, k_etaeta=blocks.k_etaeta)
    expected = blocks.k_seta.copy()
    expected[:, 1] *= -1
    np.testing.assert_allclose(seta, expected)
    np.testing.assert_allclose(etas, blocks.k_etas)
    np.testing.assert_allclose(etaeta, blocks.k_etaeta)
    assert changed == ["K_Seta"]


def test_phase_eta2_etas_sign_flip_only_flips_etas_row_one():
    blocks = _fake_blocks()
    seta, etas, etaeta, changed = apply_phase_eta2_transform(transform="phase_eta2_etas_sign_flip", k_seta=blocks.k_seta, k_etas=blocks.k_etas, k_etaeta=blocks.k_etaeta)
    expected = blocks.k_etas.copy()
    expected[1, :] *= -1
    np.testing.assert_allclose(seta, blocks.k_seta)
    np.testing.assert_allclose(etas, expected)
    np.testing.assert_allclose(etaeta, blocks.k_etaeta)
    assert changed == ["K_etaS"]


def test_phase_eta2_both_sign_flip_leaves_kernel_unchanged():
    blocks = _fake_blocks()
    seta, etas, etaeta, changed = apply_phase_eta2_transform(transform="phase_eta2_both_sign_flip", k_seta=blocks.k_seta, k_etas=blocks.k_etas, k_etaeta=blocks.k_etaeta)
    expected_seta = blocks.k_seta.copy()
    expected_etas = blocks.k_etas.copy()
    expected_seta[:, 1] *= -1
    expected_etas[1, :] *= -1
    np.testing.assert_allclose(seta, expected_seta)
    np.testing.assert_allclose(etas, expected_etas)
    np.testing.assert_allclose(etaeta, blocks.k_etaeta)
    assert changed == ["K_Seta", "K_etaS"]


def test_phase_eta2_kernel_sign_flip_only_flips_etaeta_row_and_column_one():
    blocks = _fake_blocks()
    seta, etas, etaeta, changed = apply_phase_eta2_transform(transform="phase_eta2_kernel_sign_flip", k_seta=blocks.k_seta, k_etas=blocks.k_etas, k_etaeta=blocks.k_etaeta)
    expected = blocks.k_etaeta.copy()
    expected[1, :] *= -1
    expected[:, 1] *= -1
    np.testing.assert_allclose(seta, blocks.k_seta)
    np.testing.assert_allclose(etas, blocks.k_etas)
    np.testing.assert_allclose(etaeta, expected)
    assert changed == ["K_etaeta"]


def test_phase_transform_results_remain_debug_only_and_not_casimir_ready():
    blocks = _fake_blocks()
    results = phase_transform_results(blocks=blocks, transforms=("identity", "phase_eta2_seta_sign_flip"))
    payload = phase_eta2_convention_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"q_value": 0.02, "nk": 5, "contact_scale": 1.0},
        collective_order=("amplitude_eta1", "phase_eta2"),
        raw_ansatz_channel_names=("eta1", "eta2"),
        transform_results=results,
    )
    assert payload["schema_version"] == "finite_q_tmte_phase_eta2_convention_v1"
    assert payload["collective_order"] == ["amplitude_eta1", "phase_eta2"]
    assert payload["raw_ansatz_channel_names"] == ["eta1", "eta2"]
    assert payload["valid_for_casimir_input"] is False
    for result in payload["transform_results"]:
        assert result["valid_for_casimir_input"] is False
        assert result["diagnostics"]["valid_for_casimir_input"] is False
        assert result["schur"]["valid_for_casimir_input"] is False


def test_phase_eta2_convention_cli_rejects_nonpositive_nk(tmp_path):
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


def test_phase_eta2_convention_cli_rejects_negative_matsubara_index(tmp_path):
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
