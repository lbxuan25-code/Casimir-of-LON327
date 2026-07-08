from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lno327.models.symmetry_bdg_2band.collective import SymmetryTwoBandPairingAmplitudes, build_pairing_ansatz

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.block_builder import build_effective_from_blocks
from sandbox.finite_q_tmte.tmte.pipeline.endpoint_collective_basis import (
    amplitude_a_vertex,
    amplitude_endpoint_vertex,
    amplitude_s_vertex,
    embedded_counterterm,
    endpoint_collective_basis_payload,
    endpoint_collective_vertices,
    endpoint_form_factors,
    endpoint_basis_result,
    phase_a_vertex,
    phase_endpoint_vertex,
    phase_s_vertex,
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
        metadata={"collective_order": ["amplitude_eta1", "phase_eta2"]},
    )


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_endpoint_collective_basis.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_endpoint_collective_basis_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_endpoint_decomposition_reconstructs_endpoint_vertices():
    ansatz = build_pairing_ansatz("dwave")
    amp = SymmetryTwoBandPairingAmplitudes(delta0_eV=0.1)
    phi_minus, phi_plus, phi_s, phi_a = endpoint_form_factors(ansatz, 0.4, -0.3, 0.2, 0.0, amp)
    np.testing.assert_allclose(amplitude_s_vertex(phi_s) + amplitude_a_vertex(phi_a), amplitude_endpoint_vertex(phi_minus, phi_plus))
    np.testing.assert_allclose(phase_s_vertex(phi_s) + phase_a_vertex(phi_a), phase_endpoint_vertex(phi_minus, phi_plus))


def test_s_only_2ch_matches_current_ansatz_collective_vertices():
    ansatz = build_pairing_ansatz("dwave")
    amp = SymmetryTwoBandPairingAmplitudes(delta0_eV=0.1)
    current, current_order = endpoint_collective_vertices(ansatz=ansatz, kx=0.4, ky=-0.3, qx=0.2, qy=0.0, pairing_params=amp, basis_mode="current_2ch")
    s_only, s_order = endpoint_collective_vertices(ansatz=ansatz, kx=0.4, ky=-0.3, qx=0.2, qy=0.0, pairing_params=amp, basis_mode="s_only_2ch")
    assert current_order == ("amplitude_eta1", "phase_eta2")
    assert s_order == ("amplitude_s", "phase_s")
    for left, right in zip(current, s_only, strict=True):
        np.testing.assert_allclose(left, right)


def test_expanded_4ch_collective_order():
    ansatz = build_pairing_ansatz("dwave")
    amp = SymmetryTwoBandPairingAmplitudes(delta0_eV=0.1)
    vertices, order = endpoint_collective_vertices(ansatz=ansatz, kx=0.4, ky=-0.3, qx=0.2, qy=0.0, pairing_params=amp, basis_mode="expanded_4ch")
    assert len(vertices) == 4
    assert order == ("amplitude_s", "phase_s", "amplitude_a", "phase_a")


def test_counterterm_embedding_puts_existing_block_in_symmetric_channels_and_zeroes_a_channels():
    existing = np.asarray([[1.0 + 0.1j, 2.0], [3.0, 4.0 - 0.2j]], dtype=complex)
    embedded = embedded_counterterm(existing, ("amplitude_s", "phase_s", "amplitude_a", "phase_a"))
    np.testing.assert_allclose(embedded[:2, :2], existing)
    np.testing.assert_allclose(embedded[2:, :], 0.0)
    np.testing.assert_allclose(embedded[:, 2:], 0.0)


def test_current_2ch_endpoint_result_reproduces_existing_two_channel_schur_for_fake_blocks():
    response = build_effective_from_blocks(_fake_blocks())
    result = endpoint_basis_result(basis_mode="current_2ch", response=response, counterterm_policy="embed_existing_s_counterterm_zero_a_counterterm")
    np.testing.assert_allclose(result["Schur_correction_entries"]["GTM"], response.schur.correction[0, 1])
    np.testing.assert_allclose(result["K_eff_entries"]["GTM"], response.schur.effective[0, 1])
    assert result["diagnostics"]["schur_solve_method"] == response.schur.solve_method


def test_endpoint_collective_payload_is_debug_only_and_not_casimir_ready():
    response = build_effective_from_blocks(_fake_blocks())
    result = endpoint_basis_result(basis_mode="current_2ch", response=response, counterterm_policy="embed_existing_s_counterterm_zero_a_counterterm")
    payload = endpoint_collective_basis_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"q_value": 0.02, "nk": 5, "contact_scale": 1.0},
        basis_results=[result],
    )
    assert payload["schema_version"] == "finite_q_tmte_endpoint_collective_basis_v1"
    assert payload["debug_parameters"]["debug_only_endpoint_collective_basis"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["basis_results"][0]["valid_for_casimir_input"] is False
    assert payload["basis_results"][0]["diagnostics"]["valid_for_casimir_input"] is False
    assert payload["basis_results"][0]["schur"]["valid_for_casimir_input"] is False


def test_endpoint_collective_cli_rejects_nonpositive_nk(tmp_path):
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


def test_endpoint_collective_cli_rejects_negative_matsubara_index(tmp_path):
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
