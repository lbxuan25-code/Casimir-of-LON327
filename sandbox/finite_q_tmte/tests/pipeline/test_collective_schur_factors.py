from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.collective_schur_factors import (
    collective_schur_factors_from_blocks,
    collective_schur_factors_payload,
    collective_order_from_ansatz,
    friendly_collective_order,
    schur_factor_decomposition,
    solve_collective_action,
)
from sandbox.finite_q_tmte.tmte.pipeline.contact_ablation import scaled_contact_blocks
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
    path = Path("sandbox/finite_q_tmte/scripts/debug_collective_schur_factors.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_collective_schur_factors_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _NamedAnsatz:
    channel_names = ("eta1", "eta2")


class _CustomNamedAnsatz:
    channel_names = ("density", "phase")


def test_collective_labels_prefer_ansatz_channel_names():
    labels, raw = collective_order_from_ansatz(_CustomNamedAnsatz(), 2)
    assert labels == ("density", "phase")
    assert raw == ("density", "phase")


def test_known_eta1_eta2_labels_map_to_friendly_amplitude_phase_names():
    labels, raw = collective_order_from_ansatz(_NamedAnsatz(), 2)
    assert labels == ("amplitude_eta1", "phase_eta2")
    assert raw == ("eta1", "eta2")
    assert friendly_collective_order(None, 3) == ("eta0", "eta1", "eta2")


def test_per_channel_products_reconstruct_schur_correction_entries():
    blocks = _fake_blocks()
    x_action, _ = solve_collective_action(blocks.k_etaeta, blocks.k_etas)
    correction = blocks.k_seta @ x_action
    decomposition = schur_factor_decomposition(blocks.k_seta, x_action)

    indices = {"G": 0, "TM": 1, "TE": 2}
    for entry, row_label, col_label in (("GG", "G", "G"), ("GTM", "G", "TM"), ("TMG", "TM", "G"), ("TMTM", "TM", "TM"), ("GTE", "G", "TE"), ("TEG", "TE", "G")):
        products = [item["product"] for item in decomposition[entry]["contributions"]]
        np.testing.assert_allclose(sum(products, 0.0 + 0.0j), correction[indices[row_label], indices[col_label]])
        assert decomposition[entry]["reconstruction_error"] < 1e-14


def test_collective_action_uses_solve_for_schur_construction():
    blocks = _fake_blocks()
    x_action, schur = solve_collective_action(blocks.k_etaeta, blocks.k_etas)
    np.testing.assert_allclose(x_action, np.linalg.solve(blocks.k_etaeta, blocks.k_etas))
    np.testing.assert_allclose(blocks.k_seta @ x_action, blocks.k_seta @ np.linalg.solve(blocks.k_etaeta, blocks.k_etas))
    assert schur["solve_method"] == "solve"
    assert schur["numerically_suspect"] is False


def test_collective_contact_scale_only_changes_scaled_contact_and_kss():
    blocks = _fake_blocks()
    scaled = scaled_contact_blocks(blocks, 0.75)
    np.testing.assert_allclose(scaled.k_ss_contact, 0.75 * blocks.k_ss_contact)
    np.testing.assert_allclose(scaled.k_ss, blocks.k_ss_bubble + 0.75 * blocks.k_ss_contact)
    np.testing.assert_allclose(scaled.k_seta, blocks.k_seta)
    np.testing.assert_allclose(scaled.k_etas, blocks.k_etas)
    np.testing.assert_allclose(scaled.k_etaeta, blocks.k_etaeta)


def test_collective_output_schema_is_debug_only_and_not_casimir_ready():
    pieces = collective_schur_factors_from_blocks(
        blocks=_fake_blocks(),
        contact_scale=1.0,
        shifted_payload={"average_order": "average_blocks_then_schur", "valid_for_casimir_input": False},
    )
    payload = collective_schur_factors_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"q_value": 0.02, "nk": 5, "contact_scale": 1.0},
        matrices=pieces["matrices"],
        schur_factor_decomposition_payload=pieces["schur_factor_decomposition"],
        consistency_diagnostics_payload=pieces["consistency_diagnostics"],
        ratios=pieces["ratios"],
        schur=pieces["schur"],
        collective_order=("amplitude_eta1", "phase_eta2"),
        raw_ansatz_channel_names=("eta1", "eta2"),
    )

    assert payload["schema_version"] == "finite_q_tmte_collective_schur_factors_v1"
    assert payload["collective_order"] == ["amplitude_eta1", "phase_eta2"]
    assert payload["raw_ansatz_channel_names"] == ["eta1", "eta2"]
    assert payload["debug_parameters"]["debug_only_collective_schur_factors"] is True
    assert payload["debug_parameters"]["average_order"] == "average_blocks_then_schur"
    assert payload["valid_for_casimir_input"] is False
    assert payload["status"]["valid_for_casimir_input"] is False
    assert payload["schur"]["valid_for_casimir_input"] is False
    assert payload["ratios"]["valid_for_casimir_input"] is False


def test_collective_schur_cli_rejects_nonpositive_nk(tmp_path):
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


def test_collective_schur_cli_rejects_negative_matsubara_index(tmp_path):
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
