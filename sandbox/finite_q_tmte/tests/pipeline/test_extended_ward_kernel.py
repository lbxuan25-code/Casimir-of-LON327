from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.extended_ward_kernel import (
    extended_ward_candidate_result,
    extended_ward_candidates,
    extended_ward_kernel_payload,
    solve_left_collective_vector,
    solve_right_collective_vector,
)
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.frequency import frequency_payload


def _fake_blocks() -> TargetBareBlocks:
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi_eV=0.01)
    k_ss = np.asarray(
        [
            [1.0 + 0.2j, 2.0 - 0.3j, 0.5 + 0.1j],
            [3.0 + 0.4j, 4.0 + 0.0j, 0.7],
            [0.2 - 0.1j, 0.3, 5.0],
        ],
        dtype=complex,
    )
    k_seta = np.asarray([[0.2 + 0.1j, 0.4 - 0.2j], [0.1, 0.3 + 0.4j], [0.5j, 0.2]], dtype=complex)
    k_etas = np.asarray([[0.7 - 0.1j, 0.2 + 0.5j, 0.1], [0.3 + 0.2j, 0.8, 0.4 - 0.1j]], dtype=complex)
    k_etaeta = np.asarray([[2.0 + 0.0j, 0.25 - 0.1j], [0.4 + 0.2j, 3.0 + 0.0j]], dtype=complex)
    return TargetBareBlocks(
        source_order=("G", "TM", "TE"),
        conventions=conventions,
        k_ss_bubble=k_ss.copy(),
        k_ss_contact=np.zeros_like(k_ss),
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta_bubble=k_etaeta.copy(),
        k_etaeta_counterterm=np.zeros_like(k_etaeta),
        k_etaeta=k_etaeta,
        metadata={},
    )


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_extended_ward_kernel.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_extended_ward_kernel_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_collective_candidate_returns_raw_g_row_and_column():
    blocks = _fake_blocks()
    zero = np.zeros(2, dtype=complex)
    result = extended_ward_candidate_result(
        name="zero_collective",
        description="test",
        blocks=blocks,
        w_eta_left=zero,
        w_eta_right=zero,
        collective_order=("amplitude_eta1", "phase_eta2"),
        physical_matrix_norm=1.0,
        etaeta_norm=1.0,
    )
    left = np.asarray([item["value"] for item in result["left_em_residual"]])
    right = np.asarray([item["value"] for item in result["right_em_residual"]])
    np.testing.assert_allclose(left, blocks.k_ss[0, :])
    np.testing.assert_allclose(right, blocks.k_ss[:, 0])


def test_fitted_left_collective_equation_closes_collective_block():
    blocks = _fake_blocks()
    w_left, meta = solve_left_collective_vector(blocks.k_etaeta, blocks.k_seta[0, :])
    np.testing.assert_allclose(blocks.k_seta[0, :] + w_left @ blocks.k_etaeta, np.zeros(2), atol=1e-13)
    assert meta["solve_method"] == "solve"


def test_fitted_right_collective_equation_closes_collective_block():
    blocks = _fake_blocks()
    w_right, meta = solve_right_collective_vector(blocks.k_etaeta, blocks.k_etas[:, 0])
    np.testing.assert_allclose(blocks.k_etas[:, 0] + blocks.k_etaeta @ w_right, np.zeros(2), atol=1e-13)
    assert meta["solve_method"] == "solve"


def test_fitted_em_residuals_equal_schur_g_row_and_column():
    blocks = _fake_blocks()
    candidates, derived = extended_ward_candidates(
        blocks=blocks,
        delta0_eV=0.1,
        collective_order=("amplitude_eta1", "phase_eta2"),
    )
    by_name = {row["candidate"]: row for row in candidates}
    left = np.asarray([item["value"] for item in by_name["fitted_left_collective_equation"]["left_em_residual"]])
    right = np.asarray([item["value"] for item in by_name["fitted_right_collective_equation"]["right_em_residual"]])
    k_eff = derived["K_eff"]
    np.testing.assert_allclose(left, k_eff[0, :], atol=1e-13)
    np.testing.assert_allclose(right, k_eff[:, 0], atol=1e-13)
    consistency = derived["schur_consistency"]
    assert consistency["fitted_left_em_minus_schur_g_row_norm"] < 1e-13
    assert consistency["fitted_right_em_minus_schur_g_col_norm"] < 1e-13


def test_extended_ward_payload_is_debug_only():
    blocks = _fake_blocks()
    candidates, derived = extended_ward_candidates(
        blocks=blocks,
        delta0_eV=0.1,
        collective_order=("amplitude_eta1", "phase_eta2"),
    )
    payload = extended_ward_kernel_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"q_value": 0.02, "nk": 5},
        source_order=("G", "TM", "TE"),
        collective_order=("amplitude_eta1", "phase_eta2"),
        raw_ansatz_channel_names=("eta1", "eta2"),
        block_norm_payload={"valid_for_casimir_input": False},
        candidate_results=candidates,
        schur_consistency=derived["schur_consistency"],
        flags={"valid_for_casimir_input": False},
    )
    assert payload["schema_version"] == "finite_q_tmte_extended_ward_kernel_v1"
    assert payload["collective_order"] == ["amplitude_eta1", "phase_eta2"]
    assert payload["raw_ansatz_channel_names"] == ["eta1", "eta2"]
    assert payload["debug_parameters"]["debug_only_extended_ward_kernel"] is True
    assert payload["valid_for_casimir_input"] is False
    assert all(row["valid_for_casimir_input"] is False for row in payload["candidate_results"])


def test_extended_ward_cli_rejects_nonpositive_nk(tmp_path):
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


def test_extended_ward_cli_rejects_negative_matsubara_index(tmp_path):
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
