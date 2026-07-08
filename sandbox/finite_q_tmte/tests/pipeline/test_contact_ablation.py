from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.contact_ablation import (
    SCHEMA_VERSION,
    ablation_ratios,
    contact_scale_result,
    scaled_contact_blocks,
)
from sandbox.finite_q_tmte.tmte.pipeline.block_builder import build_effective_from_blocks
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions


def _fake_blocks() -> TargetBareBlocks:
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi_eV=0.01)
    k_ss_bubble = np.asarray([[1.0, 0.2, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=complex)
    k_ss_contact = np.asarray([[0.5, 0.1, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 0.7]], dtype=complex)
    k_seta = np.asarray([[0.1, 0.0], [0.0, 0.2], [0.0, 0.0]], dtype=complex)
    k_etas = np.asarray([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=complex)
    k_etaeta = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=complex)
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
    path = Path("sandbox/finite_q_tmte/scripts/debug_contact_ablation.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_contact_ablation_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scaled_contact_blocks_only_changes_contact_and_kss():
    blocks = _fake_blocks()
    scale_one = scaled_contact_blocks(blocks, 1.0)
    scale_zero = scaled_contact_blocks(blocks, 0.0)
    scale_flip = scaled_contact_blocks(blocks, -1.0)
    np.testing.assert_allclose(scale_one.k_ss, blocks.k_ss)
    np.testing.assert_allclose(scale_zero.k_ss, blocks.k_ss_bubble)
    np.testing.assert_allclose(scale_flip.k_ss, blocks.k_ss_bubble - blocks.k_ss_contact)
    for scaled in (scale_zero, scale_flip):
        np.testing.assert_allclose(scaled.k_seta, blocks.k_seta)
        np.testing.assert_allclose(scaled.k_etas, blocks.k_etas)
        np.testing.assert_allclose(scaled.k_etaeta, blocks.k_etaeta)


def test_contact_ablation_ratios_for_fake_effective_matrix():
    elements = {"K_TMTM": 5.0 + 0.0j}
    ratios = ablation_ratios({"G_row_norm": 10.0, "GG_abs": 2.5, "physical_matrix_norm": 20.0}, elements, eps=1e-30)
    assert ratios["gauge_over_physical"] == 0.5
    assert ratios["gauge_over_tm_abs"] == 2.0
    assert ratios["gauge_gg_over_tm_abs"] == 0.5
    assert ratios["valid_for_casimir_input"] is False


def test_contact_ablation_scale_result_schema_is_debug_only():
    response = build_effective_from_blocks(scaled_contact_blocks(_fake_blocks(), 0.0))
    payload = contact_scale_result(
        contact_scale=0.0,
        response=response,
        shifted_payload={"average_order": "average_blocks_then_schur", "valid_for_casimir_input": False},
    )
    top = {
        "schema_version": SCHEMA_VERSION,
        "contact_scale_results": [payload],
        "valid_for_casimir_input": False,
    }
    assert top["valid_for_casimir_input"] is False
    assert payload["debug_only_contact_ablation"] is True
    assert payload["valid_for_casimir_input"] is False
    assert "ward_closed" not in payload
    assert "casimir_ready" not in payload


def test_contact_ablation_cli_accepts_multiple_contact_scales(tmp_path):
    module = _load_debug_script()
    args = module.build_parser().parse_args(
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
            "5",
            "--contact-scales",
            "1.0",
            "0.0",
            "-1.0",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.contact_scales == [1.0, 0.0, -1.0]


def test_contact_ablation_cli_rejects_nonpositive_nk(tmp_path):
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


def test_contact_ablation_cli_rejects_negative_matsubara_index(tmp_path):
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

