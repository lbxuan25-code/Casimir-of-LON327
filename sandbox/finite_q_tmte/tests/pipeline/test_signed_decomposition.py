from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.contact_ablation import scaled_contact_blocks
from sandbox.finite_q_tmte.tmte.pipeline.signed_decomposition import (
    signed_decomposition_from_blocks,
    signed_decomposition_payload,
    signed_entries,
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
    path = Path("sandbox/finite_q_tmte/scripts/debug_signed_decomposition.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_signed_decomposition_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_signed_entry_extraction_preserves_real_and_imaginary_parts():
    matrix = np.asarray([[1.0 + 2.0j, 3.0 - 4.0j, 5.0], [6.0 + 7.0j, 8.0, 9.0], [10.0 - 1.0j, 11.0, 12.0]], dtype=complex)
    entries = signed_entries({"K_eff": matrix})
    assert entries["GG"]["K_eff"] == 1.0 + 2.0j
    assert entries["GTM"]["K_eff"] == 3.0 - 4.0j
    assert entries["TMG"]["K_eff"] == 6.0 + 7.0j
    assert entries["GTE"]["K_eff"] == 5.0 + 0.0j
    assert entries["TEG"]["K_eff"] == 10.0 - 1.0j


def test_schur_correction_plus_keff_reconstructs_scaled_kss():
    payload = signed_decomposition_from_blocks(
        blocks=scaled_contact_blocks(_fake_blocks(), 0.5),
        contact_scale=0.5,
        shifted_payload={"average_order": "average_blocks_then_schur", "valid_for_casimir_input": False},
    )
    for entry in ("GG", "GTM", "TMG", "TMTM", "GTE", "TEG"):
        row = payload["entries"][entry]
        np.testing.assert_allclose(row["Schur_correction"] + row["K_eff"], row["K_SS_scaled"])


def test_signed_decomposition_contact_scale_changes_only_contact_and_kss():
    blocks = _fake_blocks()
    scaled = scaled_contact_blocks(blocks, -1.0)
    np.testing.assert_allclose(scaled.k_ss_contact, -blocks.k_ss_contact)
    np.testing.assert_allclose(scaled.k_ss, blocks.k_ss_bubble - blocks.k_ss_contact)
    np.testing.assert_allclose(scaled.k_seta, blocks.k_seta)
    np.testing.assert_allclose(scaled.k_etas, blocks.k_etas)
    np.testing.assert_allclose(scaled.k_etaeta, blocks.k_etaeta)


def test_signed_decomposition_schema_is_debug_only_and_not_casimir_ready():
    payload = signed_decomposition_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"contact_scale": 1.0, "average_order": "average_blocks_then_schur"},
        entries={"GG": {"K_eff": 1.0 + 0.0j}},
        schur={"solve_method": "solve"},
        ratios={"gauge_row_norm": 0.0},
    )
    assert payload["debug_parameters"]["debug_only_signed_decomposition"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["status"]["valid_for_casimir_input"] is False
    assert "ward_closed" not in payload
    assert "casimir_ready" not in payload


def test_signed_decomposition_cli_rejects_nonpositive_nk(tmp_path):
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


def test_signed_decomposition_cli_rejects_negative_matsubara_index(tmp_path):
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

