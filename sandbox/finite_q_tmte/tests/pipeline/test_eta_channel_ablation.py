from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import TargetBareBlocks
from sandbox.finite_q_tmte.tmte.pipeline.eta_channel_ablation import (
    eta_channel_ablation_from_blocks,
    eta_channel_ablation_payload,
    eta_channel_mode_result,
    eta_channel_mode_results,
)
from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.frequency import frequency_payload


def _fake_blocks(*, offdiagonal_etaeta: bool = True) -> TargetBareBlocks:
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
    k_etaeta = np.asarray([[2.0, 0.5], [0.25, 3.0]], dtype=complex) if offdiagonal_etaeta else np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=complex)
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


def _mode(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(result for result in results if result["mode"] == name)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_eta_channel_ablation.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_eta_channel_ablation_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _complex_entries(entries: dict[str, complex], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([entries[name] for name in names], dtype=complex)


def test_no_schur_keff_entries_equal_scaled_kss_entries():
    blocks = _fake_blocks()
    result = eta_channel_mode_result(
        mode="no_schur",
        k_ss_scaled=blocks.k_ss,
        k_seta=blocks.k_seta,
        k_etas=blocks.k_etas,
        k_etaeta=blocks.k_etaeta,
        channel_indices=(),
        source_order=blocks.source_order,
    )
    assert result["schur"]["solve_method"] == "none"
    np.testing.assert_allclose(result["K_eff_entries"]["GG"], blocks.k_ss[0, 0])
    np.testing.assert_allclose(result["K_eff_entries"]["GTM"], blocks.k_ss[0, 1])
    np.testing.assert_allclose(result["K_eff_entries"]["TMG"], blocks.k_ss[1, 0])
    np.testing.assert_allclose(result["K_eff_entries"]["TMTM"], blocks.k_ss[1, 1])


def test_single_channel_schur_sum_reconstructs_full_schur_for_diagonal_etaeta():
    blocks = _fake_blocks(offdiagonal_etaeta=False)
    results = eta_channel_mode_results(blocks=blocks, collective_order=("amplitude_eta1", "phase_eta2"))
    eta0 = _mode(results, "amplitude_eta1_only")
    eta1 = _mode(results, "phase_eta2_only")
    all_channels = _mode(results, "eta_all")
    names = ("GG", "GTM", "TMG", "TMTM")
    summed = _complex_entries(eta0["Schur_correction_entries"], names) + _complex_entries(eta1["Schur_correction_entries"], names)
    np.testing.assert_allclose(summed, _complex_entries(all_channels["Schur_correction_entries"], names))


def test_sliced_schur_uses_sliced_eta_blocks_consistently():
    blocks = _fake_blocks()
    result = eta_channel_mode_result(
        mode="phase_eta2_only",
        k_ss_scaled=blocks.k_ss,
        k_seta=blocks.k_seta,
        k_etas=blocks.k_etas,
        k_etaeta=blocks.k_etaeta,
        channel_indices=(1,),
        collective_order=("amplitude_eta1", "phase_eta2"),
        legacy_mode="eta1_only",
        source_order=blocks.source_order,
    )
    manual_x = np.linalg.solve(blocks.k_etaeta[np.ix_([1], [1])], blocks.k_etas[[1], :])
    manual_correction = blocks.k_seta[:, [1]] @ manual_x
    np.testing.assert_allclose(result["Schur_correction_entries"]["GG"], manual_correction[0, 0])
    np.testing.assert_allclose(result["Schur_correction_entries"]["GTM"], manual_correction[0, 1])
    assert result["diagnostics"]["etaeta_condition_number"] == pytest.approx(float(np.linalg.cond(blocks.k_etaeta[np.ix_([1], [1])])))
    assert result["legacy_mode"] == "eta1_only"
    assert result["included_collective_channels"] == ["phase_eta2"]


def test_eta_channel_mode_names_use_corrected_collective_labels():
    blocks = _fake_blocks()
    results = eta_channel_mode_results(blocks=blocks, collective_order=("amplitude_eta1", "phase_eta2"))
    assert [result["mode"] for result in results] == ["no_schur", "amplitude_eta1_only", "phase_eta2_only", "eta_all"]
    assert _mode(results, "amplitude_eta1_only")["legacy_mode"] == "eta0_only"
    assert _mode(results, "phase_eta2_only")["legacy_mode"] == "eta1_only"


def test_eta_channel_payload_is_debug_only_and_not_casimir_ready():
    pieces = eta_channel_ablation_from_blocks(
        blocks=_fake_blocks(),
        contact_scale=1.0,
        shifted_payload={"average_order": "average_blocks_then_schur", "valid_for_casimir_input": False},
    )
    payload = eta_channel_ablation_payload(
        model_name="symmetry_bdg_2band",
        pairing_name="dwave",
        frequency=frequency_payload(1, 10.0),
        debug_parameters={"q_value": 0.02, "nk": 5, "contact_scale": 1.0},
        mode_results=pieces["mode_results"],
        collective_order=("amplitude_eta1", "phase_eta2"),
        raw_ansatz_channel_names=("eta1", "eta2"),
    )
    assert payload["schema_version"] == "finite_q_tmte_eta_channel_ablation_v1"
    assert payload["collective_order"] == ["amplitude_eta1", "phase_eta2"]
    assert payload["raw_ansatz_channel_names"] == ["eta1", "eta2"]
    assert payload["debug_parameters"]["debug_only_eta_channel_ablation"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["status"]["valid_for_casimir_input"] is False
    for result in payload["mode_results"]:
        assert result["valid_for_casimir_input"] is False
        assert result["diagnostics"]["valid_for_casimir_input"] is False
        assert result["schur"]["valid_for_casimir_input"] is False


def test_eta_channel_cli_rejects_nonpositive_nk(tmp_path):
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


def test_eta_channel_cli_rejects_negative_matsubara_index(tmp_path):
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
