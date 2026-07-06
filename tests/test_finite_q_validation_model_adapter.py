from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from validation.lib.finite_q_validation_models import (
    available_finite_q_validation_models,
    get_finite_q_validation_model,
)


ROOT = Path(__file__).resolve().parents[1]


def test_available_validation_models_and_primary_default():
    names = available_finite_q_validation_models()

    assert "symmetry_bdg_2band" in names
    assert "lno327_four_orbital" in names
    assert get_finite_q_validation_model("symmetry_bdg_2band").primary_validation_model is True
    assert get_finite_q_validation_model("lno327_four_orbital").primary_validation_model is False


def test_validation_model_adapters_build_spec_ansatz_and_params():
    two_band = get_finite_q_validation_model("symmetry_bdg_2band")
    two_amp = two_band.build_pairing_params()
    two_ansatz = two_band.build_ansatz("spm", "bond_endpoint_gauge")

    assert two_band.spec.metadata().name == "symmetry_bdg_2band"
    assert two_amp.delta0_eV == pytest.approx(0.1)
    assert two_ansatz.mean_pairing(0.1, -0.2, two_amp).shape == (2, 2)
    assert two_ansatz.collective_vertices(0.1, -0.2, 0.01, 0.0, two_amp)[0].shape == (4, 4)

    four = get_finite_q_validation_model("lno327_four_orbital")
    four_amp = four.build_pairing_params()
    four_ansatz = four.build_ansatz("dwave", "bond_endpoint_gauge")

    assert four.spec.metadata().name == "lno327_four_orbital"
    assert four_amp.delta0_eV == pytest.approx(0.04)
    assert four_ansatz.mean_pairing(0.1, -0.2, four_amp).shape == (4, 4)


def test_two_band_endpoint_collective_vertices_match_symmetric_at_q0():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params()
    endpoint = model.build_ansatz("dwave", "bond_endpoint_gauge")
    symmetric = model.build_ansatz("dwave", "symmetric_kpm")

    endpoint_vertices = endpoint.collective_vertices(0.37, -0.22, 0.0, 0.0, amp)
    symmetric_vertices = symmetric.collective_vertices(0.37, -0.22, 0.0, 0.0, amp)

    for endpoint_vertex, symmetric_vertex in zip(endpoint_vertices, symmetric_vertices, strict=True):
        np.testing.assert_allclose(endpoint_vertex, symmetric_vertex)


def test_two_band_spm_endpoint_collective_vertices_match_symmetric_at_finite_q():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params()
    endpoint = model.build_ansatz("spm", "bond_endpoint_gauge")
    symmetric = model.build_ansatz("spm", "symmetric_kpm")

    endpoint_vertices = endpoint.collective_vertices(0.37, -0.22, 0.13, -0.07, amp)
    symmetric_vertices = symmetric.collective_vertices(0.37, -0.22, 0.13, -0.07, amp)

    for endpoint_vertex, symmetric_vertex in zip(endpoint_vertices, symmetric_vertices, strict=True):
        np.testing.assert_allclose(endpoint_vertex, symmetric_vertex)


def test_two_band_dwave_endpoint_collective_vertices_differ_from_symmetric_at_finite_q():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params()
    endpoint = model.build_ansatz("dwave", "bond_endpoint_gauge")
    symmetric = model.build_ansatz("dwave", "symmetric_kpm")

    endpoint_vertices = endpoint.collective_vertices(0.37, -0.22, 0.13, -0.07, amp)
    symmetric_vertices = symmetric.collective_vertices(0.37, -0.22, 0.13, -0.07, amp)

    assert np.linalg.norm(endpoint_vertices[0] - symmetric_vertices[0]) > 0.0
    assert np.linalg.norm(endpoint_vertices[1] - symmetric_vertices[1]) > 0.0


def test_two_band_endpoint_collective_vertices_have_endpoint_block_structure():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params()
    endpoint = model.build_ansatz("dwave", "bond_endpoint_gauge")
    kx = 0.37
    ky = -0.22
    qx = 0.13
    qy = -0.07
    delta0 = float(amp.delta0_eV)
    phi_minus = endpoint.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp) / delta0
    phi_plus = endpoint.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp) / delta0

    amplitude, phase = endpoint.collective_vertices(kx, ky, qx, qy, amp)
    zero = np.zeros_like(phi_plus)

    assert amplitude.shape == (4, 4)
    assert phase.shape == (4, 4)
    np.testing.assert_allclose(amplitude[:2, :2], zero)
    np.testing.assert_allclose(amplitude[:2, 2:], phi_plus)
    np.testing.assert_allclose(amplitude[2:, :2], phi_minus.conjugate().T)
    np.testing.assert_allclose(amplitude[2:, 2:], zero)
    np.testing.assert_allclose(phase[:2, :2], zero)
    np.testing.assert_allclose(phase[:2, 2:], 1j * phi_plus)
    np.testing.assert_allclose(phase[2:, :2], -1j * phi_minus.conjugate().T)
    np.testing.assert_allclose(phase[2:, 2:], zero)


def test_validation_model_adapter_rejects_unsupported_pairing():
    model = get_finite_q_validation_model("symmetry_bdg_2band")

    with pytest.raises(ValueError, match="not supported"):
        model.build_ansatz("onsite_s", "bond_endpoint_gauge")


def test_q0_alignment_script_smoke_uses_two_band_model():
    result = subprocess.run(
        [
            sys.executable,
            "validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py",
            "--model",
            "symmetry_bdg_2band",
            "--pairings",
            "normal",
            "spm",
            "dwave",
            "--omega",
            "0.01",
            "--nk",
            "3",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "model_name: symmetry_bdg_2band" in result.stdout
    assert "valid_for_casimir_input: False" in result.stdout


def test_finite_q_ward_scan_script_smoke_uses_two_band_workspace():
    result = subprocess.run(
        [
            sys.executable,
            "validation/scripts/bdg_finite_q/finite_q_ward_scan.py",
            "--model",
            "symmetry_bdg_2band",
            "--pairings",
            "spm",
            "dwave",
            "--omega",
            "0.01",
            "--q-values",
            "0.005",
            "--nk",
            "3",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "model_name: symmetry_bdg_2band" in result.stdout
    assert "workspace_evaluation: True" in result.stdout
    assert "valid_for_casimir_input: False" in result.stdout


def test_four_orbital_transfer_smoke_remains_secondary():
    result = subprocess.run(
        [
            sys.executable,
            "validation/scripts/bdg_finite_q/finite_q_ward_scan.py",
            "--model",
            "lno327_four_orbital",
            "--pairings",
            "spm",
            "--omega",
            "0.01",
            "--q-values",
            "0.005",
            "--nk",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "model_name: lno327_four_orbital" in result.stdout
    assert "workspace_evaluation: True" in result.stdout
