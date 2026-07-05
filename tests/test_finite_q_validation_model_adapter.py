from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
