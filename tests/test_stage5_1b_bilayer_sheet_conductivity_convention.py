from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.response_conventions import (
    bilayer_sheet_conductivity_convention_metadata,
    spatial_response_to_bilayer_sheet_conductivity_model,
)

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "lno327" / "response_conventions.py"
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_1b_bilayer_sheet_conductivity_convention.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("stage5_1b_bilayer_sheet_conductivity", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spatial_response_to_bilayer_sheet_conductivity_model_shape():
    response = np.zeros((3, 3), dtype=complex)
    sigma = spatial_response_to_bilayer_sheet_conductivity_model(response, 0.02)
    assert sigma.shape == (2, 2)


def test_conversion_sign():
    response = np.zeros((3, 3), dtype=complex)
    response[1, 1] = -0.3
    sigma = spatial_response_to_bilayer_sheet_conductivity_model(response, 0.02)
    assert sigma[0, 0] == 15.0


def test_offdiag_conversion():
    response = np.zeros((3, 3), dtype=complex)
    response[1, 2] = 0.01
    sigma = spatial_response_to_bilayer_sheet_conductivity_model(response, 0.02)
    assert sigma[0, 1] == -0.5


def test_zero_omega_raises():
    with pytest.raises(ValueError, match="positive"):
        spatial_response_to_bilayer_sheet_conductivity_model(np.zeros((3, 3), dtype=complex), 0.0)


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        spatial_response_to_bilayer_sheet_conductivity_model(np.zeros((2, 2), dtype=complex), 0.02)


def test_metadata():
    metadata = bilayer_sheet_conductivity_convention_metadata()
    assert metadata["electric_field_relation"] == "E_j(i xi) = - xi A_j(i xi)"
    assert metadata["model_formula"] == "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV"
    assert metadata["normalization"] == "bilayer-normalized 2D sheet conductivity"
    assert metadata["si_scaling_applied"] is False
    assert metadata["not_bulk_3d"] is True
    assert metadata["not_single_layer"] is True


def test_script_outputs_json_and_md(tmp_path):
    output_json = tmp_path / "stage5_1b.json"
    output_md = tmp_path / "stage5_1b.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )
    assert output_json.exists()
    assert output_md.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["stage"] == "Stage 5.1b"
    assert data["selected_convention"]["response_to_conductivity_formula"] == (
        "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV"
    )
    assert data["synthetic_check"]["status"] == "PASS"
    assert data["diagnostic_status"]["conductivity_convention_status"] == "CONVENTION_FIXED"
    assert "Stage 5.1b Bilayer sheet conductivity convention" in output_md.read_text(encoding="utf-8")


def test_no_reflection_or_casimir_imports():
    imported_modules: list[str] = []
    for path in (HELPER, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    assert not any("reflection" in module.lower() for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)
