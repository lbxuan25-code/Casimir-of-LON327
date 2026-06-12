from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage5_1_response_to_conductivity_convention_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_1_response_to_conductivity", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_imports():
    module = _load_module()
    assert module.JSON_OUTPUT.name == "stage5_1_response_to_conductivity_convention_audit.json"


def test_spatial_response_to_conductivity_returns_2x2_matrix():
    module = _load_module()
    response = np.zeros((3, 3), dtype=complex)
    response[1, 1] = 2.0
    response[2, 2] = 4.0
    sigma = module.spatial_response_to_conductivity(response, 2.0, "A_plus_xi")
    assert sigma.shape == (2, 2)
    np.testing.assert_allclose(sigma, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex))


def test_spatial_response_to_conductivity_rejects_zero_omega():
    module = _load_module()
    with pytest.raises(ValueError, match="nonzero"):
        module.spatial_response_to_conductivity(np.zeros((3, 3), dtype=complex), 0.0, "A_plus_xi")


def test_json_top_level_fields_and_boundaries():
    module = _load_module()
    data = module.run_audit(quick=True)
    for key in (
        "stage",
        "purpose",
        "boundary",
        "existing_code_audit",
        "response_convention",
        "candidate_conductivity_conventions",
        "selected_convention",
        "unit_audit",
        "lightweight_sanity_check",
        "diagnostic_status",
    ):
        assert key in data
    for value in data["boundary"].values():
        assert value is True


def test_ambiguous_convention_is_not_ready():
    module = _load_module()
    data = module.run_audit(quick=True)
    assert data["selected_convention"]["status"] == "AMBIGUOUS"
    assert data["diagnostic_status"]["conductivity_convention_status"] == "CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE"
    assert "READY_FOR_STAGE5_2_NUMERICAL_CONDUCTIVITY_SANITY" not in data["diagnostic_status"].values()


def test_no_downstream_modules_imported():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("reflection" in module for module in imported_modules)
    assert not any("casimir" in module.lower() for module in imported_modules)


def test_quick_sanity_mode_runs():
    module = _load_module()
    sanity = module.lightweight_sanity_check(quick=True)
    assert sanity["quick_mode"] is True
    assert sanity["all_spatial_blocks_finite"] is True
    assert sanity["all_candidate_sigmas_finite"] is True
    assert len(sanity["rows"]) == 3

