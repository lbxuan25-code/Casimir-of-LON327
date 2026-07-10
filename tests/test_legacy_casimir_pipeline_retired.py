from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path("scripts/casimir/finite_q_bdg_casimir_pipeline.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("retired_finite_q_bdg_casimir_pipeline", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_pipeline_no_longer_contains_zero_rhs_ward_route():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "def _ward_residuals" not in text
    assert "left_ward_residual_norm" not in text
    assert "right_ward_residual_norm" not in text
    assert "max_ward_residual_norm" not in text
    assert "LEGACY_PIPELINE_RETIRED = True" in text


def test_legacy_pipeline_fails_closed_instead_of_producing_results():
    module = _load_module()
    assert module.LEGACY_PIPELINE_RETIRED is True
    with pytest.raises(SystemExit, match="invalid zero-RHS finite-q Ward check"):
        module.main()
