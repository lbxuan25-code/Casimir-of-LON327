from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_9_physical_ward_residual_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_9_regression", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_projection_helper_uses_linear_spatial_projection():
    module = _load_module()
    q = np.array([1.0, 0.0])
    spatial = np.array([2.0 + 1.0j, 3.0 - 1.0j])

    longitudinal, transverse = module.project_spatial_components(q, spatial)

    assert longitudinal == 2.0 + 1.0j
    assert transverse == 3.0 - 1.0j


def test_status_classifier_fixed_rules():
    module = _load_module()

    assert module.classify_status(1e-12, 0.0) == "NUMERICALLY_CLOSED"
    assert module.classify_status(1e-5, 1.0) == "ORDER_Q_RESIDUAL"
    assert module.classify_status(1e-5, 2.0) == "ORDER_Q2_OR_BETTER_RESIDUAL"
    assert module.classify_status(1e-5, 0.2) == "NON_SCALING_OR_UNCLEAR_RESIDUAL"


def test_json_helper_rejects_complex_objects():
    module = _load_module()

    with pytest.raises(TypeError):
        module.to_jsonable({"bad": 1.0 + 2.0j})
    assert module.to_jsonable({"ok": np.array([1.0, 2.0])}) == {"ok": [1.0, 2.0]}


def test_main_regression_function_runs_with_test_mesh():
    module = _load_module()

    data = module.run_regression(mesh_size=12)

    assert data["stage"] == "Stage 4.9"
    assert data["config"]["mesh_size"] == 12
    assert set(data["responses"]) == {
        "stage48_physical_observable_source",
        "stage47_historical_observable_observable",
    }
    for response in data["responses"].values():
        assert len(response["results_by_q_scale"]) == 4
        assert "max_norm" in response["slopes"]
        assert response["status"] in {
            "NUMERICALLY_CLOSED",
            "ORDER_Q_RESIDUAL",
            "ORDER_Q2_OR_BETTER_RESIDUAL",
            "NON_SCALING_OR_UNCLEAR_RESIDUAL",
        }
