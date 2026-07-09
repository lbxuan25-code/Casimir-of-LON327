from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.pairing_contact_missing_audit import (
    DEFAULT_DELTA0_VALUES,
    DEFAULT_PAIRINGS,
    SCHEMA_VERSION,
    summarize_contact_formula_payload,
    trend_by_delta0,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_pairing_contact_missing_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_pairing_contact_missing_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _side(alpha: complex, *, norm: float = 2.0) -> dict:
    return {
        "required_over_current_scalar_projection": {
            "alpha_required_over_current": alpha,
            "residual_norm": 1e-15,
            "residual_over_required_norm": 1e-12,
        },
        "componentwise_required_over_current": [
            {"label": "A0", "required_over_current": 0.0 + 0.0j, "ratio_defined": False},
            {"label": "L", "required_over_current": alpha, "ratio_defined": True},
            {"label": "T", "required_over_current": 0.0 + 0.0j, "ratio_defined": False},
        ],
        "contact_required": {"norm": norm * abs(alpha)},
        "contact_current": {"norm": norm},
        "ward_residual_with_current_contact": {"norm": norm * abs(1.0 - alpha)},
        "parallelism": {"abs_overlap": 1.0},
    }


def _payload(alpha: complex, *, pairing: str = "dwave", delta0: float = 0.1) -> dict:
    return {
        "model": {"pairing": pairing},
        "debug_parameters": {"delta0_eV": delta0},
        "contact_formula_analysis": {
            "left": _side(alpha),
            "right": _side(alpha),
        },
    }


def test_schema_defaults():
    assert SCHEMA_VERSION == "finite_q_tmte_pairing_contact_missing_audit_v1"
    assert DEFAULT_PAIRINGS == ("dwave", "spm")
    assert DEFAULT_DELTA0_VALUES == (0.0, 0.05, 0.1, 0.15)


def test_summarize_contact_formula_payload_extracts_alpha_and_missing_fraction():
    summary = summarize_contact_formula_payload(_payload(0.8268 + 0.0j, delta0=0.1))
    assert summary["pairing"] == "dwave"
    assert summary["delta0_eV"] == pytest.approx(0.1)
    assert summary["alpha_real_mean"] == pytest.approx(0.8268)
    assert summary["missing_fraction_real"] == pytest.approx(0.1732)
    assert summary["parallelism_abs_overlap_mean"] == pytest.approx(1.0)
    assert summary["projection_residual_over_required_mean"] == pytest.approx(1e-12)


def test_trend_by_delta0_fits_alpha_vs_delta0_squared():
    rows = []
    intercept = 1.0
    slope = -12.0
    for delta0 in [0.0, 0.05, 0.1, 0.15]:
        rows.append(
            {
                "status": "ok",
                "delta0_eV": delta0,
                "alpha_real_mean": intercept + slope * delta0**2,
            }
        )
    trend = trend_by_delta0(rows)
    assert trend["status"] == "linear_fit_alpha_vs_delta0_squared"
    assert trend["intercept"] == pytest.approx(intercept)
    assert trend["slope_per_eV2"] == pytest.approx(slope)
    assert trend["max_abs_residual"] < 1e-12


def test_trend_by_delta0_handles_insufficient_points():
    trend = trend_by_delta0([{"status": "ok", "delta0_eV": 0.1, "alpha_real_mean": 0.8}])
    assert trend["status"] == "insufficient_points"


def test_pairing_contact_missing_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
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


def test_pairing_contact_missing_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
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
