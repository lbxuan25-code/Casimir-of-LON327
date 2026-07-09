from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.primitive_extended_translation_collective_audit import (
    DEFAULT_CANDIDATE,
    SCHEMA_VERSION,
    _rank_translation_vectors,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_primitive_extended_translation_collective_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_and_default_candidate():
    assert SCHEMA_VERSION == "finite_q_tmte_primitive_extended_translation_collective_audit_v1"
    assert DEFAULT_CANDIDATE == "matrix_inferred_matsubara_i_asymmetric"


def test_rank_translation_vectors_exact_minus_translation():
    translation = {
        "equal_forward": np.asarray([0.0, 2.0, 0.0], dtype=complex),
        "delta_v_mid": np.asarray([0.0, 0.5, 0.0], dtype=complex),
        "qM_mid": np.asarray([0.0, 0.1, 0.0], dtype=complex),
    }
    target = -(translation["equal_forward"] - translation["delta_v_mid"])
    rows = _rank_translation_vectors(target, translation)
    assert rows[0]["name"] == "minus_translation_forward"
    assert rows[0]["difference_norm"] == pytest.approx(0.0)


def test_cli_rejects_nonpositive_q(tmp_path):
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
                "0.0",
                "--nk",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_cli_rejects_nonpositive_nk(tmp_path):
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
