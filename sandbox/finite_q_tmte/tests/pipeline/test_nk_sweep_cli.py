from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_nk_sweep_module():
    path = Path("sandbox/finite_q_tmte/scripts/run_nk_sweep.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_run_nk_sweep_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nk_sweep_cli_accepts_multiple_positive_nk_values(tmp_path):
    module = _load_nk_sweep_module()
    args = module.build_parser().parse_args(
        [
            "--model",
            "symmetry_bdg_2band",
            "--pairing",
            "dwave",
            "--matsubara-index",
            "1",
            "--temperature-K",
            "10.0",
            "--q-values",
            "0.02",
            "--nk-values",
            "5",
            "7",
            "9",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.nk_values == [5, 7, 9]


def test_nk_sweep_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_nk_sweep_module()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "1",
                "--q-values",
                "0.02",
                "--nk-values",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )

