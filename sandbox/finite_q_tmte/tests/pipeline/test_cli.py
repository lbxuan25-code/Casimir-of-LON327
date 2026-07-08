from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_run_scan_module():
    path = Path("sandbox/finite_q_tmte/scripts/run_scan.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_run_scan_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_scan_cli_rejects_independent_omega_argument(tmp_path):
    module = _load_run_scan_module()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "1",
                "--omega",
                "0.02",
                "--q-values",
                "0.02",
                "--nk",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_run_scan_cli_accepts_matsubara_index(tmp_path):
    module = _load_run_scan_module()
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
            "--nk",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.matsubara_index == 1
    assert args.temperature_K == 10.0


def test_run_scan_cli_rejects_old_xi_argument(tmp_path):
    module = _load_run_scan_module()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--xi",
                "0.01",
                "--q-values",
                "0.02",
                "--nk",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_run_scan_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_run_scan_module()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "-1",
                "--q-values",
                "0.02",
                "--nk",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
