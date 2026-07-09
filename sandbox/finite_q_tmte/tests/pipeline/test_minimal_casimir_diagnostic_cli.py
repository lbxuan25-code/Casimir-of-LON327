from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_diagnostic_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unified_cli_q_scan_defaults_to_no_shift(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "q-scan",
        "--matsubara-index", "1",
        "--q-values", "0.01", "0.02",
        "--phi-values", "0", "180",
        "--plate2-theta-deg", "45",
        "--nk", "13",
        "--separation-nm", "20",
        "--output-dir", str(tmp_path),
    ])
    assert args.command == "q-scan"
    assert args.shift_fractions == [0.0]
    assert args.q_values == [0.01, 0.02]


def test_unified_cli_shift_scan_parse_args(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "shift-scan",
        "--matsubara-index", "1",
        "--q", "0.04",
        "--phi-values", "0", "15", "30", "45",
        "--plate2-theta-deg", "45",
        "--nk", "13",
        "--separation-nm", "20",
        "--shift-values", "0.0", "0.2", "0.4",
        "--output-dir", str(tmp_path),
    ])
    assert args.command == "shift-scan"
    assert args.shift_values == [0.0, 0.2, 0.4]
    assert args.r_norm_warning_threshold == module.DEFAULT_R_NORM_WARNING_THRESHOLD


def test_unified_cli_theta_scan_accepts_polar_q_without_plate2_arg(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "theta-scan",
        "--matsubara-index", "1",
        "--q", "0.02",
        "--phi-deg", "30",
        "--theta-values", "0", "45",
        "--nk", "13",
        "--separation-nm", "20",
        "--output-dir", str(tmp_path),
    ])
    q_lab = module._q_lab_from_args(args)
    assert q_lab.shape == (2,)


def test_unified_cli_runs_small_q_scan(tmp_path):
    module = _load_cli()
    rc = module.main([
        "q-scan",
        "--matsubara-index", "1",
        "--q-values", "0.01", "0.02",
        "--phi-values", "0", "180",
        "--plate2-theta-deg", "45",
        "--nk", "1",
        "--separation-nm", "20",
        "--skip-rhs-aware-validation",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "minimal_casimir_q_scan.json").exists()
    assert (tmp_path / "minimal_casimir_q_scan.csv").exists()
