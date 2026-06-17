from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.material_casimir_figures import (
    MaterialCasimirConfig,
    assemble_energy_data,
    compute_reflection_point,
    finite_q_superconducting_response,
    interior_q_nodes_nm_inv,
    point_id,
    point_response_config,
    q_phi_weight_map,
    response_config_for_pairing,
    run_point_grid,
    save_material_casimir_outputs,
    trace_log_grid,
)
from lno327.material_response_cache import load_reusable_point_cache, write_point_cache

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_material_casimir_figures.py"
PLOT_SCRIPT = ROOT / "scripts" / "plot_material_casimir_figures.py"
HELPER = ROOT / "src" / "lno327" / "material_casimir_figures.py"


def _small_config() -> MaterialCasimirConfig:
    return MaterialCasimirConfig(
        n_max=1,
        N_Q=2,
        N_phi=24,
        Q_max_nm_inv=0.25,
        theta_deg=(0.0, 15.0, 30.0),
        distance_nm=(50.0, 100.0),
        zero_mode_omega_eV=(1e-4, 1e-3),
        temperature_K=10.0,
    )


def _synthetic_reflection_provider(pairing: str, config: MaterialCasimirConfig) -> list[dict[str, object]]:
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = np.linspace(0.0, 360.0, config.N_phi, endpoint=False)
    scale = 0.01 if pairing == "s_pm" else 0.012
    rows = []
    for n in range(config.n_max + 1):
        for q in q_nodes:
            for phi in phi_nodes:
                angle = np.deg2rad(phi)
                reflection = np.array(
                    [
                        [scale * (1.0 + 0.1 * np.cos(2.0 * angle)), scale * 0.02 * np.sin(2.0 * angle)],
                        [-scale * 0.02 * np.sin(2.0 * angle), scale * (0.8 - 0.1 * np.cos(2.0 * angle))],
                    ],
                    dtype=complex,
                )
                rows.append(
                    {
                        "point_id": point_id(pairing, n, float(q), float(phi)),
                        "pairing": pairing,
                        "canonical_pairing": "spm" if pairing == "s_pm" else "dwave",
                        "n": n,
                        "Q_nm_inv": float(q),
                        "Q_m_inv": float(q) * 1e9,
                        "phi_deg": float(phi),
                        "phi_rad": float(np.deg2rad(phi)),
                        "kappa_m_inv": float(q) * 1e9,
                        "reflection_TE_TM": reflection,
                        "sigma_tilde_xy": np.eye(2, dtype=complex) * scale,
                        "response_matrix": np.eye(3, dtype=complex) * scale,
                        "n0_source": "reflection_level_average_over_zero_mode_omega_eV" if n == 0 else None,
                        "ward_residual": {"status": "SYNTHETIC", "total_max": None},
                        "status": "PASS",
                        "cache": {"source": "hit"},
                    }
                )
    return rows


def test_no_forbidden_local_or_old_paths_in_helper():
    text = HELPER.read_text(encoding="utf-8")
    assert "local_response_imag_axis" not in text
    assert "reflection_matrix_weak_2d" not in text
    assert "casimir_energy_integrand" not in text


def test_pairing_enters_cache_key_and_response_config_but_distance_theta_do_not():
    config = _small_config()
    assert point_id("s_pm", 1, 0.05, 0.0).startswith("s_pm_")
    assert point_id("d_wave", 1, 0.05, 0.0).startswith("d_wave_")
    base = response_config_for_pairing("s_pm", config)
    point_cfg = point_response_config("s_pm", 1, 0.05, 15.0, config)
    assert point_cfg["pairing"] == "s_pm"
    assert point_cfg["canonical_pairing"] == "spm"
    assert "distance_nm" not in base
    assert "theta_deg" not in base
    assert "distance_nm" not in point_cfg
    assert "theta_deg" not in point_cfg
    assert response_config_for_pairing("s_pm", config) != response_config_for_pairing("d_wave", config)


def test_q0_excluded_and_q_weight_appears_only_in_integration_weight():
    config = _small_config()
    nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    assert np.all(nodes > 0.0)
    weights = q_phi_weight_map(config)
    q0 = float(nodes[0])
    phi0 = 0.0
    expected = (q0 * 1e9) * (config.Q_max_nm_inv * 1e9 / config.N_Q) * (2.0 * np.pi / config.N_phi) / (2.0 * np.pi) ** 2
    assert weights[(round(q0, 12), phi0)] == pytest.approx(expected)

    rows = _synthetic_reflection_provider("s_pm", config)
    logdet, records = trace_log_grid(["s_pm"], config, rows)
    assert logdet.shape == (1, 2, 2, 24, 2, 3)
    assert "logdet" in records[0]
    assert "weight" not in records[0]


def test_n0_uses_reflection_level_limit_marker_in_rows():
    config = _small_config()
    rows = _synthetic_reflection_provider("s_pm", config)
    n0_rows = [row for row in rows if row["n"] == 0]
    assert n0_rows
    assert all(row["n0_source"] == "reflection_level_average_over_zero_mode_omega_eV" for row in n0_rows)


def test_finite_q_sc_response_fails_loudly_instead_of_local_q0_fallback():
    config = _small_config()
    with pytest.raises(NotImplementedError, match="finite-q superconducting response"):
        compute_reflection_point("s_pm", 1, 0.05, 0.0, config)
    with pytest.raises(NotImplementedError, match="local q=0 fallback"):
        finite_q_superconducting_response("d_wave", 0.01, np.zeros(2), object(), config)  # type: ignore[arg-type]


def test_resume_skip_existing_reuses_pairing_safe_cache(tmp_path):
    config = _small_config()
    row = _synthetic_reflection_provider("s_pm", config)[0]
    lattice = {"a_x_m": 1.0, "a_y_m": 1.0}
    write_point_cache(
        tmp_path,
        row,
        response_config=point_response_config("s_pm", row["n"], row["Q_nm_inv"], row["phi_deg"], config),
        lattice_convention=lattice,
    )
    hit = load_reusable_point_cache(
        tmp_path,
        point_id=row["point_id"],
        response_config=point_response_config("s_pm", row["n"], row["Q_nm_inv"], row["phi_deg"], config),
        lattice_convention=lattice,
    )
    miss = load_reusable_point_cache(
        tmp_path,
        point_id=row["point_id"],
        response_config=point_response_config("d_wave", row["n"], row["Q_nm_inv"], row["phi_deg"], config),
        lattice_convention=lattice,
    )
    assert hit is not None
    assert miss is None


def test_synthetic_reflection_rows_assemble_energy_and_torque(tmp_path):
    config = _small_config()
    pairings = ["s_pm", "d_wave"]
    rows = run_point_grid(
        pairings,
        config,
        cache_dir=tmp_path / "cache",
        workers=1,
        resume=True,
        skip_existing=True,
        force_recompute=False,
        provider=_synthetic_reflection_provider,
    )
    energy_data = assemble_energy_data(pairings, config, rows)
    assert energy_data["F_over_A_J_m2"].shape == (2, 2, 3)
    assert energy_data["delta_F_over_A_J_m2"].shape == (2, 2, 3)
    assert energy_data["tau_over_A_J_m2_rad"].shape == (2, 2, 3)
    assert energy_data["diagnostics"]["cache_hits"] == len(rows)
    assert energy_data["diagnostics"]["failed_points"] == 0
    paths = save_material_casimir_outputs(tmp_path, config, rows, energy_data)
    assert Path(paths["grid_json"]).exists()
    assert Path(paths["grid_npz"]).exists()
    assert Path(paths["integrand_json"]).exists()
    assert Path(paths["integrand_npz"]).exists()
    assert Path(paths["energy_json"]).exists()
    assert Path(paths["energy_npz"]).exists()


def test_plot_script_reads_saved_data_without_response(tmp_path):
    config = _small_config()
    pairings = ["s_pm", "d_wave"]
    rows = [row for pairing in pairings for row in _synthetic_reflection_provider(pairing, config)]
    energy_data = assemble_energy_data(pairings, config, rows)
    paths = save_material_casimir_outputs(tmp_path, config, rows, energy_data)
    proc = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--data-npz",
            paths["energy_npz"],
            "--figures-dir",
            str(tmp_path / "figures"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "response was not recomputed" in proc.stdout
    assert (tmp_path / "figures" / "material_casimir_energy_vs_distance.png").exists()


def test_run_script_dry_run_writes_plan_without_response(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--dry-run-grid-only",
            "--pairings",
            "s_pm",
            "d_wave",
            "--n-max",
            "1",
            "--N-Q",
            "2",
            "--N-phi",
            "24",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "material_casimir_dry_run_plan.json" in proc.stdout
    plan = json.loads((tmp_path / "data" / "material_casimir_dry_run_plan.json").read_text(encoding="utf-8"))
    assert plan["num_points"] == 2 * (1 + 1) * 2 * 24
