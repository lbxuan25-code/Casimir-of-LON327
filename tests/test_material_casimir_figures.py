from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from lno327.material_casimir_figures import (
    MaterialCasimirConfig,
    assemble_energy_data,
    interior_q_nodes_nm_inv,
    point_id,
    response_config_for_pairing,
    run_point_grid,
    save_material_casimir_outputs,
)
from lno327.material_response_cache import write_point_cache

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_material_casimir_figures.py"
PLOT_SCRIPT = ROOT / "scripts" / "plot_material_casimir_figures.py"


def _small_config() -> MaterialCasimirConfig:
    return MaterialCasimirConfig(
        n_max=1,
        N_Q=2,
        N_phi=2,
        Q_max_nm_inv=0.25,
        theta_deg=(0.0, 45.0, 90.0),
        distance_nm=(50.0, 100.0),
        zero_mode_omega_eV=(1e-4, 1e-3),
        temperature_K=10.0,
        bdg_nk=2,
    )


def _synthetic_provider(pairing: str, config: MaterialCasimirConfig) -> list[dict[str, object]]:
    q_nodes = interior_q_nodes_nm_inv(config.Q_max_nm_inv, config.N_Q)
    phi_nodes = np.linspace(0.0, 360.0, config.N_phi, endpoint=False)
    scale = 1.0 if pairing == "s_pm" else 1.2
    rows = []
    for n in range(config.n_max + 1):
        for q in q_nodes:
            for phi in phi_nodes:
                base = -scale * (1.0 + n) * (1.0 + float(q)) * 1e-6
                theta_factor = 1.0 + 0.02 * np.cos(2.0 * config.theta_rad)
                distance_factor = (50.0 / np.asarray(config.distance_nm))[:, None]
                integrand = base * distance_factor * theta_factor[None, :]
                rows.append(
                    {
                        "point_id": point_id(pairing, n, float(q), float(phi)),
                        "pairing": pairing,
                        "canonical_pairing": "spm" if pairing == "s_pm" else "dwave",
                        "n": n,
                        "Q_nm_inv": float(q),
                        "phi_deg": float(phi),
                        "n_weight": 0.5 if n == 0 else 1.0,
                        "integrand_grid": integrand.astype(complex),
                        "ward_residual": {"status": "NOT_EVALUATED_SYNTHETIC", "total_max": None},
                        "status": "PASS",
                        "cache": {"source": "hit"},
                    }
                )
    return rows


def test_default_interior_q_nodes_exclude_q0():
    nodes = interior_q_nodes_nm_inv(0.25, 16)
    assert len(nodes) == 16
    assert nodes[0] > 0.0
    assert nodes[-1] < 0.25


def test_pairing_enters_point_id_and_response_config():
    config = _small_config()
    assert point_id("s_pm", 1, 0.05, 0.0).startswith("s_pm_")
    assert point_id("d_wave", 1, 0.05, 0.0).startswith("d_wave_")
    spm_config = response_config_for_pairing("s_pm", config)
    dwave_config = response_config_for_pairing("d_wave", config)
    assert spm_config["pairing"] == "s_pm"
    assert dwave_config["pairing"] == "d_wave"
    assert spm_config != dwave_config


def test_pairing_safe_cache_does_not_reuse_cross_pairing(tmp_path):
    config = _small_config()
    row = _synthetic_provider("s_pm", config)[0]
    lattice = {"layer": "test"}
    write_point_cache(
        tmp_path,
        row,
        response_config=response_config_for_pairing("s_pm", config),
        lattice_convention=lattice,
    )
    payload = json.loads((tmp_path / f"{row['point_id']}.json").read_text(encoding="utf-8"))
    assert payload["response_config"]["pairing"] == "s_pm"
    assert payload["response_config"] != response_config_for_pairing("d_wave", config)


def test_synthetic_provider_assembles_energy_and_torque():
    config = _small_config()
    pairings = ["s_pm", "d_wave"]
    rows = run_point_grid(
        pairings,
        config,
        cache_dir=Path("/tmp/not-used"),
        workers=1,
        resume=False,
        skip_existing=False,
        force_recompute=False,
        provider=_synthetic_provider,
    )
    assert len(rows) == len(pairings) * (config.n_max + 1) * config.N_Q * config.N_phi
    energy_data = assemble_energy_data(pairings, config, rows)
    assert energy_data["F_over_A_J_m2"].shape == (2, 2, 3)
    assert energy_data["delta_F_over_A_J_m2"].shape == (2, 2, 3)
    assert energy_data["tau_over_A_J_m2_rad"].shape == (2, 2, 3)
    assert energy_data["diagnostics"]["failed_points"] == 0
    assert energy_data["diagnostics"]["cache_hits"] == len(rows)
    assert energy_data["diagnostics"]["max_angular_energy_variation"] > 0.0


def test_save_outputs_and_plot_script_read_saved_data(tmp_path):
    config = _small_config()
    pairings = ["s_pm", "d_wave"]
    rows = [row for pairing in pairings for row in _synthetic_provider(pairing, config)]
    energy_data = assemble_energy_data(pairings, config, rows)
    paths = save_material_casimir_outputs(tmp_path, config, rows, energy_data)
    assert Path(paths["grid_json"]).exists()
    assert Path(paths["grid_npz"]).exists()
    assert Path(paths["integrand_json"]).exists()
    assert Path(paths["integrand_npz"]).exists()
    assert Path(paths["energy_json"]).exists()
    assert Path(paths["energy_npz"]).exists()
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
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "material_casimir_dry_run_plan.json" in proc.stdout
    plan = json.loads((tmp_path / "data" / "material_casimir_dry_run_plan.json").read_text(encoding="utf-8"))
    assert plan["num_points"] == 2 * (1 + 1) * 2 * 2
    assert plan["report_label"] == "finite-grid publication-style candidate result; not full convergence audit"
