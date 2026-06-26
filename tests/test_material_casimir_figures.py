from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lno327.material_casimir_figures import (
    MISSING_SC_BACKEND_MESSAGE,
    MaterialCasimirConfig,
    VALIDATION_MARKER_PATH,
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
        adaptive_level=0,
        gauss_order=1,
        coarse_grid=1,
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
    assert "_finite_q_bdg_bubble" not in text
    assert "_finite_q_bdg_direct" not in text
    assert "_thermal_trace_bdg" not in text
    assert "_density_vertex_bdg" not in text
    assert "stageSC_1" not in text
    assert "stageSC_2" not in text
    assert "stageSC_3" not in text
    assert "stageSC_4" not in text


def test_pairing_enters_cache_key_and_response_config_but_distance_theta_do_not():
    config = _small_config()
    changed_distance_theta = MaterialCasimirConfig(
        n_max=config.n_max,
        N_Q=config.N_Q,
        N_phi=config.N_phi,
        Q_max_nm_inv=config.Q_max_nm_inv,
        theta_deg=(0.0, 15.0),
        distance_nm=(77.0,),
        zero_mode_omega_eV=config.zero_mode_omega_eV,
        temperature_K=config.temperature_K,
        adaptive_level=config.adaptive_level,
        gauss_order=config.gauss_order,
        coarse_grid=config.coarse_grid,
        delta0_eV=config.delta0_eV,
    )
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
    assert response_config_for_pairing("s_pm", config) == response_config_for_pairing("s_pm", changed_distance_theta)


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
    with pytest.raises(RuntimeError, match="validation marker is missing or not PASSED"):
        compute_reflection_point("s_pm", 0, 0.05, 0.0, config)


def test_missing_validated_finite_q_sc_backend_fails_loudly_without_local_q0_fallback():
    config = _small_config()
    for pairing in ("s_pm", "d_wave"):
        with pytest.raises(RuntimeError, match="validation marker is missing or not PASSED"):
            compute_reflection_point(pairing, 1, 0.05, 0.0, config)
        with pytest.raises(RuntimeError, match="validation marker is missing or not PASSED"):
            finite_q_superconducting_response(pairing, 0.01, np.array([0.01, 0.02]), object(), config)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="validation marker is missing or not PASSED"):
        run_point_grid(
            ["s_pm"],
            config,
            cache_dir=Path("/tmp/not-used"),
            workers=1,
            resume=False,
            skip_existing=False,
            force_recompute=True,
        )


def test_formal_validation_marker_path_and_override_flag_are_explicit():
    assert VALIDATION_MARKER_PATH.name == "bdg_finite_q_validation_status.json"
    script = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "--allow-unvalidated-bdg-response" in script


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
    changed_temperature = MaterialCasimirConfig(
        n_max=config.n_max,
        N_Q=config.N_Q,
        N_phi=config.N_phi,
        Q_max_nm_inv=config.Q_max_nm_inv,
        theta_deg=config.theta_deg,
        distance_nm=config.distance_nm,
        zero_mode_omega_eV=config.zero_mode_omega_eV,
        temperature_K=20.0,
        adaptive_level=config.adaptive_level,
        gauss_order=config.gauss_order,
        coarse_grid=config.coarse_grid,
        delta0_eV=config.delta0_eV,
    )
    changed_zero = MaterialCasimirConfig(
        n_max=config.n_max,
        N_Q=config.N_Q,
        N_phi=config.N_phi,
        Q_max_nm_inv=config.Q_max_nm_inv,
        theta_deg=config.theta_deg,
        distance_nm=config.distance_nm,
        zero_mode_omega_eV=(3e-4,),
        temperature_K=config.temperature_K,
        adaptive_level=config.adaptive_level,
        gauss_order=config.gauss_order,
        coarse_grid=config.coarse_grid,
        delta0_eV=config.delta0_eV,
    )
    changed_delta = MaterialCasimirConfig(
        n_max=config.n_max,
        N_Q=config.N_Q,
        N_phi=config.N_phi,
        Q_max_nm_inv=config.Q_max_nm_inv,
        theta_deg=config.theta_deg,
        distance_nm=config.distance_nm,
        zero_mode_omega_eV=config.zero_mode_omega_eV,
        temperature_K=config.temperature_K,
        adaptive_level=config.adaptive_level,
        gauss_order=config.gauss_order,
        coarse_grid=config.coarse_grid,
        delta0_eV=0.08,
    )
    assert hit is not None
    assert miss is None
    assert (
        load_reusable_point_cache(
            tmp_path,
            point_id=row["point_id"],
            response_config=point_response_config("s_pm", row["n"], row["Q_nm_inv"], row["phi_deg"], changed_temperature),
            lattice_convention=lattice,
        )
        is None
    )
    assert (
        load_reusable_point_cache(
            tmp_path,
            point_id=row["point_id"],
            response_config=point_response_config("s_pm", row["n"], row["Q_nm_inv"], row["phi_deg"], changed_zero),
            lattice_convention=lattice,
        )
        is None
    )
    assert (
        load_reusable_point_cache(
            tmp_path,
            point_id=row["point_id"],
            response_config=point_response_config("s_pm", row["n"], row["Q_nm_inv"], row["phi_deg"], changed_delta),
            lattice_convention=lattice,
        )
        is None
    )


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
    with np.load(paths["grid_npz"], allow_pickle=True) as loaded:
        for key in ("reflection_TE_TM", "kappa_m_inv", "sigma_tilde_xy", "response_matrix", "ward_residual", "status"):
            assert key in loaded.files


def test_assemble_reports_fail_and_missing_points_clearly():
    config = _small_config()
    rows = _synthetic_reflection_provider("s_pm", config)
    rows[0]["status"] = "FAIL"
    with pytest.raises(ValueError, match="FAIL points"):
        assemble_energy_data(["s_pm"], config, rows)
    rows = _synthetic_reflection_provider("s_pm", config)[:-1]
    with pytest.raises(ValueError, match="missing required points"):
        assemble_energy_data(["s_pm"], config, rows)


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


def test_validation_scripts_write_under_validation_outputs_only():
    for script_name in (
        "stageSC_0_bdg_operator_ward_vertex_audit.py",
        "stageSC_1_bdg_finite_q_bare_kernel_audit.py",
        "stageSC_2a_bdg_extended_ward_identity_audit.py",
        "stageSC_2_bdg_phase_gauge_restoration_audit.py",
        "stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit.py",
        "stageSC_3_bdg_normal_limit_audit.py",
        "stageSC_4_bdg_q0_limit_audit.py",
        "stageSC_5_bdg_reflection_input_audit.py",
    ):
        text = (ROOT / "validation" / "scripts" / "response" / script_name).read_text(encoding="utf-8")
        assert (
            "validation\" / \"outputs\" / \"response\" / \"bdg_finite_q\"" not in text
            or script_name.startswith(("stageSC_0", "stageSC_5"))
        )
        assert "outputs/material_casimir" not in text
        assert "outputs\" / \"material_casimir" not in text


def test_stage0_operator_ward_audit_scans_requested_candidates():
    text = (ROOT / "validation" / "scripts" / "response" / "stageSC_0_bdg_operator_ward_vertex_audit.py").read_text(
        encoding="utf-8"
    )
    assert "CANDIDATE_SPECS" in text
    assert "2.0 * amp.delta0_eV" in text
    assert "qv_sign in (1, -1)" in text
    assert "operator_residual_max_abs" in text
    assert "onsite_s operator identity failed" in text


def test_validation_scripts_encode_failed_ward_and_minus_schur_selection():
    stage2 = (ROOT / "validation" / "scripts" / "response" / "stageSC_2_bdg_phase_gauge_restoration_audit.py").read_text(
        encoding="utf-8"
    )
    assert "max_minus_schur_Ward" in stage2
    assert "max_plus_schur_Ward" in stage2
    assert "onsite_s" in stage2
    assert "PHASE_VERTICES" in stage2
    assert "phase_phase_direct_included" in stage2
    assert "selected_ward < 1e-6 and improvement > 10.0" in stage2
    assert "return \"FAILED\"" in stage2


def test_validation_md_writer_includes_summary_and_case_diagnostics():
    common = (ROOT / "validation" / "scripts" / "response" / "bdg_finite_q_audit_common.py").read_text(
        encoding="utf-8"
    )
    assert "## Summary" in common
    assert "## Case Diagnostics" in common
    assert "max_minus_schur_Ward" in common
    assert "selected_gauge_restored_Ward" in common
    assert "## Q Scaling" in common


def test_extended_ward_audit_scans_ctheta_candidates():
    stage2a = (
        ROOT / "validation" / "scripts" / "response" / "stageSC_2a_bdg_extended_ward_identity_audit.py"
    ).read_text(encoding="utf-8")
    assert "2j * 0.04" in stage2a
    assert "extended_left_residual_max" in stage2a
    assert "extended_theta_residual_max" in stage2a


def test_amplitude_phase_stage2b_is_material_gate():
    stage2b = (
        ROOT / "validation" / "scripts" / "response" / "stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit.py"
    ).read_text(encoding="utf-8")
    assert "amplitude_phase_Ward" in stage2b
    assert "goldstone_counterterm_Cg" in stage2b
    assert "onsite_s amplitude-phase benchmark failed" in stage2b


def test_stage5_requires_all_prior_stages_passed():
    stage5 = (ROOT / "validation" / "scripts" / "response" / "stageSC_5_bdg_reflection_input_audit.py").read_text(
        encoding="utf-8"
    )
    assert "status != \"PASSED\"" in stage5
    assert "stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit.json" in stage5
    assert "prior StageSC reports are missing or failed" in stage5
