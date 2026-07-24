from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lno327.casimir.material_geometry_qualification_campaign import (
    build_todo4_qualification_campaign,
    load_todo4_qualification_manifest,
)
from lno327.casimir.material_observable_impact_calibration import (
    build_observable_impact_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/configs/casimir/todo4_representative_v1.json"


def _q_hex(value: np.ndarray) -> tuple[str, str]:
    q = np.asarray(value, dtype=float)
    return (float(q[0]).hex(), float(q[1]).hex())


def _static_sample(
    q_hex: tuple[str, str],
    *,
    n_grid: int,
    shift_index: int,
) -> dict[str, object]:
    q = np.asarray([float.fromhex(value) for value in q_hex], dtype=float)
    perturbation = 1.0 + 0.002 * shift_index + 1.0 / n_grid
    chi_bar = (0.25 + 0.2 * float(np.linalg.norm(q))) * perturbation
    dbar_t = (0.08 + 0.1 * float(np.linalg.norm(q))) * perturbation
    return {
        "frequency_sector": "zero_matsubara",
        "xi_eV_hex": 0.0.hex(),
        "q_crystal_hex": list(q_hex),
        "hard_physical_passed": True,
        "sheet_validation": {
            "frequency_sector": "zero_matsubara",
            "passed": True,
            "finite": True,
            "ward_passed": True,
            "relative_imaginary_norm": 0.0,
            "relative_longitudinal_gauge_residual": 0.0,
            "relative_density_transverse_mixing": 0.0,
            "chi_bar": chi_bar,
            "dbar_t": dbar_t,
            "reality_tolerance": 1e-8,
            "longitudinal_tolerance": 1e-6,
            "mixing_tolerance": 1e-6,
            "passivity_tolerance": 1e-10,
        },
        "response": {
            "frequency_sector": "zero_matsubara",
            "chi_bar": chi_bar,
            "dbar_t": dbar_t,
            "primary_norm": float(max(chi_bar, dbar_t)),
        },
        "provenance": {"grid": {"N": n_grid}},
    }


def _positive_sample(
    q_hex: tuple[str, str],
    *,
    xi_eV: float,
    n_grid: int,
    shift_index: int,
) -> dict[str, object]:
    q = np.asarray([float.fromhex(value) for value in q_hex], dtype=float)
    perturbation = 1.0 + 0.001 * shift_index + 0.5 / n_grid
    matrix = perturbation * np.asarray(
        [
            [1.1 + 0.5 * abs(q[0]), 0.04],
            [0.04, 0.7 + 0.4 * abs(q[1])],
        ],
        dtype=float,
    )
    minimum = float(np.min(np.linalg.eigvalsh(matrix)))
    return {
        "frequency_sector": "positive_matsubara",
        "xi_eV_hex": float(xi_eV).hex(),
        "q_crystal_hex": list(q_hex),
        "hard_physical_passed": True,
        "sheet_validation": {
            "frequency_sector": "positive_matsubara",
            "passed": True,
            "finite": True,
            "relative_imaginary_norm": 0.0,
            "relative_symmetry_residual": 0.0,
            "minimum_symmetric_eigenvalue": minimum,
            "reality_tolerance": 1e-9,
            "symmetry_tolerance": 1e-9,
            "passivity_tolerance": 1e-10,
        },
        "response": {
            "frequency_sector": "positive_matsubara",
            "matrix_tilde_real": matrix.tolist(),
            "matrix_tilde_imag": np.zeros((2, 2)).tolist(),
            "spectral_norm": float(np.linalg.norm(matrix, ord=2)),
        },
        "provenance": {"grid": {"N": n_grid}},
    }


def _write_diagnostic_source(path: Path, campaign) -> None:
    direct_entries = [
        entry
        for entry in campaign.entries
        if entry.kind == "direct" and entry.pairing_name == "dwave"
    ]
    xi_by_q_n: dict[tuple[tuple[str, str], int], float] = {}
    for entry in direct_entries:
        plan = entry.geometry_plan
        for point in plan.points:
            for requirement_key in (
                point.plate_1_requirement,
                point.plate_2_requirement,
            ):
                requirement = plan.requirements[requirement_key]
                xi_by_q_n[
                    (_q_hex(requirement.q_crystal), point.matsubara_index)
                ] = float(requirement.identity.xi_eV)

    records = []
    for q_hex in sorted({key[0] for key in xi_by_q_n}):
        frequencies = {}
        for n in (0, 1):
            xi_eV = xi_by_q_n[(q_hex, n)]
            levels = []
            for n_grid in (128, 192, 256):
                samples = {}
                for shift_index in range(3):
                    label = f"shift_{shift_index}"
                    if n == 0:
                        sample = _static_sample(
                            q_hex,
                            n_grid=n_grid,
                            shift_index=shift_index,
                        )
                    else:
                        sample = _positive_sample(
                            q_hex,
                            xi_eV=xi_eV,
                            n_grid=n_grid,
                            shift_index=shift_index,
                        )
                    samples[label] = sample
                levels.append({"N": n_grid, "samples_by_shift": samples})
            frequencies[str(n)] = {
                "matsubara_index": n,
                "levels": levels,
            }
        records.append(
            {
                "pairing_name": "dwave",
                "q_crystal_hex": list(q_hex),
                "status": "diagnostic_completed",
                "frequencies": frequencies,
            }
        )

    payload = {
        "schema": "todo4-unresolved-response-diagnostic-shard-v2",
        "campaign_id": campaign.campaign_id,
        "plan_sha256": "diagnostic-plan",
        "source_commit": "diagnostic-source",
        "base_n_candidates": [128, 192, 256],
        "diagnostic_n_candidates": [128, 192, 256],
        "diagnostic_ladder_tag": "N128-192-256",
        "shard_index": 0,
        "shard_count": 1,
        "records": records,
        "error_count": 0,
        "diagnostic_completed": True,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "shard_000_of_001.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_observable_impact_uses_all_independent_shift_pairs_without_promotion(
    tmp_path: Path,
) -> None:
    campaign = build_todo4_qualification_campaign(
        load_todo4_qualification_manifest(MANIFEST)
    )
    source = tmp_path / "diagnostics"
    _write_diagnostic_source(source, campaign)

    payload = build_observable_impact_calibration(
        campaign,
        diagnostic_source_dir=source,
        pairing_name="dwave",
    )

    assert payload["summary"] == {
        "direct_plan_count": 2,
        "diagnostic_N_count": 3,
        "diagnostic_N_candidates": [128, 192, 256],
        "pair_shift_evaluation_count": 324,
        "local_observable_summary_count": 36,
        "adjacent_observable_group_count": 12,
        "partial_matsubara_pair_count": 162,
        "partial_matsubara_summary_count": 18,
    }
    assert all(
        record["shift_pair_count"] == 9
        for record in payload["local_observable_summaries"]
    )

    axis_rows = [
        row
        for row in payload["local_shift_pair_records"]
        if row["plan_id"] == "direct/axis_parallel/dwave"
    ]
    assert axis_rows
    assert all(row["angular_contrast_logdet"] == 0.0 for row in axis_rows)

    oblique_rows = [
        row
        for row in payload["local_shift_pair_records"]
        if row["plan_id"] == "direct/oblique_rotated/dwave"
    ]
    assert oblique_rows
    assert any(abs(row["angular_contrast_logdet"]) > 0.0 for row in oblique_rows)

    assert payload["contract"]["microscopic_integration_performed"] is False
    assert payload["contract"]["certified_response_cache_write_attempted"] is False
    assert payload["contract"]["diagnostic_response_promoted"] is False
    assert payload["contract"]["observable_error_budget_calibrated"] is False
    assert payload["diagnostic_only"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["production_casimir_allowed"] is False
