"""Diagnostic propagation of unresolved response spread into local Casimir observables.

This module consumes persisted unresolved-response diagnostics only. It never calls
microscopic integration, writes the certified response cache, or promotes an
uncertified response. The goal is to quantify how N/shift spread propagates through
the exact reflection and passive trace-log geometry used by TODO 4.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.casimir.material_geometry_qualification_campaign import (
    Todo4QualificationCampaign,
    Todo4QualificationPlanEntry,
)
from lno327.casimir.material_geometry_qualification_execution import require_frozen_plan
from lno327.casimir.material_geometry_qualification_io import atomic_write_json, source_commit
from lno327.electrodynamics.basis import (
    q_lab_to_crystal,
    tensor_crystal_to_lab,
    tensor_xy_to_lt,
)
from lno327.electrodynamics.conventions import SheetResponseValidation
from lno327.electrodynamics.reflection import (
    LAB_LT_TANGENTIAL_E_BASIS,
    SheetReflection,
    model_q_to_si_wavevector,
    omega_eV_to_xi_si,
    tangential_electric_reflection_matrix_LT,
    vacuum_kappa,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
    static_sheet_response_to_reflection,
)


TODO4_OBSERVABLE_IMPACT_CALIBRATION_SCHEMA = (
    "todo4-observable-impact-calibration-v1"
)


def _q_hex(value: np.ndarray) -> tuple[str, str]:
    q = np.asarray(value, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q must be a finite vector with shape (2,)")
    return (float(q[0]).hex(), float(q[1]).hex())


def _q_from_hex(value: object) -> np.ndarray:
    parts = tuple(str(item) for item in value)
    if len(parts) != 2:
        raise ValueError("q_crystal_hex must contain two components")
    q = np.asarray([float.fromhex(item) for item in parts], dtype=float)
    if not np.isfinite(q).all() or float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_crystal_hex must encode a finite nonzero vector")
    return q


def _complex_matrix(real: object, imag: object) -> np.ndarray:
    matrix = np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("diagnostic response matrix must have shape (2, 2)")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError("diagnostic response matrix must be finite")
    return matrix


def _primary_matrix(sample: Mapping[str, Any]) -> np.ndarray:
    response = dict(sample["response"])
    sector = str(sample["frequency_sector"])
    if sector == "zero_matsubara":
        return np.diag(
            [float(response["chi_bar"]), float(response["dbar_t"])]
        ).astype(complex)
    if sector == "positive_matsubara":
        return _complex_matrix(
            response["matrix_tilde_real"],
            response["matrix_tilde_imag"],
        )
    raise ValueError(f"unsupported diagnostic frequency sector: {sector!r}")


def _spread(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("spread requires at least one value")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("spread values must be finite")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    mean = float(np.mean(array))
    absolute = float(maximum - minimum)
    scale = max(float(np.max(np.abs(array))), 1e-300)
    return {
        "count": int(array.size),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "absolute_spread": absolute,
        "relative_spread_to_max_abs": absolute / scale,
    }


def _matrix_spread(samples: list[Mapping[str, Any]]) -> dict[str, float | int]:
    matrices = [_primary_matrix(sample) for sample in samples]
    if not matrices:
        raise ValueError("matrix spread requires at least one sample")
    maximum_delta = 0.0
    for left_index, left in enumerate(matrices):
        for right in matrices[left_index + 1 :]:
            maximum_delta = max(
                maximum_delta,
                float(np.linalg.norm(left - right, ord=2)),
            )
    scale = max(
        max(float(np.linalg.norm(matrix, ord=2)) for matrix in matrices),
        1e-300,
    )
    return {
        "sample_count": len(matrices),
        "maximum_pairwise_spectral_delta": maximum_delta,
        "relative_spread_to_max_spectral_norm": maximum_delta / scale,
    }


def _static_validation(payload: Mapping[str, Any]) -> StaticSheetValidation:
    return StaticSheetValidation(
        finite=bool(payload["finite"]),
        ward_passed=bool(payload["ward_passed"]),
        relative_imaginary_norm=float(payload["relative_imaginary_norm"]),
        relative_longitudinal_gauge_residual=float(
            payload["relative_longitudinal_gauge_residual"]
        ),
        relative_density_transverse_mixing=float(
            payload["relative_density_transverse_mixing"]
        ),
        chi_bar=float(payload["chi_bar"]),
        dbar_t=float(payload["dbar_t"]),
        reality_tolerance=float(payload["reality_tolerance"]),
        longitudinal_tolerance=float(payload["longitudinal_tolerance"]),
        mixing_tolerance=float(payload["mixing_tolerance"]),
        passivity_tolerance=float(payload["passivity_tolerance"]),
    )


def _positive_validation(payload: Mapping[str, Any]) -> SheetResponseValidation:
    return SheetResponseValidation(
        finite=bool(payload["finite"]),
        relative_imaginary_norm=float(payload["relative_imaginary_norm"]),
        relative_symmetry_residual=float(payload["relative_symmetry_residual"]),
        minimum_symmetric_eigenvalue=float(payload["minimum_symmetric_eigenvalue"]),
        reality_tolerance=float(payload["reality_tolerance"]),
        symmetry_tolerance=float(payload["symmetry_tolerance"]),
        passivity_tolerance=float(payload["passivity_tolerance"]),
    )


def _diagnostic_reflection(
    sample: Mapping[str, Any],
    *,
    q_lab: np.ndarray,
    theta_rad: float,
    entry: Todo4QualificationPlanEntry,
):
    if not bool(sample["hard_physical_passed"]):
        raise ValueError("observable impact requires hard-physical diagnostic samples")
    validation_payload = dict(sample["sheet_validation"])
    if not bool(validation_payload["passed"]):
        raise ValueError("observable impact requires passed sheet validation")

    q_crystal = _q_from_hex(sample["q_crystal_hex"])
    expected_q = q_lab_to_crystal(np.asarray(q_lab, dtype=float), float(theta_rad))
    q_scale = max(float(np.linalg.norm(q_lab)), 1.0)
    mismatch = float(np.linalg.norm(q_crystal - expected_q) / q_scale)
    policy = entry.geometry_plan.policy
    reflection_policy = policy.reflection_policy
    if mismatch > reflection_policy.q_match_tolerance:
        raise ValueError(
            "diagnostic response q differs from exact plate requirement: "
            f"relative mismatch={mismatch:.3e}"
        )

    sector = str(sample["frequency_sector"])
    response_payload = dict(sample["response"])
    if sector == "zero_matsubara":
        validation = _static_validation(validation_payload)
        if not validation.passed:
            raise ValueError("reconstructed static validation does not pass")
        chi_bar = float(response_payload["chi_bar"])
        dbar_t = float(response_payload["dbar_t"])
        material_policy = entry.geometry_plan.response_config.material_policy
        energy = float(material_policy.static_energy_scale_eV)
        kernel_lt = np.zeros((3, 3), dtype=complex)
        kernel_lt[0, 0] = -chi_bar / energy
        kernel_lt[2, 2] = -dbar_t * energy
        response = StaticSheetResponse(
            kernel_lt=kernel_lt,
            chi_bar=chi_bar,
            dbar_t=dbar_t,
            q_model=q_crystal,
            energy_scale_eV=energy,
            degeneracy=float(material_policy.degeneracy),
            basis=STATIC_LOCAL_BASIS,
            validation=validation,
            metadata={
                "frequency_sector": "zero_matsubara",
                "source": "unresolved diagnostic response payload",
                "diagnostic_reconstruction_only": True,
            },
        )
        return static_sheet_response_to_reflection(
            response,
            q_lab_model=q_lab,
            theta_rad=float(theta_rad),
            lattice_constant_m=reflection_policy.lattice_constant_m,
            q_match_tolerance=reflection_policy.q_match_tolerance,
            require_physical=True,
        )

    if sector != "positive_matsubara":
        raise ValueError(f"unsupported diagnostic frequency sector: {sector!r}")
    validation = _positive_validation(validation_payload)
    if not validation.passed:
        raise ValueError("reconstructed positive-frequency validation does not pass")
    matrix_crystal = _complex_matrix(
        response_payload["matrix_tilde_real"],
        response_payload["matrix_tilde_imag"],
    )
    matrix_lab = tensor_crystal_to_lab(matrix_crystal, float(theta_rad))
    matrix_lt = tensor_xy_to_lt(
        matrix_lab,
        float(q_lab[0]),
        float(q_lab[1]),
    )
    lattice = reflection_policy.lattice_constant_m
    qx_si, qy_si, q_si = model_q_to_si_wavevector(
        float(q_lab[0]),
        float(q_lab[1]),
        lattice,
        lattice,
    )
    xi_eV = float.fromhex(str(sample["xi_eV_hex"]))
    xi_si = omega_eV_to_xi_si(xi_eV)
    kappa = vacuum_kappa(q_si, xi_si)
    reflection = tangential_electric_reflection_matrix_LT(
        matrix_lt,
        xi_si,
        kappa,
    )
    return SheetReflection(
        matrix_lt=reflection,
        sigma_tilde_lt=matrix_lt,
        q_lab_model=np.asarray(q_lab, dtype=float),
        q_lab_si_m_inv=np.asarray([qx_si, qy_si], dtype=float),
        xi_eV=xi_eV,
        xi_si_s_inv=xi_si,
        kappa_m_inv=kappa,
        theta_rad=float(theta_rad),
        basis=LAB_LT_TANGENTIAL_E_BASIS,
        sheet_validation=validation,
        metadata={
            "frequency_sector": "positive_matsubara",
            "source": "unresolved diagnostic response payload",
            "diagnostic_reconstruction_only": True,
            "q_match_relative_residual": mismatch,
        },
    )


def _direct_entries(
    campaign: Todo4QualificationCampaign,
    pairing_name: str,
) -> tuple[Todo4QualificationPlanEntry, ...]:
    entries = tuple(
        entry
        for entry in campaign.entries
        if entry.kind == "direct" and entry.pairing_name == pairing_name
    )
    if not entries:
        raise ValueError(f"campaign has no direct entries for pairing {pairing_name!r}")
    return entries


def _expected_q_frequency_keys(
    entries: tuple[Todo4QualificationPlanEntry, ...],
) -> set[tuple[tuple[str, str], int]]:
    expected: set[tuple[tuple[str, str], int]] = set()
    for entry in entries:
        plan = entry.geometry_plan
        for point in plan.points:
            first = plan.requirements[point.plate_1_requirement]
            second = plan.requirements[point.plate_2_requirement]
            expected.add((_q_hex(first.q_crystal), point.matsubara_index))
            expected.add((_q_hex(second.q_crystal), point.matsubara_index))
    return expected


def _load_samples(
    diagnostic_source_dir: Path,
    *,
    campaign_id: str,
    pairing_name: str,
    expected: set[tuple[tuple[str, str], int]],
) -> tuple[
    dict[tuple[tuple[str, str], int, int, str], dict[str, Any]],
    dict[str, Any],
]:
    root = Path(diagnostic_source_dir)
    paths = sorted(root.glob("shard_*.json"))
    if not paths:
        raise RuntimeError(f"no diagnostic shard JSON files found in {root}")

    lookup: dict[
        tuple[tuple[str, str], int, int, str],
        dict[str, Any],
    ] = {}
    source_commits: set[str] = set()
    source_plans: set[str] = set()
    ladders: set[tuple[int, ...]] = set()
    tags: set[str] = set()
    shard_count_values: set[int] = set()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("campaign_id") != campaign_id:
            raise ValueError("diagnostic source campaign_id differs from campaign")
        if not bool(payload.get("diagnostic_completed")):
            raise ValueError(f"diagnostic shard did not complete: {path}")
        if int(payload.get("error_count", 0)) != 0:
            raise ValueError(f"diagnostic shard contains execution errors: {path}")
        source_commits.add(str(payload["source_commit"]))
        source_plans.add(str(payload["plan_sha256"]))
        ladder = tuple(int(value) for value in payload["diagnostic_n_candidates"])
        ladders.add(ladder)
        tags.add(str(payload["diagnostic_ladder_tag"]))
        shard_count_values.add(int(payload["shard_count"]))

        for record in payload["records"]:
            if record.get("status") != "diagnostic_completed":
                raise ValueError(f"diagnostic record is not complete in {path}")
            if str(record["pairing_name"]) != pairing_name:
                continue
            record_q = tuple(str(value) for value in record["q_crystal_hex"])
            if len(record_q) != 2:
                raise ValueError("diagnostic record q_crystal_hex is malformed")
            for raw_n, frequency in record["frequencies"].items():
                n = int(raw_n)
                key_base = ((record_q[0], record_q[1]), n)
                if key_base not in expected:
                    continue
                for level in frequency["levels"]:
                    n_grid = int(level["N"])
                    for shift, sample in level["samples_by_shift"].items():
                        sample_q = tuple(str(value) for value in sample["q_crystal_hex"])
                        if sample_q != record_q:
                            raise ValueError("sample q differs from diagnostic record q")
                        if int(frequency["matsubara_index"]) != n:
                            raise ValueError("frequency Matsubara index is inconsistent")
                        key = (key_base[0], n, n_grid, str(shift))
                        normalized = dict(sample)
                        previous = lookup.setdefault(key, normalized)
                        if previous != normalized:
                            raise ValueError(
                                "duplicate diagnostic sample key has different content"
                            )

    if len(ladders) != 1 or len(tags) != 1:
        raise ValueError("diagnostic shards do not share one ladder and tag")
    ladder = next(iter(ladders))
    tag = next(iter(tags))
    if not ladder:
        raise ValueError("diagnostic ladder is empty")
    if len(shard_count_values) != 1:
        raise ValueError("diagnostic shards disagree on shard_count")

    expected_shift_count = None
    for q_key, n in sorted(expected):
        for n_grid in ladder:
            labels = sorted(
                key[3]
                for key in lookup
                if key[0] == q_key and key[1] == n and key[2] == n_grid
            )
            if not labels:
                raise RuntimeError(
                    "diagnostic source is incomplete for "
                    f"q={q_key}, n={n}, N={n_grid}"
                )
            if expected_shift_count is None:
                expected_shift_count = len(labels)
            if len(labels) != expected_shift_count:
                raise RuntimeError("diagnostic source has inconsistent shift counts")
    metadata = {
        "diagnostic_source_dir": str(root),
        "diagnostic_source_commits": sorted(source_commits),
        "diagnostic_source_plan_sha256": sorted(source_plans),
        "diagnostic_n_candidates": list(ladder),
        "diagnostic_ladder_tag": tag,
        "diagnostic_shard_file_count": len(paths),
        "diagnostic_declared_shard_count": next(iter(shard_count_values)),
        "diagnostic_sample_count": len(lookup),
        "shift_count": int(expected_shift_count or 0),
    }
    return lookup, metadata


def _sample_labels(
    lookup: Mapping[tuple[tuple[str, str], int, int, str], Mapping[str, Any]],
    q_key: tuple[str, str],
    matsubara_index: int,
    n_grid: int,
) -> tuple[str, ...]:
    labels = tuple(
        sorted(
            key[3]
            for key in lookup
            if key[0] == q_key
            and key[1] == matsubara_index
            and key[2] == n_grid
        )
    )
    if not labels:
        raise RuntimeError(
            f"no diagnostic shifts for q={q_key}, n={matsubara_index}, N={n_grid}"
        )
    return labels


def _adjacent_metric(
    rows_by_n: Mapping[int, list[dict[str, Any]]],
    field: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    n_values = sorted(rows_by_n)
    for previous_n, current_n in zip(n_values, n_values[1:]):
        previous = {
            (str(row["plate_1_shift"]), str(row["plate_2_shift"])): float(row[field])
            for row in rows_by_n[previous_n]
        }
        current = {
            (str(row["plate_1_shift"]), str(row["plate_2_shift"])): float(row[field])
            for row in rows_by_n[current_n]
        }
        if set(previous) != set(current):
            raise ValueError("adjacent N levels have different shift-pair sets")
        deltas = [abs(current[key] - previous[key]) for key in sorted(previous)]
        scale = max(
            max(abs(value) for value in previous.values()),
            max(abs(value) for value in current.values()),
            1e-300,
        )
        result.append(
            {
                "previous_N": previous_n,
                "current_N": current_n,
                "shift_pair_count": len(deltas),
                "maximum_absolute_delta": float(max(deltas)),
                "relative_delta_to_joint_max_abs": float(max(deltas) / scale),
            }
        )
    return result


def build_observable_impact_calibration(
    campaign: Todo4QualificationCampaign,
    *,
    diagnostic_source_dir: Path,
    pairing_name: str = "dwave",
) -> dict[str, Any]:
    """Build local logdet-spread evidence from persisted unresolved diagnostics."""

    if not isinstance(campaign, Todo4QualificationCampaign):
        raise TypeError("campaign must be a Todo4QualificationCampaign")
    pairing = str(pairing_name)
    entries = _direct_entries(campaign, pairing)
    expected = _expected_q_frequency_keys(entries)
    samples, source_metadata = _load_samples(
        diagnostic_source_dir,
        campaign_id=campaign.campaign_id,
        pairing_name=pairing,
        expected=expected,
    )
    n_candidates = tuple(int(value) for value in source_metadata["diagnostic_n_candidates"])

    reflection_cache: dict[tuple[Any, ...], Any] = {}
    pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for entry in entries:
        plan = entry.geometry_plan
        for point in plan.points:
            first_requirement = plan.requirements[point.plate_1_requirement]
            second_requirement = plan.requirements[point.plate_2_requirement]
            q_first = _q_hex(first_requirement.q_crystal)
            q_second = _q_hex(second_requirement.q_crystal)
            n = int(point.matsubara_index)

            for n_grid in n_candidates:
                first_labels = _sample_labels(samples, q_first, n, n_grid)
                second_labels = _sample_labels(samples, q_second, n, n_grid)
                control_labels = _sample_labels(samples, q_first, n, n_grid)
                if second_labels != control_labels:
                    raise ValueError(
                        "actual and parallel-control plate-2 shift labels differ"
                    )
                first_sample_list = [
                    samples[(q_first, n, n_grid, label)]
                    for label in first_labels
                ]
                second_sample_list = [
                    samples[(q_second, n, n_grid, label)]
                    for label in second_labels
                ]

                for separation in plan.separations_m:
                    local_rows: list[dict[str, Any]] = []
                    for shift_1 in first_labels:
                        sample_1 = samples[(q_first, n, n_grid, shift_1)]
                        cache_key_1 = (
                            q_first,
                            n,
                            n_grid,
                            shift_1,
                            _q_hex(point.q_lab),
                            float(point.theta_1_rad).hex(),
                        )
                        reflection_1 = reflection_cache.get(cache_key_1)
                        if reflection_1 is None:
                            reflection_1 = _diagnostic_reflection(
                                sample_1,
                                q_lab=point.q_lab,
                                theta_rad=point.theta_1_rad,
                                entry=entry,
                            )
                            reflection_cache[cache_key_1] = reflection_1

                        for shift_2 in second_labels:
                            sample_2 = samples[(q_second, n, n_grid, shift_2)]
                            cache_key_2 = (
                                q_second,
                                n,
                                n_grid,
                                shift_2,
                                _q_hex(point.q_lab),
                                float(point.theta_2_rad).hex(),
                            )
                            reflection_2 = reflection_cache.get(cache_key_2)
                            if reflection_2 is None:
                                reflection_2 = _diagnostic_reflection(
                                    sample_2,
                                    q_lab=point.q_lab,
                                    theta_rad=point.theta_2_rad,
                                    entry=entry,
                                )
                                reflection_cache[cache_key_2] = reflection_2

                            control_sample_2 = samples[(q_first, n, n_grid, shift_2)]
                            control_key_2 = (
                                q_first,
                                n,
                                n_grid,
                                shift_2,
                                _q_hex(point.q_lab),
                                float(point.theta_1_rad).hex(),
                            )
                            control_reflection_2 = reflection_cache.get(control_key_2)
                            if control_reflection_2 is None:
                                control_reflection_2 = _diagnostic_reflection(
                                    control_sample_2,
                                    q_lab=point.q_lab,
                                    theta_rad=point.theta_1_rad,
                                    entry=entry,
                                )
                                reflection_cache[control_key_2] = control_reflection_2

                            actual = passive_sheet_logdet(
                                reflection_1,
                                reflection_2,
                                separation_m=float(separation),
                                compatibility_tolerance=plan.policy.compatibility_tolerance,
                                eigenvalue_imag_tolerance=plan.policy.eigenvalue_imag_tolerance,
                                eigenvalue_lower_tolerance=plan.policy.eigenvalue_lower_tolerance,
                            )
                            control = passive_sheet_logdet(
                                reflection_1,
                                control_reflection_2,
                                separation_m=float(separation),
                                compatibility_tolerance=plan.policy.compatibility_tolerance,
                                eigenvalue_imag_tolerance=plan.policy.eigenvalue_imag_tolerance,
                                eigenvalue_lower_tolerance=plan.policy.eigenvalue_lower_tolerance,
                            )
                            row = {
                                "plan_id": entry.plan_id,
                                "point_id": point.point_id,
                                "matsubara_index": n,
                                "N": n_grid,
                                "separation_m": float(separation),
                                "q_lab_hex": list(_q_hex(point.q_lab)),
                                "plate_1_q_crystal_hex": list(q_first),
                                "plate_2_q_crystal_hex": list(q_second),
                                "theta_1_rad": float(point.theta_1_rad),
                                "theta_2_rad": float(point.theta_2_rad),
                                "parallel_control_theta_2_rad": float(point.theta_1_rad),
                                "plate_1_shift": shift_1,
                                "plate_2_shift": shift_2,
                                "actual_logdet": float(actual.logdet),
                                "parallel_control_logdet": float(control.logdet),
                                "angular_contrast_logdet": float(
                                    actual.logdet - control.logdet
                                ),
                                "actual_product_eigenvalues": actual.product_eigenvalues.tolist(),
                                "actual_round_trip_eigenvalues": actual.round_trip_eigenvalues.tolist(),
                                "parallel_control_product_eigenvalues": control.product_eigenvalues.tolist(),
                                "parallel_control_round_trip_eigenvalues": control.round_trip_eigenvalues.tolist(),
                                "actual_max_round_trip_eigenvalue": float(
                                    np.max(actual.round_trip_eigenvalues)
                                ),
                                "parallel_control_max_round_trip_eigenvalue": float(
                                    np.max(control.round_trip_eigenvalues)
                                ),
                            }
                            pair_rows.append(row)
                            local_rows.append(row)

                    actual_spread = _spread(
                        [float(row["actual_logdet"]) for row in local_rows]
                    )
                    control_spread = _spread(
                        [float(row["parallel_control_logdet"]) for row in local_rows]
                    )
                    contrast_spread = _spread(
                        [float(row["angular_contrast_logdet"]) for row in local_rows]
                    )
                    first_response_spread = _matrix_spread(first_sample_list)
                    second_response_spread = _matrix_spread(second_sample_list)
                    response_relative = max(
                        float(
                            first_response_spread[
                                "relative_spread_to_max_spectral_norm"
                            ]
                        ),
                        float(
                            second_response_spread[
                                "relative_spread_to_max_spectral_norm"
                            ]
                        ),
                    )
                    logdet_relative = float(
                        actual_spread["relative_spread_to_max_abs"]
                    )
                    summaries.append(
                        {
                            "plan_id": entry.plan_id,
                            "point_id": point.point_id,
                            "matsubara_index": n,
                            "N": n_grid,
                            "separation_m": float(separation),
                            "q_lab_hex": list(_q_hex(point.q_lab)),
                            "theta_1_rad": float(point.theta_1_rad),
                            "theta_2_rad": float(point.theta_2_rad),
                            "shift_pair_count": len(local_rows),
                            "plate_1_response_spread": first_response_spread,
                            "plate_2_response_spread": second_response_spread,
                            "maximum_plate_response_relative_spread": response_relative,
                            "actual_logdet_spread": actual_spread,
                            "parallel_control_logdet_spread": control_spread,
                            "angular_contrast_logdet_spread": contrast_spread,
                            "relative_spread_transfer_ratio": (
                                None
                                if response_relative == 0.0
                                else logdet_relative / response_relative
                            ),
                            "maximum_actual_round_trip_eigenvalue": max(
                                float(row["actual_max_round_trip_eigenvalue"])
                                for row in local_rows
                            ),
                            "minimum_actual_gap_to_log_branch": min(
                                1.0 - float(row["actual_max_round_trip_eigenvalue"])
                                for row in local_rows
                            ),
                            "observable_tolerance_applied": False,
                            "observable_budget_qualified": False,
                        }
                    )

    grouped: dict[
        tuple[str, str, int, float],
        dict[int, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in pair_rows:
        key = (
            str(row["plan_id"]),
            str(row["point_id"]),
            int(row["matsubara_index"]),
            float(row["separation_m"]),
        )
        grouped[key][int(row["N"])].append(row)

    adjacent: list[dict[str, Any]] = []
    for key, rows_by_n in sorted(grouped.items()):
        plan_id, point_id, matsubara_index, separation = key
        adjacent.append(
            {
                "plan_id": plan_id,
                "point_id": point_id,
                "matsubara_index": matsubara_index,
                "separation_m": separation,
                "actual_logdet": _adjacent_metric(rows_by_n, "actual_logdet"),
                "angular_contrast_logdet": _adjacent_metric(
                    rows_by_n,
                    "angular_contrast_logdet",
                ),
            }
        )

    partial_groups: dict[
        tuple[str, int, float, str, str],
        dict[int, dict[str, Any]],
    ] = defaultdict(dict)
    for row in pair_rows:
        key = (
            str(row["plan_id"]),
            int(row["N"]),
            float(row["separation_m"]),
            str(row["plate_1_shift"]),
            str(row["plate_2_shift"]),
        )
        partial_groups[key][int(row["matsubara_index"])] = row

    partial_rows: list[dict[str, Any]] = []
    for key, by_index in sorted(partial_groups.items()):
        if 0 not in by_index or 1 not in by_index:
            continue
        plan_id, n_grid, separation, shift_1, shift_2 = key
        zero = by_index[0]
        positive = by_index[1]
        partial_rows.append(
            {
                "plan_id": plan_id,
                "N": n_grid,
                "separation_m": separation,
                "plate_1_shift": shift_1,
                "plate_2_shift": shift_2,
                "matsubara_indices": [0, 1],
                "matsubara_weights": {"0": 0.5, "1": 1.0},
                "actual_weighted_logdet": (
                    0.5 * float(zero["actual_logdet"])
                    + float(positive["actual_logdet"])
                ),
                "parallel_control_weighted_logdet": (
                    0.5 * float(zero["parallel_control_logdet"])
                    + float(positive["parallel_control_logdet"])
                ),
                "angular_contrast_weighted_logdet": (
                    0.5 * float(zero["angular_contrast_logdet"])
                    + float(positive["angular_contrast_logdet"])
                ),
            }
        )

    partial_summary_groups: dict[
        tuple[str, int, float],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in partial_rows:
        partial_summary_groups[
            (
                str(row["plan_id"]),
                int(row["N"]),
                float(row["separation_m"]),
            )
        ].append(row)
    partial_summaries: list[dict[str, Any]] = []
    for key, rows in sorted(partial_summary_groups.items()):
        plan_id, n_grid, separation = key
        partial_summaries.append(
            {
                "plan_id": plan_id,
                "N": n_grid,
                "separation_m": separation,
                "shift_pair_count": len(rows),
                "actual_weighted_logdet_spread": _spread(
                    [float(row["actual_weighted_logdet"]) for row in rows]
                ),
                "parallel_control_weighted_logdet_spread": _spread(
                    [
                        float(row["parallel_control_weighted_logdet"])
                        for row in rows
                    ]
                ),
                "angular_contrast_weighted_logdet_spread": _spread(
                    [
                        float(row["angular_contrast_weighted_logdet"])
                        for row in rows
                    ]
                ),
            }
        )

    partial_adjacent_groups: dict[
        tuple[str, float],
        dict[int, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in partial_rows:
        partial_adjacent_groups[
            (str(row["plan_id"]), float(row["separation_m"]))
        ][int(row["N"])].append(row)
    partial_adjacent: list[dict[str, Any]] = []
    for key, rows_by_n in sorted(partial_adjacent_groups.items()):
        plan_id, separation = key
        partial_adjacent.append(
            {
                "plan_id": plan_id,
                "separation_m": separation,
                "actual_weighted_logdet": _adjacent_metric(
                    rows_by_n,
                    "actual_weighted_logdet",
                ),
                "angular_contrast_weighted_logdet": _adjacent_metric(
                    rows_by_n,
                    "angular_contrast_weighted_logdet",
                ),
            }
        )

    return {
        "schema": TODO4_OBSERVABLE_IMPACT_CALIBRATION_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": campaign.manifest_sha256,
        "pairing_name": pairing,
        "source_diagnostics": source_metadata,
        "summary": {
            "direct_plan_count": len(entries),
            "diagnostic_N_count": len(n_candidates),
            "diagnostic_N_candidates": list(n_candidates),
            "pair_shift_evaluation_count": len(pair_rows),
            "local_observable_summary_count": len(summaries),
            "adjacent_observable_group_count": len(adjacent),
            "partial_matsubara_pair_count": len(partial_rows),
            "partial_matsubara_summary_count": len(partial_summaries),
        },
        "local_shift_pair_records": pair_rows,
        "local_observable_summaries": summaries,
        "adjacent_N_observable_summaries": adjacent,
        "partial_matsubara_records": partial_rows,
        "partial_matsubara_summaries": partial_summaries,
        "partial_matsubara_adjacent_N_summaries": partial_adjacent,
        "contract": {
            "diagnostic_responses_reconstructed_from_persisted_evidence": True,
            "exact_reflection_geometry_used": True,
            "independent_plate_shift_pairs_evaluated": True,
            "parallel_control_uses_theta_2_equal_theta_1": True,
            "microscopic_integration_performed": False,
            "response_certification_performed": False,
            "certified_response_cache_read_attempted": False,
            "certified_response_cache_write_attempted": False,
            "diagnostic_response_promoted": False,
            "observable_tolerance_applied": False,
            "observable_error_budget_calibrated": False,
        },
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }


def write_observable_impact_calibration(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    diagnostic_source_dir: Path,
    pairing_name: str = "dwave",
) -> dict[str, Any]:
    frozen = require_frozen_plan(campaign, output_dir)
    payload = build_observable_impact_calibration(
        campaign,
        diagnostic_source_dir=diagnostic_source_dir,
        pairing_name=pairing_name,
    )
    payload["source_commit"] = source_commit()
    payload["current_plan_sha256"] = frozen["plan_sha256"]
    destination = (
        Path(output_dir)
        / "observable_impact"
        / str(payload["source_diagnostics"]["diagnostic_ladder_tag"])
        / f"{pairing_name}.json"
    )
    atomic_write_json(destination, payload)
    return payload


__all__ = [
    "TODO4_OBSERVABLE_IMPACT_CALIBRATION_SCHEMA",
    "build_observable_impact_calibration",
    "write_observable_impact_calibration",
]
