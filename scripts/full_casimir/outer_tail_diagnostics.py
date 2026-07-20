from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from lno327.casimir.adaptive_matsubara_tail import run_adaptive_matsubara_casimir
from lno327.casimir.adaptive_outer_tail import run_adaptive_outer_tail_casimir
from lno327.casimir.certified_point_provider import (
    FrequencyExtendableCertifiedOuterQProvider,
)
from lno327.casimir.production import build_full_casimir_config

from ._diagnostic_io import config_difference_paths, mapping, sequence, sha256


def _config_from_payload(payload: Mapping[str, Any], *, point_cache_path: Path):
    outer = mapping(payload.get("outer_tail_config"))
    joint = mapping(outer.get("joint_config"))
    radial = mapping(joint.get("radial_config"))
    point = mapping(radial.get("point_config"))
    config = build_full_casimir_config(
        pairings=tuple(str(value) for value in point["pairings"]),
        temperature_K=float(point["temperature_K"]),
        separation_nm=float(point["separation_nm"]),
        plate_angles_deg=tuple(float(value) for value in point["plate_angles_deg"]),
        delta0_eV=float(point["delta0_eV"]),
        eta_eV=float(point["eta_eV"]),
        degeneracy=float(point["degeneracy"]),
        N_candidates=tuple(int(value) for value in point["N_candidates"]),
        required_consecutive_passes=int(point["required_consecutive_passes"]),
        logdet_rtol=float(point["logdet_rtol"]),
        logdet_atol=float(point["logdet_atol"]),
        workers=int(point["workers"]),
        parallel_mode=str(point["parallel_mode"]),
        memory_budget_gb=float(point["memory_budget_gb"]),
        max_context_workers=int(point["max_context_workers"]),
        cutoff_u_values=tuple(float(value) for value in outer["cutoff_u_values"]),
        outer_tail_start_u=float(outer["tail_start_u"]),
        outer_tail_window_shells=int(outer["tail_window_shells"]),
        outer_tail_ratio_max=float(outer["tail_ratio_max"]),
        matsubara_cutoff_values=tuple(int(value) for value in payload["matsubara_cutoff_values"]),
        matsubara_tail_start_n=int(payload["tail_start_n"]),
        matsubara_tail_window_terms=int(payload["tail_window_terms"]),
        matsubara_tail_ratio_max=float(payload["tail_ratio_max"]),
        total_free_energy_rtol=float(payload["total_free_energy_rtol"]),
        total_free_energy_atol_J_m2=float(payload["total_free_energy_atol_J_m2"]),
        max_total_microscopic_q_nodes=int(outer["max_total_microscopic_q_nodes"]),
        max_total_microscopic_point_entries=int(payload["max_total_microscopic_point_entries"]),
        certifier_q_batch_size=int(payload["certifier_q_batch_size"]),
        point_cache_path=point_cache_path,
    )
    expected = dict(payload)
    expected["point_cache_path"] = str(point_cache_path)
    differences = config_difference_paths(expected, config.as_dict())
    if differences:
        preview = ", ".join(differences[:12])
        suffix = "" if len(differences) <= 12 else f" (+{len(differences) - 12} more)"
        raise ValueError(
            "stored run configuration cannot be reproduced by the canonical builder; "
            f"different paths: {preview}{suffix}"
        )
    return config


def _forbid_new_certification(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "diagnostic replay encountered a cache miss; source cache is incomplete for "
        "the stored run configuration, and new microscopic work is forbidden"
    )


def outer_tail_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = mapping(payload.get("config"))
    shells = [mapping(record) for record in sequence(payload.get("shell_records"))]
    cutoff_records = [mapping(record) for record in sequence(payload.get("cutoff_records"))]
    window_size = int(config.get("tail_window_shells", 0))
    tail_start = float(config.get("tail_start_u", 0.0))
    ratio_max = float(config.get("tail_ratio_max", math.nan))
    eligible = [record for record in shells if float(record.get("left_u", -math.inf)) >= tail_start]
    window = eligible[-window_size:] if window_size > 0 else []
    output: dict[str, Any] = {
        "shell_count": len(shells),
        "eligible_shell_count": len(eligible),
        "tail_window_shells": window_size,
        "tail_start_u": tail_start,
        "tail_ratio_max": ratio_max,
        "window_available": bool(window_size >= 2 and len(window) == window_size),
        "window_left_u": None,
        "window_right_u": None,
        "equal_shell_widths": False,
        "pairings": {},
    }
    if not output["window_available"]:
        output["dominant_failure"] = "outer_tail_window_not_established"
        return output
    widths = [float(record["width_u"]) for record in window]
    equal_widths = bool(
        np.allclose(
            np.asarray(widths, dtype=float),
            widths[-1],
            rtol=float(config.get("shell_width_rtol", 0.0)),
            atol=float(config.get("shell_width_atol", 0.0)),
        )
    )
    output.update(
        {
            "window_left_u": float(window[0]["left_u"]),
            "window_right_u": float(window[-1]["right_u"]),
            "shell_widths_u": widths,
            "equal_shell_widths": equal_widths,
        }
    )
    if not cutoff_records:
        output["dominant_failure"] = "outer_cutoff_records_missing"
        return output
    latest = cutoff_records[-1]
    current_pairings = mapping(latest.get("pairing_results"))
    finite_errors_by_pairing = mapping(latest.get("finite_domain_error_bounds_J_m2"))
    total_rtol = float(config.get("total_outer_rtol", 0.0))
    total_atol = float(config.get("total_outer_atol_J_m2", 0.0))
    finite_fraction = float(config.get("finite_domain_budget_fraction", 0.0))
    tail_fraction = float(config.get("tail_budget_fraction", 0.0))
    decay_all = finite_all = tail_all = total_all = equal_widths
    for pairing, current_raw in current_pairings.items():
        current = mapping(current_raw)
        indices = [int(value) for value in sequence(current.get("matsubara_indices"))]
        values = np.asarray(current.get("contributions_J_m2", []), dtype=float)
        finite_errors = np.asarray(finite_errors_by_pairing.get(pairing, []), dtype=float)
        amplitudes = np.asarray(
            [
                mapping(mapping(record.get("pairings")).get(pairing)).get(
                    "shell_envelope_amplitudes_J_m2", []
                )
                for record in window
            ],
            dtype=float,
        )
        expected = (window_size, len(indices))
        if (
            amplitudes.shape != expected
            or values.shape != (len(indices),)
            or finite_errors.shape != (len(indices),)
        ):
            output["pairings"][str(pairing)] = {
                "status": "shape_mismatch",
                "expected_shell_shape": list(expected),
                "observed_shell_shape": list(amplitudes.shape),
                "value_shape": list(values.shape),
                "finite_error_shape": list(finite_errors.shape),
            }
            decay_all = finite_all = tail_all = total_all = False
            continue
        denominator = amplitudes[:-1]
        ratios = np.divide(
            amplitudes[1:],
            denominator,
            out=np.full_like(amplitudes[1:], np.inf),
            where=denominator > 0.0,
        )
        ratios[(denominator == 0.0) & (amplitudes[1:] == 0.0)] = 0.0
        ratio_envelopes = np.max(ratios, axis=0)
        decay_passed = ratio_envelopes <= ratio_max
        tail_bounds = amplitudes[-1] * ratio_max / (1.0 - ratio_max)
        total_tolerance = np.maximum(total_atol, total_rtol * np.abs(values))
        finite_tolerance = finite_fraction * total_tolerance
        tail_tolerance = tail_fraction * total_tolerance
        finite_passed = finite_errors <= finite_tolerance
        tail_passed = tail_bounds <= tail_tolerance
        total_errors = finite_errors + tail_bounds
        total_passed = total_errors <= total_tolerance
        with np.errstate(divide="ignore", invalid="ignore"):
            shell_to_finite_error = np.divide(
                amplitudes[-1],
                finite_errors,
                out=np.full_like(amplitudes[-1], np.inf),
                where=finite_errors > 0.0,
            )
        decay_all = decay_all and bool(np.all(decay_passed))
        finite_all = finite_all and bool(np.all(finite_passed))
        tail_all = tail_all and bool(np.all(tail_passed))
        total_all = total_all and bool(np.all(total_passed))
        output["pairings"][str(pairing)] = {
            "status": "analyzed",
            "matsubara_indices": indices,
            "shell_envelope_amplitudes_J_m2": amplitudes.tolist(),
            "observed_shell_ratios": ratios.tolist(),
            "ratio_envelopes": ratio_envelopes.tolist(),
            "decay_channel_passed": decay_passed.tolist(),
            "latest_shell_to_finite_error_ratio": shell_to_finite_error.tolist(),
            "finite_domain_error_bounds_J_m2": finite_errors.tolist(),
            "estimated_tail_bounds_J_m2": tail_bounds.tolist(),
            "total_outer_tolerances_J_m2": total_tolerance.tolist(),
            "finite_domain_budget_tolerances_J_m2": finite_tolerance.tolist(),
            "tail_budget_tolerances_J_m2": tail_tolerance.tolist(),
            "finite_domain_channel_passed": finite_passed.tolist(),
            "tail_channel_passed": tail_passed.tolist(),
            "estimated_total_outer_errors_J_m2": total_errors.tolist(),
            "total_outer_channel_passed": total_passed.tolist(),
        }
    output.update(
        {
            "decay_all_passed": decay_all,
            "finite_domain_budget_all_passed": finite_all,
            "tail_budget_all_passed": tail_all,
            "total_outer_budget_all_passed": total_all,
        }
    )
    if not equal_widths:
        dominant = "outer_tail_shell_width_contract_failed"
    elif not decay_all:
        dominant = "outer_tail_decay_ratio_not_established"
    elif not finite_all:
        dominant = "finite_domain_budget_not_met"
    elif not tail_all:
        dominant = "outer_tail_budget_not_met"
    elif not total_all:
        dominant = "total_outer_budget_not_met"
    else:
        dominant = "outer_cutoff_and_tail_tolerances_met"
    output["dominant_failure"] = dominant
    return output


def replay_outer_tail_cache_only(
    *,
    run_dir: Path,
    config_payload: Mapping[str, Any],
    point_cache_path: Path,
) -> dict[str, Any]:
    source_hash_before = sha256(point_cache_path)
    with TemporaryDirectory(prefix="casimir-diagnostics-") as temporary:
        temporary_cache = Path(temporary) / "certified_points.json"
        shutil.copy2(point_cache_path, temporary_cache)
        config = _config_from_payload(config_payload, point_cache_path=temporary_cache)
        first_cutoff = int(config.matsubara_cutoff_values[0])
        base_point = config.outer_tail_config.joint_config.radial_config.point_config
        first_point = replace(base_point, matsubara_indices=tuple(range(first_cutoff + 1)))
        provider = FrequencyExtendableCertifiedOuterQProvider(
            first_point,
            cache_path=temporary_cache,
            runner=_forbid_new_certification,
            certifier_q_batch_size=config.certifier_q_batch_size,
        )
        captured = []

        def outer_runner(outer_config, *, provider=None):
            result = run_adaptive_outer_tail_casimir(outer_config, provider=provider)
            captured.append(result)
            return result

        started = perf_counter()
        result = run_adaptive_matsubara_casimir(
            config,
            provider=provider,
            outer_tail_runner=outer_runner,
        )
        wall_seconds = perf_counter() - started
        captured_payloads = [item.as_dict() for item in captured]
    source_hash_after = sha256(point_cache_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source cache changed during supposedly read-only diagnostic replay")
    return {
        "schema": "cache-only-outer-tail-replay-v1",
        "source_run_dir": str(run_dir),
        "source_cache": str(point_cache_path),
        "source_cache_sha256": source_hash_before,
        "new_microscopic_work_forbidden": True,
        "wall_seconds": wall_seconds,
        "matsubara_result": result.as_dict(),
        "outer_tail_runs": [
            {
                "matsubara_cutoff": int(
                    payload["config"]["joint_config"]["radial_config"]
                    ["point_config"]["matsubara_indices"][-1]
                ),
                "result": payload,
                "diagnostic_metrics": outer_tail_metrics(payload),
            }
            for payload in captured_payloads
        ],
    }
