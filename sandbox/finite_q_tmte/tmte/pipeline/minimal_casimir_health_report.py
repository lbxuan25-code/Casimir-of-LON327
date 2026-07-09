"""Offline health report for minimal Casimir diagnostic artifacts.

The health report is a credibility gate for existing result artifacts.  It does
not run BdG, does not schedule scans, and does not convert sandbox diagnostics
into production Casimir inputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io.writers import write_json

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_health_report_v1"
DEFAULT_R_NORM_WARNING_THRESHOLD = 2.0
DEFAULT_RDIFF_WARNING_THRESHOLD = 2.0
DEFAULT_PHI_RANGE_WARNING_THRESHOLD = 1.0e-3

FINDING_CSV_COLUMNS = [
    "source_path",
    "record_type",
    "record_index",
    "health_status",
    "severity",
    "classification",
    "recommended_action",
    "max_R_norm",
    "Rdiff",
    "phi_range",
    "finite_ok",
    "kappa_ok",
    "evidence",
]

SUMMARY_NUMERIC_KEYS = (
    "max_Rdiff",
    "max_Rdiff_over_q",
    "max_Rdiff_over_nq",
    "max_Rdiff_over_known_terms",
    "max_Rdiff_all_theta",
    "range_logdet_abs",
    "max_range_phi_logdet_abs",
    "max_range_phi_logdet_abs_over_nq",
    "max_range_phi_logdet_abs_over_known_terms",
    "max_phi_range",
    "max_phi_range_all_theta",
)
FINITE_KEYS = (
    "all_finite_logdet",
    "finite_logdet",
    "all_finite_R1",
    "finite_R1",
    "all_finite_R2",
    "finite_R2",
    "all_finite_logdet_known_terms",
    "all_finite_logdet_all_theta",
)
KAPPA_KEYS = (
    "all_kappa_match",
    "kappa_match",
    "all_kappa_match_known_terms",
    "all_kappa_match_all_theta",
)
RDIFF_KEYS = (
    "Rdiff",
    "max_Rdiff",
    "max_Rdiff_over_q",
    "max_Rdiff_over_nq",
    "max_Rdiff_over_known_terms",
    "max_Rdiff_all_theta",
)
PHI_RANGE_KEYS = (
    "range_phi_logdet_abs",
    "max_range_phi_logdet_abs",
    "max_range_phi_logdet_abs_over_nq",
    "max_range_phi_logdet_abs_over_known_terms",
    "max_phi_range",
    "max_phi_range_all_theta",
    "range_logdet_abs",
)
R_NORM_KEYS = (
    "max_R_norm",
    "R_norm",
    "R1_norm",
    "R2_norm",
)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _max_of_keys(record: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    values = [_as_float(record.get(key)) for key in keys]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return max(finite_values)


def _all_known_bools(record: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    values = [_as_bool(record.get(key)) for key in keys]
    known = [value for value in values if value is not None]
    if not known:
        return None
    return all(known)


def _has_health_columns(record: Mapping[str, Any]) -> bool:
    keys = set(record.keys())
    health_keys = set(FINITE_KEYS) | set(KAPPA_KEYS) | set(RDIFF_KEYS) | set(PHI_RANGE_KEYS) | set(R_NORM_KEYS)
    return bool(keys & health_keys)


def _classify_record(
    *,
    source_path: str,
    record_type: str,
    record_index: int | str,
    record: Mapping[str, Any],
    r_norm_warning_threshold: float,
    rdiff_warning_threshold: float,
    phi_range_warning_threshold: float,
) -> dict[str, Any] | None:
    if not _has_health_columns(record):
        return None

    finite_ok = _all_known_bools(record, FINITE_KEYS)
    kappa_ok = _all_known_bools(record, KAPPA_KEYS)
    max_r_norm = _max_of_keys(record, R_NORM_KEYS)
    rdiff = _max_of_keys(record, RDIFF_KEYS)
    phi_range = _max_of_keys(record, PHI_RANGE_KEYS)

    labels: list[str] = []
    evidence: list[str] = []
    recommended_actions: list[str] = []
    severity = "clean"

    if finite_ok is False:
        labels.append("nonfinite_result")
        evidence.append("finite flag is false")
        recommended_actions.append("treat as hard failure; rerun or inspect numerical construction")
        severity = "critical"
    if kappa_ok is False:
        labels.append("kappa_mismatch")
        evidence.append("kappa_match flag is false")
        recommended_actions.append("inspect q/geometry matching before using this result")
        severity = "critical"

    if max_r_norm is not None and max_r_norm > r_norm_warning_threshold:
        labels.append("reflection_norm_pathology")
        evidence.append(f"max_R_norm={max_r_norm:.8g} > {r_norm_warning_threshold:.8g}")
        recommended_actions.append("run local nk/shift/q/phi convergence probe; do not trust as physical until classified")
        if severity != "critical":
            severity = "warning"

    if rdiff is not None and rdiff > rdiff_warning_threshold:
        labels.append("large_Rdiff")
        evidence.append(f"Rdiff={rdiff:.8g} > {rdiff_warning_threshold:.8g}")
        recommended_actions.append("inspect reflection mismatch; check whether R_norm is benign or pathological")
        if severity == "clean":
            severity = "warning"

    if phi_range is not None and phi_range > phi_range_warning_threshold:
        labels.append("large_phi_range")
        evidence.append(f"phi_range={phi_range:.8g} > {phi_range_warning_threshold:.8g}")
        recommended_actions.append("check whether angular variation is expected, under-resolved, or a local artifact")
        if severity == "clean":
            severity = "warning"

    if not labels:
        labels.append("clean")
        recommended_actions.append("no health issue detected by configured thresholds")

    if severity == "critical":
        health_status = "fail"
    elif severity == "warning":
        health_status = "needs_review"
    else:
        health_status = "pass"

    return {
        "source_path": source_path,
        "record_type": record_type,
        "record_index": record_index,
        "health_status": health_status,
        "severity": severity,
        "classification": "+".join(labels),
        "recommended_action": "; ".join(dict.fromkeys(recommended_actions)),
        "max_R_norm": max_r_norm,
        "Rdiff": rdiff,
        "phi_range": phi_range,
        "finite_ok": finite_ok,
        "kappa_ok": kappa_ok,
        "evidence": "; ".join(evidence),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _json_records(path: Path, payload: Mapping[str, Any]) -> list[tuple[str, int | str, Mapping[str, Any]]]:
    records: list[tuple[str, int | str, Mapping[str, Any]]] = []
    for key in ("summary", "comparison", "status"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            records.append((key, key, value))
    for key in ("rows", "term_rows", "tail_fit_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    records.append((key, index, item))
    return records


def _csv_records(path: Path) -> list[tuple[str, int, Mapping[str, Any]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [("csv_row", index, row) for index, row in enumerate(reader)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _markdown_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Minimal Casimir health report",
        "",
        "This report is an offline credibility gate for existing result artifacts. It does not run BdG or schedule scans.",
        "",
        "## Summary",
        "",
        f"- health_status: `{summary['health_status']}`",
        f"- num_input_json: `{summary['num_input_json']}`",
        f"- num_input_csv: `{summary['num_input_csv']}`",
        f"- num_findings: `{summary['num_findings']}`",
        f"- num_fail: `{summary['num_fail']}`",
        f"- num_needs_review: `{summary['num_needs_review']}`",
        f"- num_pass: `{summary['num_pass']}`",
        f"- max_R_norm_observed: `{summary['max_R_norm_observed']}`",
        f"- max_Rdiff_observed: `{summary['max_Rdiff_observed']}`",
        f"- max_phi_range_observed: `{summary['max_phi_range_observed']}`",
        f"- valid_for_casimir_input: `{summary['valid_for_casimir_input']}`",
        "",
        "## Guardrails",
        "",
        "```text",
        "credibility gate only",
        "does not rerun BdG",
        "does not define a production Casimir policy",
        "does not convert diagnostics to Casimir input",
        "valid_for_casimir_input: False",
        "```",
        "",
        "## Findings needing attention",
        "",
    ]
    findings = payload["findings"]
    attention = [finding for finding in findings if finding["health_status"] != "pass"]
    if not attention:
        lines.append("No findings exceeded the configured thresholds.")
    else:
        for finding in attention[:100]:
            lines.extend(
                [
                    f"### {finding['health_status']}: {finding['classification']}",
                    "",
                    f"- source: `{finding['source_path']}`",
                    f"- record: `{finding['record_type']}[{finding['record_index']}]`",
                    f"- max_R_norm: `{finding['max_R_norm']}`",
                    f"- Rdiff: `{finding['Rdiff']}`",
                    f"- phi_range: `{finding['phi_range']}`",
                    f"- finite_ok: `{finding['finite_ok']}`",
                    f"- kappa_ok: `{finding['kappa_ok']}`",
                    f"- evidence: {finding['evidence'] or 'n/a'}",
                    f"- recommended_action: {finding['recommended_action']}",
                    "",
                ]
            )
        if len(attention) > 100:
            lines.append(f"... truncated {len(attention) - 100} additional attention findings in markdown; see CSV/JSON.")
    return "\n".join(lines) + "\n"


def run_minimal_casimir_health_report(
    *,
    input_json_paths: Sequence[Path] = (),
    input_csv_paths: Sequence[Path] = (),
    r_norm_warning_threshold: float = DEFAULT_R_NORM_WARNING_THRESHOLD,
    rdiff_warning_threshold: float = DEFAULT_RDIFF_WARNING_THRESHOLD,
    phi_range_warning_threshold: float = DEFAULT_PHI_RANGE_WARNING_THRESHOLD,
) -> dict[str, Any]:
    if not input_json_paths and not input_csv_paths:
        raise ValueError("provide at least one JSON or CSV artifact")
    findings: list[dict[str, Any]] = []

    for path_like in input_json_paths:
        path = Path(path_like)
        payload = _read_json(path)
        for record_type, record_index, record in _json_records(path, payload):
            finding = _classify_record(
                source_path=str(path),
                record_type=record_type,
                record_index=record_index,
                record=record,
                r_norm_warning_threshold=r_norm_warning_threshold,
                rdiff_warning_threshold=rdiff_warning_threshold,
                phi_range_warning_threshold=phi_range_warning_threshold,
            )
            if finding is not None:
                findings.append(finding)

    for path_like in input_csv_paths:
        path = Path(path_like)
        for record_type, record_index, record in _csv_records(path):
            finding = _classify_record(
                source_path=str(path),
                record_type=record_type,
                record_index=record_index,
                record=record,
                r_norm_warning_threshold=r_norm_warning_threshold,
                rdiff_warning_threshold=rdiff_warning_threshold,
                phi_range_warning_threshold=phi_range_warning_threshold,
            )
            if finding is not None:
                findings.append(finding)

    num_fail = sum(1 for finding in findings if finding["health_status"] == "fail")
    num_needs_review = sum(1 for finding in findings if finding["health_status"] == "needs_review")
    num_pass = sum(1 for finding in findings if finding["health_status"] == "pass")
    if num_fail:
        health_status = "fail"
    elif num_needs_review:
        health_status = "needs_review"
    else:
        health_status = "pass"

    r_norm_values = [finding["max_R_norm"] for finding in findings if finding.get("max_R_norm") is not None]
    rdiff_values = [finding["Rdiff"] for finding in findings if finding.get("Rdiff") is not None]
    phi_range_values = [finding["phi_range"] for finding in findings if finding.get("phi_range") is not None]

    summary = {
        "health_status": health_status,
        "num_input_json": len(input_json_paths),
        "num_input_csv": len(input_csv_paths),
        "num_findings": len(findings),
        "num_fail": num_fail,
        "num_needs_review": num_needs_review,
        "num_pass": num_pass,
        "max_R_norm_observed": None if not r_norm_values else float(max(r_norm_values)),
        "max_Rdiff_observed": None if not rdiff_values else float(max(rdiff_values)),
        "max_phi_range_observed": None if not phi_range_values else float(max(phi_range_values)),
        "thresholds": {
            "r_norm_warning_threshold": r_norm_warning_threshold,
            "rdiff_warning_threshold": rdiff_warning_threshold,
            "phi_range_warning_threshold": phi_range_warning_threshold,
        },
        "valid_for_casimir_input": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "offline_artifact_health_report_only": True,
            "credibility_gate_only": True,
            "does_not_rerun_bdg": True,
            "does_not_schedule_scans": True,
            "does_not_define_casimir_policy": True,
            "not_a_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "input": {
            "input_json_paths": [str(Path(path)) for path in input_json_paths],
            "input_csv_paths": [str(Path(path)) for path in input_csv_paths],
            "valid_for_casimir_input": False,
        },
        "summary": summary,
        "findings": findings,
        "interpretation_guardrails": {
            "intended_for_future_result_credibility_checks": True,
            "not_intended_for_bulk_sandbox_scanning": True,
            "health_status_pass_does_not_mean_physical_correctness": True,
            "health_status_needs_review_requires_local_diagnosis_or_override": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_minimal_casimir_health_report(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_health_report(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_health_report.json", payload)
    _write_csv(output / "minimal_casimir_health_report_findings.csv", payload["findings"], FINDING_CSV_COLUMNS)
    (output / "minimal_casimir_health_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload
