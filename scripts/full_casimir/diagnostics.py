from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._diagnostic_io import atomic_json, finite_number, mapping, read_json, sequence, sha256
from .budget_audit import (
    audit_evidence_gaps,
    candidate_policy_screen,
    tail_resolution_audit,
)
from .outer_tail_diagnostics import replay_outer_tail_cache_only
from .point_diagnostics import point_cache_diagnostics, tolerance_replay_audit
from .policy_audit import compare_policy_snapshots

_DIAGNOSTIC_SCHEMA = "casimir-run-diagnostics-v1"
_AUDIT_SCHEMA = "full-casimir-convergence-audit-v1"
_DEFAULT_CANDIDATE_RTOLS = (0.0015, 0.00175, 0.002, 0.0025, 0.003)


def _format_number(value: Any) -> str:
    number = finite_number(value)
    return "n/a" if number is None else f"{number:.8g}"


def format_diagnostic_summary(report: Mapping[str, Any]) -> str:
    artifacts = mapping(report.get("artifacts"))
    point_cache = mapping(report.get("point_cache"))
    lines = [
        f"=== {report.get('run_dir')} ===",
        f"status: {artifacts.get('summary_status', 'missing')}",
        f"termination: {artifacts.get('termination_reason', '')}",
        (
            "point cache: "
            f"entries={point_cache.get('entry_count', 0)}, "
            f"unresolved={point_cache.get('unresolved_count', 0)}"
        ),
    ]
    for point in sequence(point_cache.get("unresolved_points")):
        row = mapping(point)
        blocker = mapping(row.get("latest_blocker"))
        cross = mapping(blocker.get("cross_shift"))
        identity = sequence(row.get("identity"))
        pairing = identity[0] if len(identity) > 0 else "?"
        n_value = identity[1] if len(identity) > 1 else "?"
        lines.append(
            "  unresolved "
            f"pairing={pairing} n={n_value} "
            f"q={row.get('q_model')} "
            f"latest_N={blocker.get('latest_N')} "
            f"blocker={blocker.get('classification')}"
        )
        lines.append(
            "    cross_shift: "
            f"passed={cross.get('passed')} "
            f"absolute={_format_number(cross.get('absolute'))} "
            f"relative={_format_number(cross.get('relative'))}"
        )
        for label, failure in mapping(blocker.get("adjacent_N_failures")).items():
            failure_map = mapping(failure)
            lines.append(
                f"    adjacent {label}: passed={failure_map.get('passed')} "
                f"absolute={_format_number(failure_map.get('absolute'))} "
                f"relative={_format_number(failure_map.get('relative'))}"
            )
    replay = mapping(report.get("outer_tail_replay"))
    for outer_run in sequence(replay.get("outer_tail_runs")):
        outer = mapping(outer_run)
        metrics = mapping(outer.get("diagnostic_metrics"))
        lines.append(
            "outer-tail replay: "
            f"matsubara_cutoff={outer.get('matsubara_cutoff')} "
            f"failure={metrics.get('dominant_failure')} "
            f"window={metrics.get('window_left_u')}..{metrics.get('window_right_u')}"
        )
        for pairing_name, channel in mapping(metrics.get("pairings")).items():
            channel_map = mapping(channel)
            indices = sequence(channel_map.get("matsubara_indices"))
            ratios = sequence(channel_map.get("ratio_envelopes"))
            shells = sequence(channel_map.get("shell_envelope_amplitudes_J_m2"))
            latest_shell = sequence(shells[-1]) if shells else ()
            finite_errors = sequence(channel_map.get("finite_domain_error_bounds_J_m2"))
            for index, ratio, shell, finite_error in zip(
                indices, ratios, latest_shell, finite_errors, strict=False
            ):
                lines.append(
                    f"  {pairing_name} n={index}: ratio={_format_number(ratio)} "
                    f"last_shell_envelope={_format_number(shell)} "
                    f"finite_error={_format_number(finite_error)}"
                )
    return "\n".join(lines)


def _read_run_artifacts(run: Path) -> dict[str, Any]:
    summary = read_json(run / "summary.json", label="run summary")
    manifest = read_json(run / "manifest.json", label="run manifest")
    result = read_json(run / "result.json", label="run result")
    config = read_json(run / "config.json", label="run config")
    cache_path = run / "cache" / "certified_points.json"
    cache = read_json(cache_path, label="certified point cache")
    extension_path = run / "cache" / "extension_report.json"
    extension = (
        read_json(extension_path, label="cache extension report")
        if extension_path.is_file()
        else {}
    )
    return {
        "summary": summary,
        "manifest": manifest,
        "result": result,
        "config": config,
        "cache": cache,
        "cache_path": cache_path,
        "extension": extension,
    }


def diagnose_run(
    run_dir: Path,
    *,
    replay_outer_tail: bool = False,
    output_name: str = "diagnostics.json",
) -> tuple[dict[str, Any], Path]:
    run = Path(run_dir)
    artifacts = _read_run_artifacts(run)
    summary = artifacts["summary"]
    manifest = artifacts["manifest"]
    result = artifacts["result"]
    config = artifacts["config"]
    cache = artifacts["cache"]
    cache_path = artifacts["cache_path"]
    extension = artifacts["extension"]
    dropped = tuple(
        (str(row[0]), int(row[1]), str(row[2]), str(row[3]))
        for row in sequence(extension.get("dropped_identities"))
        if isinstance(row, list) and len(row) == 4
    )
    points = point_cache_diagnostics(cache, source_dropped_identities=dropped)
    report: dict[str, Any] = {
        "schema": _DIAGNOSTIC_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run),
        "artifacts": {
            "manifest_status": manifest.get("status"),
            "summary_status": summary.get("status"),
            "result_status": result.get("status"),
            "termination_reason": summary.get(
                "termination_reason", result.get("termination_reason")
            ),
            "selected_matsubara_cutoff": summary.get("selected_matsubara_cutoff"),
            "selected_u_max": summary.get("selected_u_max"),
            "git_commit": manifest.get("git_commit"),
        },
        "point_cache": points,
        "extension_lineage": extension,
        "existing_result_diagnostic_gaps": {
            "top_level_matsubara_cutoff_records_include_outer_shell_records": any(
                "outer_shell_records" in mapping(record)
                for record in sequence(result.get("cutoff_records"))
            ),
            "cache_only_replay_requested": bool(replay_outer_tail),
        },
    }
    if replay_outer_tail:
        if int(points["unresolved_count"]) != 0:
            raise ValueError(
                "cache-only outer-tail replay requires zero unresolved cached points; "
                f"found {points['unresolved_count']}"
            )
        report["outer_tail_replay"] = replay_outer_tail_cache_only(
            run_dir=run,
            config_payload=config,
            point_cache_path=cache_path,
        )
    output = run / "reports" / output_name
    atomic_json(output, report)
    return report, output


def _audit_one_run(
    run: Path,
    *,
    candidate_logdet_rtols: Sequence[float],
    replay_outer_tail: bool,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    artifacts = _read_run_artifacts(run)
    summary = artifacts["summary"]
    manifest = artifacts["manifest"]
    result = artifacts["result"]
    config = artifacts["config"]
    cache = artifacts["cache"]
    cache_path = artifacts["cache_path"]
    points = point_cache_diagnostics(cache)
    report: dict[str, Any] = {
        "run_dir": str(run),
        "provenance": {
            "git_commit": manifest.get("git_commit"),
            "config_sha256": sha256(run / "config.json"),
            "cache_sha256": sha256(cache_path),
            "manifest_status": manifest.get("status"),
            "summary_status": summary.get("status"),
            "result_status": result.get("status"),
            "termination_reason": summary.get(
                "termination_reason", result.get("termination_reason")
            ),
        },
        "point_cache": points,
        "tolerance_replay": tolerance_replay_audit(
            cache,
            candidate_logdet_rtols=candidate_logdet_rtols,
        ),
    }
    if replay_outer_tail:
        if int(points["unresolved_count"]) == 0:
            replay = replay_outer_tail_cache_only(
                run_dir=run,
                config_payload=config,
                point_cache_path=cache_path,
            )
            report["outer_tail_replay"] = replay
            report["outer_tail_resolution"] = tail_resolution_audit(replay)
        else:
            report["outer_tail_replay_status"] = {
                "status": "blocked_by_unresolved_point_cache",
                "unresolved_count": int(points["unresolved_count"]),
                "scientific_interpretation": (
                    "No pairing-specific exception is applied. The same tail replay will "
                    "run after this cache satisfies the selected microscopic policy."
                ),
            }
    return report, config


def _default_audit_output(run_dirs: Sequence[Path]) -> Path:
    first = Path(run_dirs[0])
    if len(run_dirs) == 1:
        return first / "reports" / "convergence_audit.json"
    if all(Path(run).parent == first.parent for run in run_dirs):
        return first.parent.parent / "reports" / "convergence_audit.json"
    return Path("convergence_audit.json")


def audit_runs(
    run_dirs: Sequence[Path],
    *,
    candidate_logdet_rtols: Sequence[float] = _DEFAULT_CANDIDATE_RTOLS,
    replay_outer_tail: bool = True,
    output: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    runs = tuple(Path(run) for run in run_dirs)
    if not runs:
        raise ValueError("at least one run directory is required")
    run_reports = []
    named_configs = []
    for run in runs:
        report, config = _audit_one_run(
            run,
            candidate_logdet_rtols=candidate_logdet_rtols,
            replay_outer_tail=replay_outer_tail,
        )
        run_reports.append(report)
        named_configs.append((str(run), config))
    parity = compare_policy_snapshots(named_configs)
    candidate_screen = candidate_policy_screen(run_reports)
    evidence = audit_evidence_gaps(run_reports=run_reports, policy_parity=parity)
    report = {
        "schema": _AUDIT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "run_dirs": [str(run) for run in runs],
            "candidate_logdet_rtols": [float(value) for value in candidate_logdet_rtols],
            "hard_physical_gates_unchanged": True,
            "required_consecutive_passes_unchanged": True,
            "pairing_specific_acceptance_exceptions": False,
        },
        "runs": run_reports,
        "policy_parity": parity,
        "candidate_policy_screen": candidate_screen,
        "evidence_ledger": evidence,
        "decision": {
            "status": (
                "production_change_not_authorized"
                if not bool(evidence.get("production_policy_change_authorized"))
                else "evidence_complete"
            ),
            "reason": (
                "The framework reports replay evidence and numerical-policy differences "
                "without converting missing quadrature, holdout, or analytic-tail evidence "
                "into an acceptance recommendation."
            ),
        },
    }
    destination = Path(output) if output is not None else _default_audit_output(runs)
    atomic_json(destination, report)
    return report, destination


def format_audit_summary(report: Mapping[str, Any]) -> str:
    parity = mapping(report.get("policy_parity"))
    evidence = mapping(report.get("evidence_ledger"))
    lines = [
        "=== convergence audit ===",
        f"runs: {len(sequence(mapping(report.get('scope')).get('run_dirs')))}",
        (
            "pairing-blind scientific policy: "
            f"{parity.get('pairing_blind_scientific_policy')} "
            f"(differences={parity.get('scientific_policy_difference_count')})"
        ),
        f"decision: {mapping(report.get('decision')).get('status')}",
        "candidate replay screen:",
    ]
    for candidate in sequence(mapping(report.get("candidate_policy_screen")).get("candidates")):
        row = mapping(candidate)
        lines.append(
            f"  rtol={row.get('logdet_rtol')}: "
            f"all_points={row.get('all_stored_points_established')} "
            f"hard_failures={row.get('hard_physical_failure_count')} "
            f"production_ready={row.get('production_ready')}"
        )
    lines.append("missing evidence:")
    for item in sequence(evidence.get("missing_evidence")):
        lines.append(f"  - {item}")
    for run in sequence(report.get("runs")):
        row = mapping(run)
        tail = mapping(row.get("outer_tail_resolution"))
        if tail:
            lines.append(f"tail resolution: {row.get('run_dir')} status={tail.get('status')}")
            for outer in sequence(tail.get("outer_tail_runs")):
                for pairing, channel in mapping(mapping(outer).get("pairings")).items():
                    classes = [
                        mapping(record).get("classification")
                        for record in sequence(mapping(channel).get("channels"))
                    ]
                    lines.append(f"  {pairing}: {classes}")
        elif row.get("outer_tail_replay_status"):
            lines.append(
                f"tail replay: {row.get('run_dir')} "
                f"{mapping(row.get('outer_tail_replay_status')).get('status')}"
            )
    return "\n".join(lines)


def _diagnose_parser(*, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Generate read-only point diagnostics and optionally replay the outer-tail "
            "controller from a temporary cache copy."
        ),
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--replay-outer-tail", action="store_true")
    parser.add_argument("--output-name", default="diagnostics.json")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.diagnostics audit",
        description=(
            "Build one fail-closed convergence audit across one or more pairing runs."
        ),
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--candidate-logdet-rtol",
        nargs="+",
        type=float,
        default=_DEFAULT_CANDIDATE_RTOLS,
    )
    parser.add_argument("--no-replay-outer-tail", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _run_diagnose(args: argparse.Namespace) -> int:
    status = 0
    for run_dir in args.run_dir:
        try:
            report, output = diagnose_run(
                run_dir,
                replay_outer_tail=bool(args.replay_outer_tail),
                output_name=str(args.output_name),
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            status = 2
            print(f"DIAGNOSTIC FAILED: {run_dir}: {type(exc).__name__}: {exc}")
            continue
        if not args.quiet:
            print(format_diagnostic_summary(report))
        print(f"written: {output}")
    return status


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv or ())
    if argv is None:
        import sys

        raw = list(sys.argv[1:])
    if raw and raw[0] == "audit":
        args = _audit_parser().parse_args(raw[1:])
        try:
            report, output = audit_runs(
                args.run_dir,
                candidate_logdet_rtols=tuple(args.candidate_logdet_rtol),
                replay_outer_tail=not bool(args.no_replay_outer_tail),
                output=args.output,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            print(f"AUDIT FAILED: {type(exc).__name__}: {exc}")
            return 2
        if not args.quiet:
            print(format_audit_summary(report))
        print(f"written: {output}")
        return 0
    if raw and raw[0] == "diagnose":
        args = _diagnose_parser(
            prog="python -m scripts.full_casimir.diagnostics diagnose"
        ).parse_args(raw[1:])
        return _run_diagnose(args)
    args = _diagnose_parser(
        prog="python -m scripts.full_casimir.diagnostics"
    ).parse_args(raw)
    return _run_diagnose(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_runs",
    "diagnose_run",
    "format_audit_summary",
    "format_diagnostic_summary",
    "main",
]
