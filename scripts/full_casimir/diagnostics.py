from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._diagnostic_io import atomic_json, finite_number, mapping, read_json, sequence, sha256
from .budget_audit import (
    audit_completion_ledger,
    audit_evidence_gaps,
    budget_fraction_sensitivity,
    candidate_policy_screen,
    tail_resolution_audit,
)
from .candidate_replay_audit import (
    candidate_cache_replay_audit,
    production_equivalence_audit,
)
from .outer_tail_diagnostics import _config_from_payload, replay_outer_tail_cache_only
from .point_diagnostics import point_cache_diagnostics, tolerance_replay_audit
from .policy_audit import compare_policy_snapshots

_DIAGNOSTIC_SCHEMA = "casimir-run-diagnostics-v1"
_AUDIT_SCHEMA = "full-casimir-convergence-audit-v2"
_DEFAULT_CANDIDATE_RTOLS = (0.0015, 0.00175, 0.002, 0.0025, 0.003)


def _format_number(value: Any) -> str:
    number = finite_number(value)
    return "n/a" if number is None else f"{number:.8g}"


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
            f"pairing={pairing} n={n_value} q={row.get('q_model')} "
            f"latest_N={blocker.get('latest_N')} blocker={blocker.get('classification')}"
        )
        lines.append(
            "    cross_shift: "
            f"passed={cross.get('passed')} absolute={_format_number(cross.get('absolute'))} "
            f"relative={_format_number(cross.get('relative'))}"
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
    return "\n".join(lines)


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
) -> tuple[dict[str, Any], Mapping[str, Any], Any]:
    artifacts = _read_run_artifacts(run)
    summary = artifacts["summary"]
    manifest = artifacts["manifest"]
    result = artifacts["result"]
    config = artifacts["config"]
    cache = artifacts["cache"]
    cache_path = artifacts["cache_path"]
    points = point_cache_diagnostics(cache)
    config_object = _config_from_payload(config, point_cache_path=cache_path)
    point_config = config_object.outer_tail_config.joint_config.radial_config.point_config
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
        "production_replay_equivalence": production_equivalence_audit(
            cache,
            source_logdet_rtol=point_config.logdet_rtol,
            source_logdet_atol=point_config.logdet_atol,
            required_consecutive_passes=point_config.required_consecutive_passes,
        ),
        "tolerance_replay": tolerance_replay_audit(
            cache,
            candidate_logdet_rtols=candidate_logdet_rtols,
            logdet_atol=point_config.logdet_atol,
            required_consecutive_passes=point_config.required_consecutive_passes,
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
                    "The source cache is not modified. Candidate-policy projection is "
                    "performed separately in a temporary cache."
                ),
            }
    return report, config, config_object


def _default_audit_output(run_dirs: Sequence[Path]) -> Path:
    first = Path(run_dirs[0])
    if len(run_dirs) == 1:
        return first / "reports" / "convergence_audit.json"
    if all(Path(run).parent == first.parent for run in run_dirs):
        return first.parent.parent / "reports" / "convergence_audit.json"
    return Path("convergence_audit.json")


def _candidate_run_state(candidate: Mapping[str, Any], run_dir: str) -> Mapping[str, Any]:
    for raw in sequence(candidate.get("runs")):
        row = mapping(raw)
        if str(row.get("run_dir")) == run_dir:
            return row
    return {}


def audit_runs(
    run_dirs: Sequence[Path],
    *,
    candidate_logdet_rtols: Sequence[float] = _DEFAULT_CANDIDATE_RTOLS,
    replay_outer_tail: bool = True,
    closure_candidate_rtols: Sequence[float] | None = None,
    unified_radial_budget_fraction: float = 0.8,
    output: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    runs = tuple(Path(run) for run in run_dirs)
    if not runs:
        raise ValueError("at least one run directory is required")
    run_reports: list[dict[str, Any]] = []
    named_configs: list[tuple[str, Mapping[str, Any]]] = []
    config_objects: dict[str, Any] = {}
    artifacts_by_run: dict[str, dict[str, Any]] = {}
    for run in runs:
        report, config_payload, config_object = _audit_one_run(
            run,
            candidate_logdet_rtols=candidate_logdet_rtols,
            replay_outer_tail=replay_outer_tail,
        )
        run_reports.append(report)
        named_configs.append((str(run), config_payload))
        config_objects[str(run)] = config_object
        artifacts_by_run[str(run)] = _read_run_artifacts(run)

    parity = compare_policy_snapshots(named_configs)
    candidate_screen = candidate_policy_screen(run_reports)
    passing = [
        float(mapping(row).get("logdet_rtol"))
        for row in sequence(candidate_screen.get("candidates"))
        if mapping(row).get("replay_screen_passed") is True
    ]
    selected_rtols = (
        tuple(float(value) for value in closure_candidate_rtols)
        if closure_candidate_rtols is not None
        else tuple(passing[:2])
    )
    if not selected_rtols:
        raise ValueError("no candidate tolerance passed the stored-history replay screen")

    union_N = sorted(
        {
            int(value)
            for config_object in config_objects.values()
            for value in config_object.outer_tail_config.joint_config.radial_config.point_config.N_candidates
        }
    )
    closure_replays: list[dict[str, Any]] = []
    for rtol in selected_rtols:
        candidate_summary = next(
            (
                mapping(row)
                for row in sequence(candidate_screen.get("candidates"))
                if float(mapping(row).get("logdet_rtol")) == rtol
            ),
            {},
        )
        if not candidate_summary or not bool(candidate_summary.get("replay_screen_passed")):
            continue
        for run in runs:
            run_key = str(run)
            state = _candidate_run_state(candidate_summary, run_key)
            if int(state.get("unresolved_count", 1)) != 0:
                continue
            artifacts = artifacts_by_run[run_key]
            replay = candidate_cache_replay_audit(
                run_dir=run,
                config_payload=artifacts["config"],
                cache_payload=artifacts["cache"],
                point_cache_path=artifacts["cache_path"],
                candidate_logdet_rtol=rtol,
                radial_fraction=float(unified_radial_budget_fraction),
                N_candidates_override=union_N,
            )
            for outer in sequence(replay.get("outer_runs")):
                outer_map = mapping(outer)
                if isinstance(outer_map, dict):
                    outer_map["budget_fraction_sensitivity"] = budget_fraction_sensitivity(
                        mapping(outer_map.get("outer_result"))
                    )
            closure_replays.append(replay)

    legacy_evidence = audit_evidence_gaps(
        run_reports=run_reports, policy_parity=parity
    )
    completion = audit_completion_ledger(
        run_reports=run_reports,
        closure_replays=closure_replays,
        policy_parity=parity,
    )
    report = {
        "schema": _AUDIT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "run_dirs": [str(run) for run in runs],
            "candidate_logdet_rtols": [float(value) for value in candidate_logdet_rtols],
            "closure_candidate_logdet_rtols": list(selected_rtols),
            "unified_radial_budget_fraction": float(unified_radial_budget_fraction),
            "unified_angular_budget_fraction": 1.0
            - float(unified_radial_budget_fraction),
            "unified_N_candidates": union_N,
            "hard_physical_gates_unchanged": True,
            "required_consecutive_passes_unchanged": True,
            "pairing_specific_acceptance_exceptions": False,
        },
        "runs": run_reports,
        "policy_parity": parity,
        "candidate_policy_screen": candidate_screen,
        "candidate_closure_replays": closure_replays,
        "legacy_evidence_ledger": legacy_evidence,
        "audit_completion": completion,
        "decision": {
            "status": "production_change_not_authorized",
            "reason": (
                "The audit implementation can generate weighted impact, holdout plans, "
                "conditional analytic bounds, and nonduplicating ledgers. Production "
                "authorization still requires executing the independent holdout, proving "
                "the power-metric contraction premise, and benchmarking the frozen candidate "
                "with real microscopic work."
            ),
        },
    }
    destination = Path(output) if output is not None else _default_audit_output(runs)
    atomic_json(destination, report)
    return report, destination


def format_audit_summary(report: Mapping[str, Any]) -> str:
    parity = mapping(report.get("policy_parity"))
    completion = mapping(report.get("audit_completion"))
    lines = [
        "=== convergence audit ===",
        f"runs: {len(sequence(mapping(report.get('scope')).get('run_dirs')))}",
        (
            "source pairing-blind scientific policy: "
            f"{parity.get('pairing_blind_scientific_policy')} "
            f"(differences={parity.get('scientific_policy_difference_count')})"
        ),
        f"decision: {mapping(report.get('decision')).get('status')}",
        f"closure replays: {len(sequence(report.get('candidate_closure_replays')))}",
    ]
    for run in sequence(report.get("runs")):
        row = mapping(run)
        equivalence = mapping(row.get("production_replay_equivalence"))
        lines.append(
            f"production equivalence: {row.get('run_dir')} "
            f"equivalent={equivalence.get('equivalent')} "
            f"mismatches={equivalence.get('mismatch_count')}"
        )
    for replay in sequence(report.get("candidate_closure_replays")):
        row = mapping(replay)
        lines.append(
            f"candidate replay: rtol={row.get('candidate_logdet_rtol')} "
            f"run={row.get('run_dir')} wall={_format_number(row.get('replay_wall_seconds'))}s"
        )
        for outer in sequence(row.get("outer_runs")):
            impact = mapping(mapping(outer).get("microscopic_impact"))
            lines.append(
                "  weighted microscopic: "
                f"delta={_format_number(impact.get('total_signed_delta_J_m2'))} "
                f"bound={_format_number(impact.get('total_absolute_error_bound_J_m2'))}"
            )
    lines.append(
        "framework implementation complete: "
        f"{completion.get('framework_implementation_complete')}"
    )
    lines.append("remaining execution/proof evidence:")
    for item in sequence(completion.get("missing_execution_or_proof_evidence")):
        lines.append(f"  - {item}")
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
        description="Build one fail-closed convergence audit across pairing runs.",
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--candidate-logdet-rtol",
        nargs="+",
        type=float,
        default=_DEFAULT_CANDIDATE_RTOLS,
    )
    parser.add_argument(
        "--closure-candidate-logdet-rtol",
        nargs="+",
        type=float,
        help=(
            "Candidates to run through projected-cache closure; defaults to first two "
            "stored-history replay-screen passes."
        ),
    )
    parser.add_argument(
        "--unified-radial-budget-fraction", type=float, default=0.8
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
                closure_candidate_rtols=(
                    None
                    if args.closure_candidate_logdet_rtol is None
                    else tuple(args.closure_candidate_logdet_rtol)
                ),
                unified_radial_budget_fraction=float(
                    args.unified_radial_budget_fraction
                ),
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
