from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._diagnostic_io import atomic_json, finite_number, mapping, read_json, sequence
from .outer_tail_diagnostics import replay_outer_tail_cache_only
from .point_diagnostics import point_cache_diagnostics

_DIAGNOSTIC_SCHEMA = "casimir-run-diagnostics-v1"


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
        lines.append(
            "  unresolved "
            f"pairing={row.get('identity', ['?'])[0]} "
            f"n={row.get('identity', ['?', '?'])[1]} "
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
        for pairing, channel in mapping(metrics.get("pairings")).items():
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
                    f"  {pairing} n={index}: ratio={_format_number(ratio)} "
                    f"last_shell_envelope={_format_number(shell)} "
                    f"finite_error={_format_number(finite_error)}"
                )
    return "\n".join(lines)


def diagnose_run(
    run_dir: Path,
    *,
    replay_outer_tail: bool = False,
    output_name: str = "diagnostics.json",
) -> tuple[dict[str, Any], Path]:
    run = Path(run_dir)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.diagnostics",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["diagnose_run", "format_diagnostic_summary", "main"]
