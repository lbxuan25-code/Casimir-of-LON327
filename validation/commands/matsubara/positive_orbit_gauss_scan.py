"""Staged total Matsubara spm/d-wave scan using one composite Gauss method.

Every q case uses the same full-period equal-panel composite Gauss-Legendre rule.
Only transverse order and point budget change.  Exact ``n=0`` and all positive
Matsubara frequencies are evaluated in one batched microscopic call so eigensystems
are shared; postprocessing then branches to static density/stiffness or positive
conductivity as required.  A successful performance/correctness preflight manifest
for the current git head is mandatory before a formal run.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

DEFAULT_OUTPUT_ROOT = Path(
    "validation/outputs/matsubara/total_orbit_gauss_scan"
)
DEFAULT_PREFLIGHT = Path(
    "validation/outputs/matsubara/orbit_gauss_preflight/preflight.json"
)
DEFAULT_CASES = (
    "axis_min:1:0",
    "diagonal_min:1:1",
    "generic_small:2:1",
    "generic_mid:3:2",
    "reference:6:4",
    "axis_mid:6:0",
    "diagonal_mid:6:6",
    "generic_large:9:6",
)
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    mx: int
    my: int


@dataclass(frozen=True)
class StepMetrics:
    physical_all: bool
    observable_all: bool
    static_strict_all: bool
    positive_strict_all: bool
    positive_soft_all: bool
    strict_all: bool
    soft_all: bool
    max_primary_relative: float
    max_static_relative: float
    max_positive_relative: float
    max_reflection_relative: float
    max_logdet_relative: float
    rows: tuple[dict[str, Any], ...]


def _parse_case(text: str) -> CaseSpec:
    parts = str(text).split(":")
    if len(parts) != 3:
        raise ValueError("case must have form LABEL:MX:MY")
    label = parts[0].strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not label or any(char not in allowed for char in label):
        raise ValueError(f"invalid case label: {label!r}")
    mx, my = int(parts[1]), int(parts[2])
    if mx == 0 and my == 0:
        raise ValueError("q=(0,0) is not a finite-q Matsubara case")
    return CaseSpec(label, mx, my)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairings",
        nargs="+",
        choices=("spm", "dwave"),
        default=["spm", "dwave"],
    )
    parser.add_argument("--case", action="append", default=None, metavar="LABEL:MX:MY")
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument(
        "--matsubara-indices",
        nargs="+",
        type=int,
        default=[0, 1, 2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--gauss-orders",
        nargs="+",
        type=int,
        default=[64, 96, 160, 192, 320, 384],
        help=(
            "low/high stage pairs using one composite Gauss method; default pairs "
            "are (64,96), (160,192), (320,384)"
        ),
    )
    parser.add_argument("--panel-count", type=int, default=16)
    parser.add_argument("--transverse-workers", type=int, default=8)
    parser.add_argument("--transverse-task-size", type=int, default=4)
    parser.add_argument("--strict-sigma-rtol", type=float, default=1e-3)
    parser.add_argument("--soft-sigma-rtol", type=float, default=2e-3)
    parser.add_argument("--observable-rtol", type=float, default=1e-3)
    parser.add_argument("--soft-confirmations", type=int, default=2)
    parser.add_argument("--minimum-strict-fraction", type=float, default=0.9)
    parser.add_argument(
        "--soft-cut-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cut-audit-shift", type=float, default=float(np.pi / 32.0))
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--preflight-manifest", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument(
        "--require-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    raw_cases = args.case if args.case is not None else list(DEFAULT_CASES)
    try:
        cases = tuple(_parse_case(value) for value in raw_cases)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if len({case.label for case in cases}) != len(cases):
        parser.error("case labels must be unique")
    if len({(case.mx, case.my) for case in cases}) != len(cases):
        parser.error("q coordinates must be unique")
    args.cases = cases

    if args.nk <= 0 or args.panel_count <= 0:
        parser.error("nk and panel count must be positive")
    if args.transverse_workers <= 0 or args.transverse_task_size <= 0:
        parser.error("transverse workers and task size must be positive")
    if args.soft_confirmations <= 0:
        parser.error("soft confirmations must be positive")
    indices = tuple(sorted(set(int(value) for value in args.matsubara_indices)))
    if any(index < 0 for index in indices):
        parser.error("Matsubara indices must be non-negative")
    if 0 not in indices:
        parser.error("total pre-integration validation requires exact Matsubara n=0")
    if not any(index > 0 for index in indices):
        parser.error("total validation also requires at least one positive Matsubara index")
    args.matsubara_indices = indices

    orders = tuple(int(value) for value in args.gauss_orders)
    if len(orders) < 2 or len(orders) % 2 != 0:
        parser.error("Gauss orders must contain one or more low/high pairs")
    if any(order <= 0 or order % args.panel_count != 0 for order in orders):
        parser.error("every Gauss order must be positive and divisible by panel count")
    if any(right <= left for left, right in zip(orders[:-1], orders[1:])):
        parser.error("Gauss orders must be strictly increasing")
    args.gauss_orders = orders
    args.gauss_stages = tuple(
        (orders[index], orders[index + 1])
        for index in range(0, len(orders), 2)
    )
    args.pairings = tuple(dict.fromkeys(args.pairings))

    if not (0.0 <= args.minimum_strict_fraction <= 1.0):
        parser.error("minimum strict fraction must lie in [0,1]")
    if not (0.0 <= args.strict_sigma_rtol <= args.soft_sigma_rtol):
        parser.error("require 0 <= strict response rtol <= soft positive response rtol")
    if args.observable_rtol < 0.0 or not math.isfinite(args.observable_rtol):
        parser.error("observable rtol must be finite and non-negative")
    if not math.isfinite(args.cut_audit_shift):
        parser.error("cut audit shift must be finite")
    return args


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    path = args.preflight_manifest
    if not path.is_file():
        raise SystemExit(
            f"required preflight manifest does not exist: {path}; run "
            "python -m validation matsubara orbit-gauss-preflight first"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read preflight manifest {path}: {exc}") from exc
    status = payload.get("status", {})
    if payload.get("schema") != "total_matsubara_orbit_gauss_preflight_v1":
        raise SystemExit("preflight manifest has the wrong schema")
    if not bool(status.get("passed")) or not bool(status.get("formal_scan_allowed")):
        raise SystemExit("preflight did not pass; formal total scan is blocked")
    manifest_args = payload.get("arguments", {})
    required_matches = {
        "nk": int(args.nk),
        "panel_count": int(args.panel_count),
        "transverse_workers": int(args.transverse_workers),
        "transverse_task_size": int(args.transverse_task_size),
    }
    for name, expected in required_matches.items():
        if int(manifest_args.get(name, -1)) != expected:
            raise SystemExit(
                f"preflight {name}={manifest_args.get(name)!r} does not match "
                f"formal scan value {expected}"
            )
    covered_pairings = set(manifest_args.get("pairings", []))
    if not set(args.pairings).issubset(covered_pairings):
        raise SystemExit("preflight did not cover every requested pairing")
    preflight_indices = {int(value) for value in manifest_args.get("matsubara_indices", [])}
    if 0 not in preflight_indices or not any(value > 0 for value in preflight_indices):
        raise SystemExit("preflight did not exercise a combined zero/positive batch")
    current_head = _git_head()
    manifest_head = str(payload.get("git_head", "unknown"))
    if current_head != "unknown" and manifest_head != current_head:
        raise SystemExit(
            f"preflight git head {manifest_head} does not match current head {current_head}"
        )
    return payload


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in _THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _origin_count(case: CaseSpec, subgrid_average: str) -> int:
    common = math.gcd(abs(case.mx), abs(case.my))
    return 2 if subgrid_average == "auto" and common % 2 == 1 else 1


def _budget(args: argparse.Namespace, case: CaseSpec, order: int) -> int:
    return int(args.nk) * _origin_count(case, args.subgrid_average) * int(order)


def _output_matches(path: Path, fingerprint: str) -> bool:
    manifest = path.with_suffix(".task.json")
    if not path.is_file() or not path.with_suffix(".json").is_file() or not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("state") == "completed"
        and payload.get("fingerprint") == fingerprint
    )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run_command(
    *,
    args: argparse.Namespace,
    pairing: str,
    case: CaseSpec,
    order: int,
    output: Path,
    reference_csv: Path | None,
    integration_start: float = -np.pi,
    stage: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "validation",
        "matsubara",
        "positive-orbit-gauss-crosscheck",
        "--pairing",
        pairing,
        "--nk",
        str(args.nk),
        "--mx",
        str(case.mx),
        "--my",
        str(case.my),
        "--matsubara-indices",
        *(str(value) for value in args.matsubara_indices),
        "--gauss-orders",
        str(order),
        "--panel-count",
        str(args.panel_count),
        "--integration-start",
        str(float(integration_start)),
        "--transverse-workers",
        str(args.transverse_workers),
        "--transverse-task-size",
        str(args.transverse_task_size),
        "--shift-s",
        str(args.shift_s),
        "--subgrid-average",
        str(args.subgrid_average),
        "--max-point-evaluations",
        str(_budget(args, case, order)),
        "--temperature-K",
        str(args.temperature_K),
        "--delta0-eV",
        str(args.delta0_eV),
        "--eta-eV",
        str(args.eta_eV),
        "--degeneracy",
        str(args.degeneracy),
        "--separation-nm",
        str(args.separation_nm),
        "--ward-tolerance",
        str(args.ward_tolerance),
        "--ward-absolute-tolerance",
        str(args.ward_absolute_tolerance),
        "--condition-max",
        str(args.condition_max),
        "--reference-matrix-rtol",
        str(args.strict_sigma_rtol),
        "--reference-logdet-rtol",
        str(args.observable_rtol),
        "--output",
        str(output),
    ]
    if reference_csv is not None:
        command.extend(("--reference-csv", str(reference_csv)))

    fingerprint = _fingerprint(
        {"git_head": _git_head(), "stage": stage, "command": command[1:]}
    )
    manifest = output.with_suffix(".task.json")
    if args.resume and _output_matches(output, fingerprint):
        print(f"[skip] {pairing}/{case.label}/{stage}/C{order}", flush=True)
        return
    print(f"[run]  {pairing}/{case.label}/{stage}/C{order}", flush=True)
    print("       " + " ".join(command), flush=True)
    if args.dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        manifest,
        {
            "state": "running",
            "fingerprint": fingerprint,
            "command": command,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    started = time.perf_counter()
    log_path = output.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_environment(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{pairing}/{case.label}] {line.rstrip()}", flush=True)
        returncode = int(process.wait())
    wall = float(time.perf_counter() - started)
    completed = bool(
        returncode == 0 and output.is_file() and output.with_suffix(".json").is_file()
    )
    _write_manifest(
        manifest,
        {
            "state": "completed" if completed else "failed",
            "fingerprint": fingerprint,
            "command": command,
            "returncode": returncode,
            "wall_seconds": wall,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if not completed:
        raise RuntimeError(
            f"child failed for {pairing}/{case.label}/{stage}/C{order}; see {log_path}"
        )


def _read_rows(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("inf")


def _metrics(
    path: Path,
    *,
    sigma_strict: float,
    sigma_soft: float,
    observable: float,
) -> StepMetrics:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"no rows in {path}")
    physical_all = all(
        _as_bool(row["point_pipeline_passed"])
        and _as_bool(row["ward_passed"])
        and _as_bool(row["sheet_validation_passed"])
        and _as_bool(row["reflection_constructed"])
        and _as_bool(row["logdet_passed"])
        for row in rows
    )
    primary_values = tuple(
        _finite_float(row["reference_primary_response_relative"])
        for row in rows
    )
    static_values = tuple(
        _finite_float(row["reference_primary_response_relative"])
        for row in rows
        if int(row["matsubara_index"]) == 0
    )
    positive_values = tuple(
        _finite_float(row["reference_primary_response_relative"])
        for row in rows
        if int(row["matsubara_index"]) > 0
    )
    reflection_values = tuple(
        _finite_float(row["reference_reflection_matrix_relative"]) for row in rows
    )
    logdet_values = tuple(
        _finite_float(row["reference_logdet_relative"]) for row in rows
    )
    max_primary = max(primary_values, default=float("inf"))
    max_static = max(static_values, default=float("inf"))
    max_positive = max(positive_values, default=float("inf"))
    max_reflection = max(reflection_values, default=float("inf"))
    max_logdet = max(logdet_values, default=float("inf"))
    zero_rows = [row for row in rows if int(row["matsubara_index"]) == 0]
    static_strict = bool(
        len(zero_rows) == 1
        and _as_bool(zero_rows[0]["strict_static_ward_passed"])
        and max_static <= sigma_strict
    )
    positive_strict = bool(positive_values and max_positive <= sigma_strict)
    positive_soft = bool(positive_values and max_positive <= sigma_soft)
    observable_all = bool(
        physical_all
        and max_reflection <= observable
        and max_logdet <= observable
    )
    return StepMetrics(
        physical_all=physical_all,
        observable_all=observable_all,
        static_strict_all=static_strict,
        positive_strict_all=positive_strict,
        positive_soft_all=positive_soft,
        strict_all=bool(observable_all and static_strict and positive_strict),
        soft_all=bool(observable_all and static_strict and positive_soft),
        max_primary_relative=max_primary,
        max_static_relative=max_static,
        max_positive_relative=max_positive,
        max_reflection_relative=max_reflection,
        max_logdet_relative=max_logdet,
        rows=rows,
    )


def _final_rows(
    *,
    pairing: str,
    case: CaseSpec,
    reference_order: int,
    final_order: int,
    classification: str,
    metrics: StepMetrics,
    cut_audit: StepMetrics | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in metrics.rows:
        index = int(row["matsubara_index"])
        primary_relative = float(row["reference_primary_response_relative"])
        result.append(
            {
                "pairing": pairing,
                "case": case.label,
                "mx": case.mx,
                "my": case.my,
                "q_norm": float(row["q_norm"]),
                "matsubara_index": index,
                "response_sector": str(row["response_sector"]),
                "matsubara_prime_weight": float(row["matsubara_prime_weight"]),
                "xi_eV": float(row["xi_eV"]),
                "reference_gauss_order": reference_order,
                "final_gauss_order": final_order,
                "panel_count": int(row["panel_count"]),
                "classification": classification,
                "point_pipeline_passed": _as_bool(row["point_pipeline_passed"]),
                "ward_passed": _as_bool(row["ward_passed"]),
                "strict_static_ward_passed": _as_bool(row["strict_static_ward_passed"]),
                "primary_response_relative": primary_relative,
                "static_response_relative": primary_relative if index == 0 else float("nan"),
                "sigma_relative": primary_relative if index > 0 else float("nan"),
                "reflection_relative": float(row["reference_reflection_matrix_relative"]),
                "logdet_relative": float(row["reference_logdet_relative"]),
                "material_workspace_implementation": row["material_workspace_implementation"],
                "q_workspace_implementation": row["q_workspace_implementation"],
                "execution_strategy": row["execution_strategy"],
                "cut_audit_performed": cut_audit is not None,
                "cut_audit_soft_passed": bool(cut_audit is not None and cut_audit.soft_all),
                "cut_primary_relative_max": (
                    float("nan") if cut_audit is None else cut_audit.max_primary_relative
                ),
                "cut_reflection_relative_max": (
                    float("nan") if cut_audit is None else cut_audit.max_reflection_relative
                ),
                "cut_logdet_relative_max": (
                    float("nan") if cut_audit is None else cut_audit.max_logdet_relative
                ),
            }
        )
    return result


def main() -> None:
    args = _parse_args()
    preflight_payload: dict[str, Any] | None = None
    if args.require_preflight and not args.dry_run:
        preflight_payload = _validate_preflight(args)

    print("single-method total Matsubara orbit staged scan", flush=True)
    print(f"pairings={args.pairings}; cases={[case.label for case in args.cases]}", flush=True)
    print(
        f"Gauss stage pairs={args.gauss_stages}; Matsubara n={args.matsubara_indices}",
        flush=True,
    )
    print(
        "n=0 uses exact static divided differences; all frequencies share eigensystems",
        flush=True,
    )
    if preflight_payload is not None:
        print(
            f"preflight accepted for git head {preflight_payload.get('git_head')}",
            flush=True,
        )

    if args.dry_run:
        for pairing in args.pairings:
            for case in args.cases:
                print(f"would scan {pairing}/{case.label} through {args.gauss_stages}")
        return

    final_rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for pairing in args.pairings:
        for case in args.cases:
            case_root = args.output_root / "raw" / pairing / case.label
            previous_stage_positive = float("inf")
            soft_streak = 0
            classification = "unresolved"
            final_reference_order = int(args.gauss_stages[-1][0])
            final_order = int(args.gauss_stages[-1][1])
            final_metrics: StepMetrics | None = None

            for stage_index, (low_order, high_order) in enumerate(args.gauss_stages):
                low_output = case_root / f"c{low_order}.csv"
                high_output = case_root / f"c{high_order}.csv"
                _run_command(
                    args=args,
                    pairing=pairing,
                    case=case,
                    order=low_order,
                    output=low_output,
                    reference_csv=None,
                    stage=f"stage_{stage_index}_low",
                )
                _run_command(
                    args=args,
                    pairing=pairing,
                    case=case,
                    order=high_order,
                    output=high_output,
                    reference_csv=low_output,
                    stage=f"stage_{stage_index}_high",
                )
                metrics = _metrics(
                    high_output,
                    sigma_strict=args.strict_sigma_rtol,
                    sigma_soft=args.soft_sigma_rtol,
                    observable=args.observable_rtol,
                )
                final_metrics = metrics
                final_reference_order = low_order
                final_order = high_order

                if metrics.strict_all:
                    classification = "strict"
                    break

                nonworsening = bool(
                    metrics.max_positive_relative
                    <= previous_stage_positive * (1.0 + 1e-12)
                )
                if metrics.soft_all and nonworsening:
                    soft_streak += 1
                elif metrics.soft_all:
                    soft_streak = 1
                else:
                    soft_streak = 0
                previous_stage_positive = metrics.max_positive_relative
                if metrics.soft_all and soft_streak >= args.soft_confirmations:
                    classification = "soft_confirmed"
                    break

            if final_metrics is None:
                raise RuntimeError(f"no comparable order pair for {pairing}/{case.label}")
            if classification == "unresolved":
                if final_metrics.soft_all:
                    classification = "soft_at_max_order"
                elif not final_metrics.physical_all:
                    classification = "physical_failure"
                elif not final_metrics.static_strict_all:
                    classification = "static_unresolved"
                elif not final_metrics.observable_all:
                    classification = "observable_failure"
                else:
                    classification = "positive_response_unresolved"

            cut_metrics: StepMetrics | None = None
            if args.soft_cut_audit and classification.startswith("soft"):
                final_csv = case_root / f"c{final_order}.csv"
                audit_output = case_root / f"c{final_order}_shifted_cut.csv"
                _run_command(
                    args=args,
                    pairing=pairing,
                    case=case,
                    order=final_order,
                    output=audit_output,
                    reference_csv=final_csv,
                    integration_start=-np.pi + args.cut_audit_shift,
                    stage="soft_cut_audit",
                )
                cut_metrics = _metrics(
                    audit_output,
                    sigma_strict=args.strict_sigma_rtol,
                    sigma_soft=args.soft_sigma_rtol,
                    observable=args.observable_rtol,
                )
                if not cut_metrics.soft_all:
                    classification = "cut_audit_failure"

            final_rows.extend(
                _final_rows(
                    pairing=pairing,
                    case=case,
                    reference_order=final_reference_order,
                    final_order=final_order,
                    classification=classification,
                    metrics=final_metrics,
                    cut_audit=cut_metrics,
                )
            )
            case_records.append(
                {
                    "pairing": pairing,
                    "case": case.label,
                    "mx": case.mx,
                    "my": case.my,
                    "classification": classification,
                    "reference_gauss_order": final_reference_order,
                    "final_gauss_order": final_order,
                    "max_primary_relative": final_metrics.max_primary_relative,
                    "max_static_relative": final_metrics.max_static_relative,
                    "max_positive_relative": final_metrics.max_positive_relative,
                    "max_reflection_relative": final_metrics.max_reflection_relative,
                    "max_logdet_relative": final_metrics.max_logdet_relative,
                    "physical_all": final_metrics.physical_all,
                    "observable_all": final_metrics.observable_all,
                    "static_strict_all": final_metrics.static_strict_all,
                    "positive_strict_all": final_metrics.positive_strict_all,
                    "positive_soft_all": final_metrics.positive_soft_all,
                    "strict_all": final_metrics.strict_all,
                    "soft_all": final_metrics.soft_all,
                    "cut_audit_performed": cut_metrics is not None,
                    "cut_audit_soft_passed": bool(cut_metrics is not None and cut_metrics.soft_all),
                }
            )
            print(
                f"[final] {pairing}/{case.label}: {classification}; "
                f"C{final_reference_order}/C{final_order}; "
                f"static={final_metrics.max_static_relative:.3e}; "
                f"positive={final_metrics.max_positive_relative:.3e}; "
                f"R={final_metrics.max_reflection_relative:.3e}; "
                f"logdet={final_metrics.max_logdet_relative:.3e}",
                flush=True,
            )
            if args.stop_on_error and classification in {
                "physical_failure",
                "static_unresolved",
                "observable_failure",
                "cut_audit_failure",
            }:
                raise SystemExit(f"stopping on {pairing}/{case.label}: {classification}")

    if not final_rows or not case_records:
        raise RuntimeError("staged scan produced no final rows")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    points_csv = output_root / "scan_points.csv"
    with points_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    cases_csv = output_root / "scan_cases.csv"
    with cases_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_records[0]))
        writer.writeheader()
        writer.writerows(case_records)

    accepted_classes = {"strict", "soft_confirmed", "soft_at_max_order"}
    total_points = len(final_rows)
    strict_points = sum(
        row["classification"] == "strict"
        and float(row["primary_response_relative"]) <= args.strict_sigma_rtol
        for row in final_rows
    )
    accepted_points = sum(
        row["classification"] in accepted_classes
        and float(row["primary_response_relative"])
        <= (
            args.strict_sigma_rtol
            if int(row["matsubara_index"]) == 0
            else args.soft_sigma_rtol
        )
        and float(row["reflection_relative"]) <= args.observable_rtol
        and float(row["logdet_relative"]) <= args.observable_rtol
        for row in final_rows
    )
    zero_rows = [row for row in final_rows if int(row["matsubara_index"]) == 0]
    strict_fraction = strict_points / total_points
    accepted_fraction = accepted_points / total_points
    all_closure = all(bool(row["point_pipeline_passed"]) for row in final_rows)
    all_observables = all(
        float(row["reflection_relative"]) <= args.observable_rtol
        and float(row["logdet_relative"]) <= args.observable_rtol
        for row in final_rows
    )
    all_static_strict = bool(zero_rows) and all(
        bool(row["strict_static_ward_passed"])
        and float(row["primary_response_relative"]) <= args.strict_sigma_rtol
        for row in zero_rows
    )
    all_cases_accepted = all(
        record["classification"] in accepted_classes for record in case_records
    )
    outer_candidate = bool(
        preflight_payload is not None
        and all_closure
        and all_observables
        and all_static_strict
        and all_cases_accepted
        and accepted_fraction == 1.0
        and strict_fraction >= args.minimum_strict_fraction
    )
    total_wall = float(time.perf_counter() - total_started)
    payload = {
        "schema": "total_matsubara_pointwise_gauss_scan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "preflight": {
            "manifest": str(args.preflight_manifest),
            "required": bool(args.require_preflight),
            "accepted": preflight_payload is not None,
            "git_head": None if preflight_payload is None else preflight_payload.get("git_head"),
        },
        "total_wall_seconds": total_wall,
        "cases": case_records,
        "status": {
            "single_transverse_method": "full_period_equal_panel_composite_gauss_legendre",
            "gauss_stage_pairs": [list(stage) for stage in args.gauss_stages],
            "zero_matsubara_included": bool(zero_rows),
            "zero_uses_exact_static_divided_difference": True,
            "zero_conductivity_division_used": False,
            "zero_and_positive_share_eigensystems": True,
            "all_closure_checks_passed": all_closure,
            "all_observable_checks_passed": all_observables,
            "all_static_points_strict": all_static_strict,
            "strict_point_fraction": strict_fraction,
            "strict_or_soft_point_fraction": accepted_fraction,
            "minimum_strict_fraction": args.minimum_strict_fraction,
            "all_cases_accepted": all_cases_accepted,
            "preflight_passed": preflight_payload is not None,
            "outer_integral_candidate": outer_candidate,
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    summary_json = output_root / "scan_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"scan complete: closure={all_closure}, observables={all_observables}, "
        f"static_strict={all_static_strict}, strict_fraction={strict_fraction:.3f}, "
        f"accepted_fraction={accepted_fraction:.3f}, "
        f"outer_integral_candidate={outer_candidate}",
        flush=True,
    )
    print(f"points CSV: {points_csv}")
    print(f"cases CSV:  {cases_csv}")
    print(f"summary:    {summary_json}")


if __name__ == "__main__":
    main()
