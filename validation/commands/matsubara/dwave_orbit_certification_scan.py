"""Recoverable certification scan for positive-Matsubara d-wave q-orbit response.

This command is an orchestration layer only. It runs the existing periodic-orbit
adaptive and fixed-Gauss CLIs, stores every q-case/stage independently, resumes
completed tasks by an exact parameter fingerprint, and produces one fail-closed
scan summary. No primitive response, Schur, Ward, sheet, reflection, or logdet
physics is reimplemented here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_OUTPUT_ROOT = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_certification_scan"
)
DEFAULT_CASE_TEXT = (
    "axis_min:1:0:full",
    "diagonal_min:1:1:full",
    "generic_small:2:1:full",
    "generic_mid:3:2:screen",
    "reference:6:4:screen",
    "axis_mid:6:0:screen",
    "diagonal_mid:6:6:screen",
    "generic_large:9:6:screen",
)
_LEVELS = {"screen", "tight", "full"}
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CaseSpec:
    label: str
    mx: int
    my: int
    level: str = "screen"

    @property
    def mandatory_tight(self) -> bool:
        return self.level in {"tight", "full"}

    @property
    def mandatory_gauss(self) -> bool:
        return self.level == "full"


@dataclass(frozen=True)
class TaskSpec:
    case: CaseSpec
    stage: str
    output_csv: Path
    command: tuple[str, ...]
    fingerprint: str

    @property
    def output_json(self) -> Path:
        return self.output_csv.with_suffix(".json")

    @property
    def output_summary(self) -> Path:
        return self.output_csv.with_suffix(".summary.txt")

    @property
    def log_path(self) -> Path:
        return self.output_csv.with_suffix(".log")

    @property
    def manifest_path(self) -> Path:
        return self.output_csv.with_suffix(".task.json")


@dataclass(frozen=True)
class TaskResult:
    task: TaskSpec
    state: str
    returncode: int | None
    message: str
    wall_seconds: float

    @property
    def usable(self) -> bool:
        return self.state in {"completed", "skipped"}


@dataclass(frozen=True)
class CaseMetrics:
    label: str
    task_usable: bool
    adaptive_success: bool
    all_point_pipelines_passed: bool
    scaled_error_estimate: float
    max_ward_ratio: float
    max_condition: float


def _parse_case(text: str) -> CaseSpec:
    parts = str(text).split(":")
    if len(parts) not in {3, 4}:
        raise ValueError("case must have form LABEL:MX:MY[:screen|tight|full]")
    label = parts[0].strip()
    if not _LABEL_RE.fullmatch(label):
        raise ValueError(
            f"invalid case label {label!r}; use letters, numbers, '.', '_' or '-'"
        )
    try:
        mx, my = int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(f"case {label!r} has non-integer mx/my") from exc
    if mx == 0 and my == 0:
        raise ValueError(f"case {label!r} cannot use q=(0,0)")
    level = parts[3].strip().lower() if len(parts) == 4 else "screen"
    if level not in _LEVELS:
        raise ValueError(f"case {label!r} has unknown level {level!r}")
    return CaseSpec(label=label, mx=mx, my=my, level=level)


def _validate_cases(cases: Sequence[CaseSpec]) -> tuple[CaseSpec, ...]:
    if not cases:
        raise ValueError("at least one q case is required")
    labels: set[str] = set()
    coordinates: set[tuple[int, int]] = set()
    result: list[CaseSpec] = []
    for case in cases:
        if case.label in labels:
            raise ValueError(f"duplicate case label: {case.label}")
        coordinate = (case.mx, case.my)
        if coordinate in coordinates:
            raise ValueError(f"duplicate q coordinate: {coordinate}")
        labels.add(case.label)
        coordinates.add(coordinate)
        result.append(case)
    return tuple(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        metavar="LABEL:MX:MY[:LEVEL]",
        help=(
            "repeatable q case; LEVEL is screen, tight, or full. "
            "Supplying any --case replaces the built-in plan"
        ),
    )
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--pilot-order", type=int, default=16)
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--scale-floor-relative", type=float, default=1e-8)
    parser.add_argument("--scale-floor-absolute", type=float, default=1e-14)

    parser.add_argument("--screen-epsabs", type=float, default=2e-5)
    parser.add_argument("--screen-epsrel", type=float, default=2e-3)
    parser.add_argument(
        "--screen-max-point-evaluations",
        type=int,
        default=0,
        help="0 chooses the exact budget for periodic T256",
    )
    parser.add_argument("--tight-epsabs", type=float, default=5e-6)
    parser.add_argument("--tight-epsrel", type=float, default=5e-4)
    parser.add_argument(
        "--tight-max-point-evaluations",
        type=int,
        default=0,
        help="0 chooses the exact budget for periodic T512",
    )
    parser.add_argument("--gauss-order", type=int, default=224)
    parser.add_argument(
        "--gauss-max-point-evaluations",
        type=int,
        default=0,
        help="0 chooses the exact budget for the requested Gauss order",
    )
    parser.add_argument("--t512-top-k", type=int, default=2)
    parser.add_argument("--gauss-top-k", type=int, default=1)

    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--reference-matrix-rtol", type=float, default=1e-3)
    parser.add_argument("--reference-logdet-rtol", type=float, default=1e-3)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed tasks only when their exact parameter fingerprint matches",
    )
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--no-gauss", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_cases = args.case if args.case is not None else list(DEFAULT_CASE_TEXT)
    try:
        args.cases = _validate_cases(tuple(_parse_case(value) for value in raw_cases))
    except ValueError as exc:
        parser.error(str(exc))
    if args.nk <= 0 or args.pilot_order <= 0:
        parser.error("--nk and --pilot-order must be positive")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if args.gauss_order <= 0:
        parser.error("--gauss-order must be positive")
    for name in (
        "screen_max_point_evaluations",
        "tight_max_point_evaluations",
        "gauss_max_point_evaluations",
    ):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.t512_top_k < 0 or args.gauss_top_k < 0:
        parser.error("--t512-top-k and --gauss-top-k must be non-negative")
    for name in (
        "screen_epsabs",
        "screen_epsrel",
        "tight_epsabs",
        "tight_epsrel",
        "reference_matrix_rtol",
        "reference_logdet_rtol",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_payload(
    task: TaskSpec,
    *,
    state: str,
    returncode: int | None,
    message: str,
    wall: float,
) -> dict[str, Any]:
    return {
        "schema": "dwave_positive_orbit_certification_task_v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "case": asdict(task.case),
        "stage": task.stage,
        "fingerprint": task.fingerprint,
        "state": state,
        "returncode": returncode,
        "message": message,
        "wall_seconds": float(wall),
        "command": list(task.command),
        "output_csv": str(task.output_csv),
        "output_json": str(task.output_json),
        "log": str(task.log_path),
    }


def _resume_match(task: TaskSpec) -> bool:
    if (
        not task.manifest_path.is_file()
        or not task.output_csv.is_file()
        or not task.output_json.is_file()
    ):
        return False
    try:
        payload = json.loads(task.manifest_path.read_text(encoding="utf-8"))
        child = json.loads(task.output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("schema") == "dwave_positive_orbit_certification_task_v1"
        and payload.get("state") == "completed"
        and payload.get("fingerprint") == task.fingerprint
        and isinstance(child, dict)
        and child.get("schema")
    )


def _run_task(task: TaskSpec, *, resume: bool, dry_run: bool) -> TaskResult:
    if resume and _resume_match(task):
        print(
            f"[skip] {task.case.label} {task.stage}: exact completed fingerprint",
            flush=True,
        )
        return TaskResult(task, "skipped", 0, "exact completed fingerprint", 0.0)
    print(f"[run]  {task.case.label} {task.stage}", flush=True)
    print("       " + " ".join(task.command), flush=True)
    if dry_run:
        return TaskResult(task, "planned", None, "dry run", 0.0)

    task.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        task.manifest_path,
        _task_payload(
            task,
            state="running",
            returncode=None,
            message="",
            wall=0.0,
        ),
    )
    started = time.perf_counter()
    returncode: int | None = None
    message = ""
    try:
        with task.log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(task.command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = int(process.wait())
    except OSError as exc:
        returncode = -1
        message = f"failed to start child process: {exc}"
    wall = float(time.perf_counter() - started)
    if returncode == 0 and task.output_csv.is_file() and task.output_json.is_file():
        state = "completed"
        message = "completed"
    else:
        state = "failed"
        if not message:
            missing = [
                str(path)
                for path in (task.output_csv, task.output_json)
                if not path.is_file()
            ]
            detail = f"; missing outputs={missing}" if missing else ""
            message = f"child return code {returncode}{detail}"
    _atomic_write_json(
        task.manifest_path,
        _task_payload(
            task,
            state=state,
            returncode=returncode,
            message=message,
            wall=wall,
        ),
    )
    print(f"[{state}] {task.case.label} {task.stage}: {message}", flush=True)
    return TaskResult(task, state, returncode, message, wall)


def _orbit_origin_count(case: CaseSpec, subgrid_average: str) -> int:
    common = math.gcd(abs(int(case.mx)), abs(int(case.my)))
    return 2 if subgrid_average == "auto" and common % 2 == 1 else 1


def _resolved_budget(
    configured: int,
    *,
    nk: int,
    case: CaseSpec,
    subgrid_average: str,
    transverse_order: int,
) -> int:
    if int(configured) > 0:
        return int(configured)
    return int(nk) * _orbit_origin_count(case, subgrid_average) * int(
        transverse_order
    )


def _common_child_args(args: argparse.Namespace, case: CaseSpec) -> list[str]:
    return [
        "--nk",
        str(args.nk),
        "--mx",
        str(case.mx),
        "--my",
        str(case.my),
        "--matsubara-indices",
        *(str(value) for value in sorted(set(args.matsubara_indices))),
        "--shift-s",
        str(args.shift_s),
        "--subgrid-average",
        str(args.subgrid_average),
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
    ]


def _make_periodic_task(
    args: argparse.Namespace,
    case: CaseSpec,
    *,
    tight: bool,
) -> TaskSpec:
    stage = "periodic_tight" if tight else "periodic_screen"
    filename = "periodic_tight.csv" if tight else "periodic_screen.csv"
    output = args.output_root / "raw" / case.label / filename
    epsabs = args.tight_epsabs if tight else args.screen_epsabs
    epsrel = args.tight_epsrel if tight else args.screen_epsrel
    configured_maximum = (
        args.tight_max_point_evaluations
        if tight
        else args.screen_max_point_evaluations
    )
    maximum = _resolved_budget(
        configured_maximum,
        nk=args.nk,
        case=case,
        subgrid_average=args.subgrid_average,
        transverse_order=512 if tight else 256,
    )
    command = (
        sys.executable,
        "-m",
        "validation",
        "matsubara",
        "dwave-orbit-adaptive",
        *_common_child_args(args, case),
        "--pilot-order",
        str(args.pilot_order),
        "--epsabs",
        str(epsabs),
        "--epsrel",
        str(epsrel),
        "--norm",
        str(args.norm),
        "--scale-floor-relative",
        str(args.scale_floor_relative),
        "--scale-floor-absolute",
        str(args.scale_floor_absolute),
        "--max-point-evaluations",
        str(maximum),
        "--output",
        str(output),
    )
    fingerprint = _fingerprint(
        {
            "schema": "periodic_task_v1",
            "case": asdict(case),
            "stage": stage,
            "command": list(command[1:]),
        }
    )
    return TaskSpec(case, stage, output, command, fingerprint)


def _make_gauss_task(
    args: argparse.Namespace,
    case: CaseSpec,
    reference_csv: Path,
) -> TaskSpec:
    stage = "fixed_gauss_crosscheck"
    output = (
        args.output_root
        / "raw"
        / case.label
        / f"gauss_g{int(args.gauss_order)}_vs_tight.csv"
    )
    maximum = _resolved_budget(
        args.gauss_max_point_evaluations,
        nk=args.nk,
        case=case,
        subgrid_average=args.subgrid_average,
        transverse_order=args.gauss_order,
    )
    command = (
        sys.executable,
        "-m",
        "validation",
        "matsubara",
        "dwave-orbit-gauss-crosscheck",
        *_common_child_args(args, case),
        "--gauss-orders",
        str(args.gauss_order),
        "--max-point-evaluations",
        str(maximum),
        "--reference-matrix-rtol",
        str(args.reference_matrix_rtol),
        "--reference-logdet-rtol",
        str(args.reference_logdet_rtol),
        "--reference-csv",
        str(reference_csv),
        "--output",
        str(output),
    )
    fingerprint = _fingerprint(
        {
            "schema": "gauss_task_v1",
            "case": asdict(case),
            "stage": stage,
            "command": list(command[1:]),
        }
    )
    return TaskSpec(case, stage, output, command, fingerprint)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _metrics(task_result: TaskResult) -> CaseMetrics:
    label = task_result.task.case.label
    if not task_result.usable:
        return CaseMetrics(
            label,
            False,
            False,
            False,
            float("inf"),
            float("inf"),
            float("inf"),
        )
    try:
        payload = json.loads(task_result.task.output_json.read_text(encoding="utf-8"))
        rows = _read_csv(task_result.task.output_csv)
        quadrature = payload["quadrature"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return CaseMetrics(
            label,
            False,
            False,
            False,
            float("inf"),
            float("inf"),
            float("inf"),
        )
    return CaseMetrics(
        label=label,
        task_usable=True,
        adaptive_success=bool(quadrature.get("adaptive_success", False)),
        all_point_pipelines_passed=bool(rows)
        and all(_as_bool(row.get("point_pipeline_passed")) for row in rows),
        scaled_error_estimate=_safe_float(
            quadrature.get("scaled_error_estimate"),
            float("inf"),
        ),
        max_ward_ratio=max(
            (
                _safe_float(
                    row.get("ward_effective_mixed_ratio_max"),
                    float("inf"),
                )
                for row in rows
            ),
            default=float("inf"),
        ),
        max_condition=max(
            (
                _safe_float(
                    row.get("schur_condition_number"),
                    float("inf"),
                )
                for row in rows
            ),
            default=float("inf"),
        ),
    )


def _difficulty_key(metrics: CaseMetrics) -> tuple[int, int, float, float, float]:
    def finite_or_inf(value: float) -> float:
        return value if math.isfinite(value) else float("inf")

    return (
        int(not metrics.task_usable or not metrics.all_point_pipelines_passed),
        int(not metrics.adaptive_success),
        finite_or_inf(metrics.scaled_error_estimate),
        finite_or_inf(metrics.max_ward_ratio),
        finite_or_inf(metrics.max_condition),
    )


def _select_labels(
    cases: Sequence[CaseSpec],
    metrics_by_label: dict[str, CaseMetrics],
    *,
    mandatory_attribute: str,
    top_k: int,
    include_failures: bool,
) -> tuple[str, ...]:
    selected = {
        case.label for case in cases if bool(getattr(case, mandatory_attribute))
    }
    if include_failures:
        selected.update(
            label
            for label, metrics in metrics_by_label.items()
            if not metrics.task_usable
            or not metrics.adaptive_success
            or not metrics.all_point_pipelines_passed
        )
    ranked = sorted(
        (case.label for case in cases if case.label not in selected),
        key=lambda label: _difficulty_key(metrics_by_label[label]),
        reverse=True,
    )
    selected.update(ranked[: max(0, int(top_k))])
    return tuple(case.label for case in cases if case.label in selected)


def _matrix_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(
                    _safe_float(row[f"{prefix}_xx_real"]),
                    _safe_float(row[f"{prefix}_xx_imag"]),
                ),
                complex(
                    _safe_float(row[f"{prefix}_xy_real"]),
                    _safe_float(row[f"{prefix}_xy_imag"]),
                ),
            ],
            [
                complex(
                    _safe_float(row[f"{prefix}_yx_real"]),
                    _safe_float(row[f"{prefix}_yx_imag"]),
                ),
                complex(
                    _safe_float(row[f"{prefix}_yy_real"]),
                    _safe_float(row[f"{prefix}_yy_imag"]),
                ),
            ],
        ],
        dtype=complex,
    )


def _matrix_relative(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    return float(np.linalg.norm(a - b) / denominator)


def _scalar_relative(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(
        abs(float(a)),
        abs(float(b)),
        1e-30,
    )


def _load_by_index(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["matsubara_index"]): row for row in _read_csv(path)}


def _periodic_summary_rows(
    case: CaseSpec,
    task_result: TaskResult,
    *,
    screen_rows: dict[int, dict[str, str]] | None,
) -> list[dict[str, Any]]:
    if not task_result.usable:
        return []
    payload = json.loads(task_result.task.output_json.read_text(encoding="utf-8"))
    quadrature = payload.get("quadrature", {})
    rows = _load_by_index(task_result.task.output_csv)
    output: list[dict[str, Any]] = []
    for index, row in sorted(rows.items()):
        comparison_sigma = float("nan")
        comparison_reflection = float("nan")
        comparison_logdet = float("nan")
        if screen_rows is not None and index in screen_rows:
            baseline = screen_rows[index]
            comparison_sigma = _matrix_relative(
                _matrix_from_row(row, "sigma_tilde"),
                _matrix_from_row(baseline, "sigma_tilde"),
            )
            comparison_reflection = _matrix_relative(
                _matrix_from_row(row, "reflection"),
                _matrix_from_row(baseline, "reflection"),
            )
            comparison_logdet = _scalar_relative(
                _safe_float(row["logdet"]),
                _safe_float(baseline["logdet"]),
            )
        output.append(
            {
                "case": case.label,
                "mx": case.mx,
                "my": case.my,
                "q_norm": _safe_float(row.get("q_norm")),
                "q_angle_rad": math.atan2(case.my, case.mx),
                "stage": task_result.task.stage,
                "method": "periodic_nested",
                "matsubara_index": index,
                "xi_eV": _safe_float(row.get("xi_eV")),
                "transverse_evaluations": int(
                    _safe_float(quadrature.get("transverse_evaluations"), 0.0)
                ),
                "point_evaluations": int(
                    _safe_float(quadrature.get("point_evaluations"), 0.0)
                ),
                "quadrature_wall_seconds": _safe_float(
                    quadrature.get("quadrature_wall_seconds")
                ),
                "adaptive_success": bool(
                    quadrature.get("adaptive_success", False)
                ),
                "adaptive_status": int(
                    _safe_float(quadrature.get("adaptive_status"), -1.0)
                ),
                "scaled_error_estimate": _safe_float(
                    quadrature.get("scaled_error_estimate")
                ),
                "ward_ratio": _safe_float(
                    row.get("ward_effective_mixed_ratio_max")
                ),
                "condition": _safe_float(row.get("schur_condition_number")),
                "point_pipeline_passed": _as_bool(
                    row.get("point_pipeline_passed")
                ),
                "crosscheck_passed": "",
                "screen_to_tight_sigma_relative": comparison_sigma,
                "screen_to_tight_reflection_relative": comparison_reflection,
                "screen_to_tight_logdet_relative": comparison_logdet,
                "gauss_to_tight_sigma_relative": float("nan"),
                "gauss_to_tight_reflection_relative": float("nan"),
                "gauss_to_tight_logdet_relative": float("nan"),
                "logdet": _safe_float(row.get("logdet")),
            }
        )
    return output


def _gauss_summary_rows(
    case: CaseSpec,
    task_result: TaskResult,
) -> list[dict[str, Any]]:
    if not task_result.usable:
        return []
    output: list[dict[str, Any]] = []
    for row in _read_csv(task_result.task.output_csv):
        output.append(
            {
                "case": case.label,
                "mx": case.mx,
                "my": case.my,
                "q_norm": _safe_float(row.get("q_norm")),
                "q_angle_rad": math.atan2(case.my, case.mx),
                "stage": task_result.task.stage,
                "method": "fixed_gauss",
                "matsubara_index": int(row["matsubara_index"]),
                "xi_eV": _safe_float(row.get("xi_eV")),
                "transverse_evaluations": int(row["gauss_order"]),
                "point_evaluations": int(row["point_evaluations"]),
                "quadrature_wall_seconds": _safe_float(
                    row.get("quadrature_wall_seconds")
                ),
                "adaptive_success": "",
                "adaptive_status": "",
                "scaled_error_estimate": float("nan"),
                "ward_ratio": _safe_float(
                    row.get("ward_effective_mixed_ratio_max")
                ),
                "condition": _safe_float(row.get("schur_condition_number")),
                "point_pipeline_passed": _as_bool(
                    row.get("point_pipeline_passed")
                ),
                "crosscheck_passed": _as_bool(row.get("crosscheck_passed")),
                "screen_to_tight_sigma_relative": float("nan"),
                "screen_to_tight_reflection_relative": float("nan"),
                "screen_to_tight_logdet_relative": float("nan"),
                "gauss_to_tight_sigma_relative": _safe_float(
                    row.get("reference_sigma_matrix_relative")
                ),
                "gauss_to_tight_reflection_relative": _safe_float(
                    row.get("reference_reflection_matrix_relative")
                ),
                "gauss_to_tight_logdet_relative": _safe_float(
                    row.get("reference_logdet_relative")
                ),
                "logdet": _safe_float(row.get("logdet")),
            }
        )
    return output


def _finite_max(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float))
        and math.isfinite(float(row[field]))
    ]
    return max(values, default=float("nan"))


def _summary_text(
    *,
    cases: Sequence[CaseSpec],
    tight_labels: Sequence[str],
    gauss_labels: Sequence[str],
    task_results: Sequence[TaskResult],
    rows: Sequence[dict[str, Any]],
    case_status: dict[str, bool],
) -> str:
    failed_tasks = [
        f"{result.task.case.label}:{result.task.stage}"
        for result in task_results
        if result.state == "failed"
    ]
    lines = [
        "positive-Matsubara d-wave q-orbit certification scan",
        "=" * 78,
        f"cases = {tuple(case.label for case in cases)}",
        f"tight selected = {tuple(tight_labels)}",
        f"fixed-Gauss selected = {tuple(gauss_labels)}",
        f"failed tasks = {tuple(failed_tasks)}",
        "",
        " case                 q=(mx,my)  level   certified",
        "-" * 62,
    ]
    for case in cases:
        lines.append(
            f" {case.label:20s} ({case.mx:3d},{case.my:3d})  "
            f"{case.level:6s}  {str(bool(case_status.get(case.label, False))):>9s}"
        )
    lines.extend(
        [
            "",
            "worst screen->tight sigma relative = "
            f"{_finite_max(rows, 'screen_to_tight_sigma_relative'):.6e}",
            "worst screen->tight reflection relative = "
            f"{_finite_max(rows, 'screen_to_tight_reflection_relative'):.6e}",
            "worst screen->tight logdet relative = "
            f"{_finite_max(rows, 'screen_to_tight_logdet_relative'):.6e}",
            "worst Gauss->tight sigma relative = "
            f"{_finite_max(rows, 'gauss_to_tight_sigma_relative'):.6e}",
            "worst Gauss->tight reflection relative = "
            f"{_finite_max(rows, 'gauss_to_tight_reflection_relative'):.6e}",
            "worst Gauss->tight logdet relative = "
            f"{_finite_max(rows, 'gauss_to_tight_logdet_relative'):.6e}",
            "",
            "certification_scan_passed = "
            f"{all(case_status.values()) and not failed_tasks}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_scan_outputs(
    args: argparse.Namespace,
    cases: Sequence[CaseSpec],
    tight_labels: Sequence[str],
    gauss_labels: Sequence[str],
    task_results: Sequence[TaskResult],
) -> None:
    task_by_key = {
        (result.task.case.label, result.task.stage): result
        for result in task_results
    }
    rows: list[dict[str, Any]] = []
    case_status: dict[str, bool] = {}
    for case in cases:
        screen = task_by_key.get((case.label, "periodic_screen"))
        tight = task_by_key.get((case.label, "periodic_tight"))
        gauss = task_by_key.get((case.label, "fixed_gauss_crosscheck"))
        screen_rows = (
            _load_by_index(screen.task.output_csv)
            if screen and screen.usable
            else None
        )
        if screen:
            rows.extend(
                _periodic_summary_rows(case, screen, screen_rows=None)
            )
        if tight:
            rows.extend(
                _periodic_summary_rows(case, tight, screen_rows=screen_rows)
            )
        if gauss:
            rows.extend(_gauss_summary_rows(case, gauss))

        authoritative = tight if case.label in tight_labels else screen
        authoritative_ok = bool(authoritative and authoritative.usable)
        if authoritative_ok:
            authoritative_ok = all(
                _as_bool(row.get("point_pipeline_passed"))
                for row in _read_csv(authoritative.task.output_csv)
            )
        gauss_ok = True
        if case.label in gauss_labels:
            gauss_ok = bool(gauss and gauss.usable)
            if gauss_ok:
                gauss_ok = all(
                    _as_bool(row.get("crosscheck_passed"))
                    for row in _read_csv(gauss.task.output_csv)
                )
        case_status[case.label] = bool(authoritative_ok and gauss_ok)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "certification_scan.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = _summary_text(
        cases=cases,
        tight_labels=tight_labels,
        gauss_labels=gauss_labels,
        task_results=task_results,
        rows=rows,
        case_status=case_status,
    )
    summary_path = output_root / "certification_scan.summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_positive_orbit_certification_scan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: (
                [asdict(case) for case in value]
                if key == "cases"
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
            if key != "case"
        },
        "tight_selected": list(tight_labels),
        "gauss_selected": list(gauss_labels),
        "tasks": [
            {
                "case": result.task.case.label,
                "stage": result.task.stage,
                "state": result.state,
                "returncode": result.returncode,
                "message": result.message,
                "wall_seconds": result.wall_seconds,
                "output_csv": str(result.task.output_csv),
                "fingerprint": result.task.fingerprint,
            }
            for result in task_results
        ],
        "case_status": case_status,
        "rows": rows,
        "status": {
            "certification_scan_passed": all(case_status.values())
            and not any(result.state == "failed" for result in task_results),
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    json_path = output_root / "certification_scan.json"
    json_path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(summary)
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


def main() -> None:
    args = _parse_args()
    cases: tuple[CaseSpec, ...] = args.cases
    print("positive d-wave q-orbit certification plan")
    for case in cases:
        print(
            f"  {case.label:20s} q=({case.mx},{case.my}) level={case.level}"
        )
    print(
        f"dynamic additions: tight top-k={args.t512_top_k}; "
        f"Gauss top-k={args.gauss_top_k}",
        flush=True,
    )

    task_results: list[TaskResult] = []
    screen_results: dict[str, TaskResult] = {}
    for case in cases:
        result = _run_task(
            _make_periodic_task(args, case, tight=False),
            resume=args.resume,
            dry_run=args.dry_run,
        )
        task_results.append(result)
        screen_results[case.label] = result
        if result.state == "failed" and args.stop_on_error:
            raise SystemExit(result.message)

    if args.dry_run:
        mandatory_tight = tuple(
            case.label for case in cases if case.mandatory_tight
        )
        mandatory_gauss = tuple(
            case.label for case in cases if case.mandatory_gauss
        )
        print(f"mandatory tight cases = {mandatory_tight}")
        print(f"mandatory Gauss cases = {mandatory_gauss}")
        print("dynamic top-k selections are resolved after screen results")
        return

    metrics_by_label = {
        label: _metrics(result) for label, result in screen_results.items()
    }
    tight_labels: tuple[str, ...] = ()
    gauss_labels: tuple[str, ...] = ()
    tight_results: dict[str, TaskResult] = {}
    if not args.screen_only:
        tight_labels = _select_labels(
            cases,
            metrics_by_label,
            mandatory_attribute="mandatory_tight",
            top_k=args.t512_top_k,
            include_failures=True,
        )
        print(f"selected tight cases: {tight_labels}", flush=True)
        case_by_label = {case.label: case for case in cases}
        for label in tight_labels:
            case = case_by_label[label]
            result = _run_task(
                _make_periodic_task(args, case, tight=True),
                resume=args.resume,
                dry_run=False,
            )
            task_results.append(result)
            tight_results[label] = result
            if result.state == "failed" and args.stop_on_error:
                raise SystemExit(result.message)

        if not args.no_gauss:
            tight_metrics = {
                label: _metrics(result)
                for label, result in tight_results.items()
            }
            eligible_cases = tuple(
                case
                for case in cases
                if case.label in tight_metrics
                and tight_metrics[case.label].task_usable
                and tight_metrics[case.label].all_point_pipelines_passed
            )
            eligible_metrics = {
                case.label: tight_metrics[case.label]
                for case in eligible_cases
            }
            gauss_labels = _select_labels(
                eligible_cases,
                eligible_metrics,
                mandatory_attribute="mandatory_gauss",
                top_k=args.gauss_top_k,
                include_failures=False,
            )
            print(
                f"selected fixed-Gauss cases: {gauss_labels}",
                flush=True,
            )
            for label in gauss_labels:
                case = case_by_label[label]
                reference_csv = tight_results[label].task.output_csv
                result = _run_task(
                    _make_gauss_task(args, case, reference_csv),
                    resume=args.resume,
                    dry_run=False,
                )
                task_results.append(result)
                if result.state == "failed" and args.stop_on_error:
                    raise SystemExit(result.message)

    _write_scan_outputs(
        args,
        cases,
        tight_labels,
        gauss_labels,
        task_results,
    )


if __name__ == "__main__":
    main()
