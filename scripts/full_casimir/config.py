from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "casimir" / "runs"
DEFAULT_POSTPROCESS_ROOT = REPO_ROOT / "outputs" / "casimir" / "postprocessed"
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs" / "casimir" / "workflow_logs"

DEFAULT_N_CANDIDATES = (128, 192, 256, 384, 512, 640, 768, 896)
DEFAULT_MATSUBARA_CUTOFFS = (1, 3, 7, 11, 15, 23, 31)
DEFAULT_OUTER_CUTOFFS_U = (6.0, 10.0, 14.0, 18.0, 24.0, 30.0, 36.0, 42.0)
DEFAULT_PAIRINGS = ("spm", "dwave")
DEFAULT_TEMPERATURE_K = 10.0
DEFAULT_SEPARATION_NM = 20.0
DEFAULT_RTOL = 5e-3
DEFAULT_ATOL_J_M2 = 1e-12
DEFAULT_LOGDET_RTOL = 1.5e-3
DEFAULT_LOGDET_ATOL = 1e-6
DEFAULT_CERTIFIER_Q_BATCH_SIZE = 512
DEFAULT_MEMORY_BUDGET_GB = 16.0
DEFAULT_MAX_CONTEXT_WORKERS = 1
DEFAULT_RESERVED_LOGICAL_CPUS = 6
DEFAULT_WORKER_CAP = 26
DEFAULT_SCAN_MIN_DEG = -4
DEFAULT_SCAN_MAX_DEG = 94
DEFAULT_SCAN_STEP_DEG = 2
DEFAULT_TARGET_MIN_DEG = 0
DEFAULT_TARGET_MAX_DEG = 90
PROFILE_NAME = "runtime_budget_v3"
PILOT_PROFILE = "0deg_pilot_v3"
LEGACY_PILOT_PROFILE = "0deg_pilot_v2"


@dataclass(frozen=True)
class RuntimeResources:
    visible_cpus: tuple[int, ...]
    selected_cpus: tuple[int, ...]
    reserved_cpus: tuple[int, ...]

    @property
    def workers(self) -> int:
        return len(self.selected_cpus)

    @property
    def cpuset(self) -> str:
        return ",".join(str(value) for value in self.selected_cpus)

    @property
    def reserved_cpuset(self) -> str:
        return ",".join(str(value) for value in self.reserved_cpus) or "none"


def inclusive_integer_grid(start: int, stop: int, step: int) -> tuple[int, ...]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    values = tuple(range(int(start), int(stop) + 1, int(step)))
    if not values or values[-1] != int(stop):
        raise ValueError("angle interval must be exactly divisible by step")
    return values


def angle_token(angle_deg: int | float) -> str:
    rounded = int(round(float(angle_deg)))
    if abs(float(angle_deg) - rounded) > 1e-9:
        raise ValueError("case naming currently requires integer-degree angles")
    return f"m{abs(rounded):03d}" if rounded < 0 else f"p{rounded:03d}"


def case_name(pairing: str, angle_deg: int | float, *, profile: str = PROFILE_NAME) -> str:
    if pairing not in DEFAULT_PAIRINGS:
        raise ValueError(f"unsupported pairing: {pairing}")
    return f"{pairing}_T10K_d20nm_theta_{angle_token(angle_deg)}deg_{profile}"


def _read_topology(cpu: int) -> tuple[int, int]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        return (
            int((root / "physical_package_id").read_text()),
            int((root / "core_id").read_text()),
        )
    except (OSError, ValueError):
        return 0, int(cpu)


def select_runtime_resources(
    *,
    available_cpus: Sequence[int] | None = None,
    reserve_logical_cpus: int = DEFAULT_RESERVED_LOGICAL_CPUS,
    worker_cap: int = DEFAULT_WORKER_CAP,
) -> RuntimeResources:
    if reserve_logical_cpus < 0:
        raise ValueError("reserve_logical_cpus must be non-negative")
    if worker_cap <= 0:
        raise ValueError("worker_cap must be positive")
    visible = tuple(sorted(
        os.sched_getaffinity(0)
        if available_cpus is None
        else {int(value) for value in available_cpus}
    ))
    if not visible:
        raise RuntimeError("no CPUs are visible to the process")
    target = min(worker_cap, max(1, len(visible) - reserve_logical_cpus))
    groups: dict[tuple[int, int], list[int]] = {}
    for cpu in visible:
        groups.setdefault(_read_topology(cpu), []).append(cpu)
    selected = set(visible)
    reserved: list[int] = []
    for key in reversed(sorted(groups)):
        if len(selected) <= target:
            break
        group = [cpu for cpu in sorted(groups[key]) if cpu in selected]
        if not group or len(selected) - len(group) < target:
            continue
        for cpu in group:
            selected.remove(cpu)
            reserved.append(cpu)
    if len(selected) > target:
        extra = sorted(selected)[target:]
        for cpu in extra:
            selected.remove(cpu)
            reserved.append(cpu)
    selected_tuple = tuple(sorted(selected))
    if not selected_tuple:
        raise RuntimeError("CPU reservation left no workers")
    return RuntimeResources(
        visible_cpus=visible,
        selected_cpus=selected_tuple,
        reserved_cpus=tuple(sorted(set(reserved))),
    )


def apply_single_thread_environment() -> None:
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "MALLOC_ARENA_MAX": "4",
        "PYTHONUNBUFFERED": "1",
    })


def apply_cpu_affinity(resources: RuntimeResources) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(resources.selected_cpus))


def validate_pairings(values: Iterable[str]) -> tuple[str, ...]:
    pairings = tuple(str(value) for value in values)
    invalid = [value for value in pairings if value not in DEFAULT_PAIRINGS]
    if invalid:
        raise ValueError(f"unsupported pairings: {invalid}")
    if not pairings:
        raise ValueError("at least one pairing is required")
    return pairings
