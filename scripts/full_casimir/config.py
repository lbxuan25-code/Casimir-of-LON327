from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence
import math
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "casimir" / "runs"
DEFAULT_PRODUCTION_ROOT = REPO_ROOT / "outputs" / "casimir" / "production"
DEFAULT_POSTPROCESS_ROOT = REPO_ROOT / "outputs" / "casimir" / "postprocessed"
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs" / "casimir" / "workflow_logs"

# TODO 5 frozen microscopic policy.  The ladder and tolerances are the common
# pairing-blind candidate selected by the completed 0-degree transverse audit.
MICROSCOPIC_POLICY_ID = "full-casimir-microscopic-policy-v1"
DEFAULT_N_CANDIDATES = (
    128,
    192,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
)
DEFAULT_LOGDET_RTOL = 2.0e-3
DEFAULT_LOGDET_ATOL = 1.0e-6
DEFAULT_REQUIRED_CONSECUTIVE_PASSES = 2
DEFAULT_TRANSVERSE_SHIFTS = ((0.5, 0.5), (0.25, 0.75), (0.75, 0.25))

# The outer-Q and Matsubara ladders are qualification candidates.  Their final
# economical endpoints remain conditional on the fresh 0-degree run after TODO 8.
OUTER_QUALIFICATION_CANDIDATE_ID = "full-casimir-outer-candidate-v1"
MATSUBARA_QUALIFICATION_CANDIDATE_ID = "full-casimir-matsubara-candidate-v1"
DEFAULT_MATSUBARA_CUTOFFS = (1, 3, 7, 15, 31, 63)
DEFAULT_OUTER_CUTOFFS_U = (
    6.0,
    10.0,
    14.0,
    18.0,
    24.0,
    30.0,
    36.0,
    42.0,
    48.0,
    54.0,
    60.0,
)
DEFAULT_PAIRINGS = ("spm", "dwave")
DEFAULT_TEMPERATURE_K = 10.0
DEFAULT_SEPARATION_NM = 20.0
DEFAULT_DELTA0_EV = 0.1
DEFAULT_ETA_EV = 1e-8
DEFAULT_DEGENERACY = 1.0
DEFAULT_RTOL = 5e-3
DEFAULT_ATOL_J_M2 = 1e-12

# TODO 5 frozen execution policy.  These values do not enter the scientific
# identity; they define the qualified engineering route used to realize it.
EXECUTION_POLICY_ID = "full-casimir-execution-policy-v1"
DEFAULT_CERTIFIER_Q_BATCH_SIZE = 512
DEFAULT_MEMORY_BUDGET_GB = 16.0
DEFAULT_MAX_CONTEXT_WORKERS = 1
DEFAULT_RESERVED_LOGICAL_CPUS = 6
DEFAULT_WORKER_CAP = 26
DEFAULT_PARALLEL_MODE = "q"
DEFAULT_MEMORY_SAFETY_FACTOR = 1.5
DEFAULT_FALLBACK_CONTEXT_BYTES_PER_POINT = 16_384.0
DEFAULT_CANONICAL_BLOCK = 4096
DEFAULT_RUNTIME_CHUNK = 16_384
DEFAULT_OUTER_TAIL_START_U = 24.0
DEFAULT_OUTER_TAIL_WINDOW_SHELLS = 3
DEFAULT_OUTER_TAIL_RATIO_MAX = 0.8
DEFAULT_MATSUBARA_TAIL_START_N = 4
# Two training blocks plus one transition and one final holdout block.
DEFAULT_MATSUBARA_TAIL_WINDOW_TERMS = 4
DEFAULT_MATSUBARA_TAIL_RATIO_MAX = 0.8
DEFAULT_RADIAL_BUDGET_FRACTION = 0.8
DEFAULT_MAX_TOTAL_MICROSCOPIC_Q_NODES = 250_000
DEFAULT_MAX_TOTAL_MICROSCOPIC_POINT_ENTRIES = 1_000_000
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


@dataclass(frozen=True)
class ExecutionProfile:
    """Deterministic execution-only profile for one hardware class."""

    name: str
    reserve_logical_cpus: int
    worker_cap: int
    parallel_mode: str
    memory_budget_gb: float
    max_context_workers: int
    certifier_q_batch_size: int
    memory_safety_factor: float
    fallback_context_bytes_per_point: float
    canonical_block: int
    runtime_chunk: int

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


LOCAL_EXECUTION_PROFILE = ExecutionProfile(
    name="local-workstation-v1",
    reserve_logical_cpus=DEFAULT_RESERVED_LOGICAL_CPUS,
    worker_cap=DEFAULT_WORKER_CAP,
    parallel_mode=DEFAULT_PARALLEL_MODE,
    memory_budget_gb=DEFAULT_MEMORY_BUDGET_GB,
    max_context_workers=DEFAULT_MAX_CONTEXT_WORKERS,
    certifier_q_batch_size=DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    memory_safety_factor=DEFAULT_MEMORY_SAFETY_FACTOR,
    fallback_context_bytes_per_point=DEFAULT_FALLBACK_CONTEXT_BYTES_PER_POINT,
    canonical_block=DEFAULT_CANONICAL_BLOCK,
    runtime_chunk=DEFAULT_RUNTIME_CHUNK,
)

# The server profile keeps the same formally qualified q-parallel numerical
# execution strategy but resolves the worker count from the server affinity.
SERVER_EXECUTION_PROFILE = ExecutionProfile(
    name="server-throughput-v1",
    reserve_logical_cpus=2,
    worker_cap=0,
    parallel_mode=DEFAULT_PARALLEL_MODE,
    memory_budget_gb=0.0,
    max_context_workers=DEFAULT_MAX_CONTEXT_WORKERS,
    certifier_q_batch_size=DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    memory_safety_factor=DEFAULT_MEMORY_SAFETY_FACTOR,
    fallback_context_bytes_per_point=DEFAULT_FALLBACK_CONTEXT_BYTES_PER_POINT,
    canonical_block=DEFAULT_CANONICAL_BLOCK,
    runtime_chunk=DEFAULT_RUNTIME_CHUNK,
)


def execution_profile(name: str) -> ExecutionProfile:
    profiles = {
        LOCAL_EXECUTION_PROFILE.name: LOCAL_EXECUTION_PROFILE,
        SERVER_EXECUTION_PROFILE.name: SERVER_EXECUTION_PROFILE,
    }
    try:
        return profiles[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown execution profile {name!r}; expected one of {tuple(profiles)}"
        ) from exc


def microscopic_policy_payload() -> dict[str, object]:
    return {
        "policy_id": MICROSCOPIC_POLICY_ID,
        "status": "frozen",
        "pairing_blind": True,
        "N_candidates": list(DEFAULT_N_CANDIDATES),
        "shifts": [list(value) for value in DEFAULT_TRANSVERSE_SHIFTS],
        "logdet_rtol": DEFAULT_LOGDET_RTOL,
        "logdet_atol": DEFAULT_LOGDET_ATOL,
        "required_consecutive_passes": DEFAULT_REQUIRED_CONSECUTIVE_PASSES,
        "evidence": "completed 0deg qualification-v5 transverse audit",
    }


def qualification_candidate_payload() -> dict[str, object]:
    return {
        "outer": {
            "policy_id": OUTER_QUALIFICATION_CANDIDATE_ID,
            "status": "candidate_pending_post_todo8_fresh_0deg",
            "cutoff_u_values": list(DEFAULT_OUTER_CUTOFFS_U),
            "analytic_certificate_attempted_at_each_cutoff": True,
        },
        "matsubara": {
            "policy_id": MATSUBARA_QUALIFICATION_CANDIDATE_ID,
            "status": "candidate_pending_post_todo8_fresh_0deg",
            "cutoff_values": list(DEFAULT_MATSUBARA_CUTOFFS),
            "dyadic_blocks": True,
            "holdout_blocks": 1,
        },
    }


def _decimal(value: int | float, *, label: str) -> Decimal:
    try:
        number = float(value)
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return result


def inclusive_float_grid(
    start: int | float,
    stop: int | float,
    step: int | float,
) -> tuple[float, ...]:
    """Return an exactly divisible inclusive decimal grid without float drift."""

    start_decimal = _decimal(start, label="grid start")
    stop_decimal = _decimal(stop, label="grid stop")
    step_decimal = _decimal(step, label="grid step")
    if step_decimal <= 0:
        raise ValueError("step must be positive")
    if stop_decimal < start_decimal:
        raise ValueError("stop must be greater than or equal to start")
    quotient = (stop_decimal - start_decimal) / step_decimal
    integral = quotient.to_integral_value()
    if quotient != integral:
        raise ValueError("interval must be exactly divisible by step")
    return tuple(
        float(start_decimal + index * step_decimal)
        for index in range(int(integral) + 1)
    )


def inclusive_integer_grid(start: int, stop: int, step: int) -> tuple[int, ...]:
    values = inclusive_float_grid(start, stop, step)
    return tuple(int(value) for value in values)


def _normalized_number_text(value: int | float, *, label: str) -> str:
    number = float(_decimal(value, label=label))
    if abs(number) < 5e-10:
        number = 0.0
    text = f"{number:.9f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _unsigned_token(value: int | float, *, label: str) -> str:
    text = _normalized_number_text(value, label=label)
    if text.startswith("-"):
        raise ValueError(f"{label} must be non-negative")
    return text.replace(".", "p")


def angle_token(angle_deg: int | float) -> str:
    text = _normalized_number_text(angle_deg, label="angle")
    negative = text.startswith("-")
    magnitude = text[1:] if negative else text
    whole, separator, fraction = magnitude.partition(".")
    token = whole.zfill(3)
    if separator:
        token += f"p{fraction}"
    return f"m{token}" if negative else f"p{token}"


def physical_case_name(
    pairing: str,
    angle_deg: int | float,
    *,
    temperature_K: int | float = DEFAULT_TEMPERATURE_K,
    separation_nm: int | float = DEFAULT_SEPARATION_NM,
) -> str:
    """Name one physical case without a human development-version suffix."""

    if pairing not in DEFAULT_PAIRINGS:
        raise ValueError(f"unsupported pairing: {pairing}")
    temperature = float(_decimal(temperature_K, label="temperature_K"))
    separation = float(_decimal(separation_nm, label="separation_nm"))
    if temperature <= 0.0:
        raise ValueError("temperature_K must be positive")
    if separation <= 0.0:
        raise ValueError("separation_nm must be positive")
    temperature_token = _unsigned_token(temperature, label="temperature_K")
    separation_token = _unsigned_token(separation, label="separation_nm")
    return (
        f"{pairing}_T{temperature_token}K_d{separation_token}nm_"
        f"theta_{angle_token(angle_deg)}deg"
    )


def case_name(
    pairing: str,
    angle_deg: int | float,
    *,
    temperature_K: int | float = DEFAULT_TEMPERATURE_K,
    separation_nm: int | float = DEFAULT_SEPARATION_NM,
    profile: str = PROFILE_NAME,
) -> str:
    """Legacy case naming surface retained for historical workflows."""

    return (
        f"{physical_case_name(pairing, angle_deg, temperature_K=temperature_K, separation_nm=separation_nm)}_"
        f"{profile}"
    )


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
    if worker_cap < 0:
        raise ValueError("worker_cap must be non-negative; zero means no cap")
    visible = tuple(
        sorted(
            os.sched_getaffinity(0)
            if available_cpus is None
            else {int(value) for value in available_cpus}
        )
    )
    if not visible:
        raise RuntimeError("no CPUs are visible to the process")
    available_after_reserve = max(1, len(visible) - reserve_logical_cpus)
    target = (
        available_after_reserve
        if worker_cap == 0
        else min(worker_cap, available_after_reserve)
    )
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
    os.environ.update(
        {
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
        }
    )


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
