#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

EXPECTED_COMMIT="74e262a80d1c1a7315eb583af0adced88839e26b"

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "ERROR: 当前 HEAD 不是预期提交" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

# 减少多进程内存分配器额外 arena。
export MALLOC_ARENA_MAX=4

TEMPERATURE_K=10
SEPARATION_NM=20
ANGLE_1_DEG=0
ANGLE_2_DEG=0

N_CANDIDATES=(
    128
    192
    256
    384
    512
    640
    768
    896
)

MATSUBARA_CUTOFFS=(
    1
    3
    7
    11
    15
    23
    31
)

OUTER_CUTOFFS=(
    6
    10
    14
    18
    24
    30
    36
    42
)

RTOL=5e-3
ATOL_J_M2=1e-12

# 留出较充足内存给 Windows、浏览器和桌面程序。
MEMORY_BUDGET_GB=16
MAX_CONTEXT_WORKERS=1

OUTPUT_ROOT="outputs/casimir/runs"
LOG_ROOT="outputs/casimir/0deg_runtime_budget_pilot_logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

DRIVER_LOG="$LOG_ROOT/driver.log"

log()
{
    printf '[%s] %s\n' \
        "$(date --iso-8601=seconds)" \
        "$*" \
        | tee -a "$DRIVER_LOG"
}

# ------------------------------------------------------------
# 尽量预留两个完整物理核，而不是随便空出四个 SMT 线程。
# 32 线程 / 16 核机器通常会得到 28 个可计算逻辑 CPU。
# ------------------------------------------------------------
read -r CPUSET RESERVED_CPUSET WORKERS < <(
    python - <<'PY'
from collections import defaultdict
from pathlib import Path
import os

available = sorted(os.sched_getaffinity(0))

if len(available) < 8:
    raise SystemExit(
        f"only {len(available)} logical CPUs are visible"
    )

target = min(28, max(1, len(available) - 4))

groups = defaultdict(list)

for cpu in available:
    topology = Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology"
    )

    try:
        package = int(
            (topology / "physical_package_id").read_text()
        )
        core = int((topology / "core_id").read_text())
        key = (package, core)
    except (OSError, ValueError):
        key = (0, cpu)

    groups[key].append(cpu)

selected = set(available)
reserved = []

for key in reversed(sorted(groups)):
    if len(selected) <= target:
        break

    group = groups[key]

    # 预留整个物理核。
    for cpu in group:
        selected.discard(cpu)
        reserved.append(cpu)

selected = sorted(selected)
reserved = sorted(reserved)

if len(selected) > target:
    extra = selected[target:]
    selected = selected[:target]
    reserved.extend(extra)
    reserved = sorted(set(reserved))

if len(selected) < 24:
    raise SystemExit(
        f"only {len(selected)} CPUs remain after reservation"
    )

print(
    ",".join(str(cpu) for cpu in selected),
    ",".join(str(cpu) for cpu in reserved) or "none",
    len(selected),
)
PY
)

log "selected CPU set: $CPUSET"
log "reserved CPU set: $RESERVED_CPUSET"
log "worker count: $WORKERS"
log "memory budget: ${MEMORY_BUDGET_GB} GiB"
log "N ladder: ${N_CANDIDATES[*]}"
log "Matsubara ladder: ${MATSUBARA_CUTOFFS[*]}"

run_case()
{
    local pairing="$1"
    local case_name
    local run_dir
    local manifest_status=""
    local stdout_log
    local stderr_log
    local time_log
    local command_log
    local exit_code
    local resume_args=()

    case_name="${pairing}_T10K_d20nm_theta0deg_runtime_budget_pilot"
    run_dir="$OUTPUT_ROOT/$case_name"

    stdout_log="$LOG_ROOT/${case_name}.stdout.log"
    stderr_log="$LOG_ROOT/${case_name}.stderr.log"
    time_log="$LOG_ROOT/${case_name}.time.txt"
    command_log="$LOG_ROOT/${case_name}.command.txt"

    if [[ -f "$run_dir/manifest.json" ]]; then
        manifest_status=$(
            python - "$run_dir/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload.get("status", "unknown"))
PY
        )
    fi

    if [[ "$manifest_status" == "completed" ]]; then
        log "SKIP already converged case: $case_name"
        return 0
    fi

    if [[ -d "$run_dir" ]]; then
        resume_args=(--resume)
        log "RESUME case: $case_name"
    else
        log "START fresh case: $case_name"
    fi

    command=(
        python
        -m
        lno327.casimir
        --case
        "$case_name"
        --output-root
        "$OUTPUT_ROOT"
        "${resume_args[@]}"
        --pairings
        "$pairing"
        --temperature-K
        "$TEMPERATURE_K"
        --separation-nm
        "$SEPARATION_NM"
        --plate-angles-deg
        "$ANGLE_1_DEG"
        "$ANGLE_2_DEG"
        --N-candidates
        "${N_CANDIDATES[@]}"
        --workers
        "$WORKERS"
        --parallel-mode
        auto
        --memory-budget-gb
        "$MEMORY_BUDGET_GB"
        --max-context-workers
        "$MAX_CONTEXT_WORKERS"
        --matsubara-cutoffs
        "${MATSUBARA_CUTOFFS[@]}"
        --outer-cutoffs-u
        "${OUTER_CUTOFFS[@]}"
        --rtol
        "$RTOL"
        --atol-J-m2
        "$ATOL_J_M2"
    )

    printf '%q ' "${command[@]}" > "$command_log"
    printf '\n' >> "$command_log"

    log "running pairing=$pairing"
    log "case=$case_name"

    if /usr/bin/time -v \
        -o "$time_log" \
        nice -n 15 \
        ionice -c 2 -n 7 \
        taskset -c "$CPUSET" \
        "${command[@]}" \
        > >(tee "$stdout_log") \
        2> >(tee "$stderr_log" >&2)
    then
        exit_code=0
    else
        exit_code=$?
    fi

    case "$exit_code" in
        0)
            log "CONVERGED case: $case_name"
            ;;
        2)
            log "UNRESOLVED but artifacts retained: $case_name"
            ;;
        130)
            log "INTERRUPTED case: $case_name"
            exit 130
            ;;
        *)
            log "ENGINEERING FAILURE: $case_name"
            log "exit_code=$exit_code"
            log "stderr=$stderr_log"
            log "resources=$time_log"
            exit "$exit_code"
            ;;
    esac
}

# 必须顺序执行，不允许两个 28-worker case 同时运行。
run_case spm
run_case dwave

log "both 0-degree pilots finished"
