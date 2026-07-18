#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

# ============================================================
# Physical configuration
# ============================================================
TEMPERATURE_K=10
SEPARATION_NM=20

ANGLE_MIN=-4
ANGLE_MAX=94
ANGLE_STEP=2

# ============================================================
# Extended transverse-N ladder
# ============================================================
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

REQUIRED_CONSECUTIVE_PASSES=2

# ============================================================
# Parallel configuration
# ============================================================
WORKERS=30

# 0 means automatic: approximately 70% of currently available
# memory according to the production resource planner.
MEMORY_BUDGET_GB=0

# Keep only one full material context live at high N.
# The shared-context q worker count can still reach 30.
MAX_CONTEXT_WORKERS=1

# ============================================================
# Accuracy configuration
# ============================================================
RTOL=5e-3
ATOL_J_M2=1e-12

OUTPUT_ROOT="outputs/casimir/runs"
LOG_ROOT="outputs/casimir/N896_scan_logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

DRIVER_LOG="$LOG_ROOT/N896_scan_driver.log"
STATUS_TSV="$LOG_ROOT/N896_scan_status.tsv"

if [[ ! -f "$STATUS_TSV" ]]; then
    printf \
        "pairing\tangle_deg\tcase\texit_code\tstatus\ttermination_reason\n" \
        > "$STATUS_TSV"
fi

# Select exactly 30 CPUs from the current process affinity.
CPUSET=$(
    python - <<'PY'
import os

available = sorted(os.sched_getaffinity(0))

if len(available) < 30:
    raise SystemExit(
        f"only {len(available)} affinity CPUs are available"
    )

selected = available[:30]
print(",".join(str(value) for value in selected))
PY
)

log()
{
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" \
        | tee -a "$DRIVER_LOG"
}

angle_token()
{
    local angle="$1"

    if (( angle < 0 )); then
        printf 'm%03d' "$((-angle))"
    else
        printf 'p%03d' "$angle"
    fi
}

record_status()
{
    local pairing="$1"
    local angle="$2"
    local case_name="$3"
    local exit_code="$4"
    local summary_path="$5"

    python - \
        "$pairing" \
        "$angle" \
        "$case_name" \
        "$exit_code" \
        "$summary_path" \
        "$STATUS_TSV" <<'PY'
import json
import sys
from pathlib import Path

(
    pairing,
    angle,
    case_name,
    exit_code,
    summary_name,
    table_name,
) = sys.argv[1:]

status = "missing_summary"
reason = ""

summary_path = Path(summary_name)

if summary_path.exists():
    payload = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    status = str(payload.get("status", "unknown"))
    reason = str(payload.get("termination_reason", ""))

with Path(table_name).open("a", encoding="utf-8") as handle:
    handle.write(
        f"{pairing}\t{angle}\t{case_name}\t"
        f"{exit_code}\t{status}\t{reason}\n"
    )
PY
}

run_case()
{
    local pairing="$1"
    local angle="$2"
    local token
    local case_name
    local run_dir
    local stdout_log
    local stderr_log
    local time_log
    local exit_code

    token=$(angle_token "$angle")

    case_name="${pairing}_T10K_d20nm_theta_${token}deg_N896_grid2"
    run_dir="$OUTPUT_ROOT/$case_name"

    stdout_log="$LOG_ROOT/${case_name}.stdout.log"
    stderr_log="$LOG_ROOT/${case_name}.stderr.log"
    time_log="$LOG_ROOT/${case_name}.time.txt"

    if [[ -f "$run_dir/summary.json" ]]; then
        log "SKIP result-present case: $case_name"

        record_status \
            "$pairing" \
            "$angle" \
            "$case_name" \
            "skipped" \
            "$run_dir/summary.json"

        return 0
    fi

    resume_args=()

    if [[ -d "$run_dir" ]]; then
        resume_args=(--resume)
        log "RESUME case: $case_name"
    else
        log "START case: $case_name"
    fi

    command=(
        taskset
        -c
        "$CPUSET"
        python
        run_full_casimir_case_N896.py
        --case
        "$case_name"
        --output-root
        "$OUTPUT_ROOT"
        "${resume_args[@]}"
        --pairing
        "$pairing"
        --temperature-K
        "$TEMPERATURE_K"
        --separation-nm
        "$SEPARATION_NM"
        --angle-deg
        "$angle"
        --workers
        "$WORKERS"
        --parallel-mode
        auto
        --memory-budget-gb
        "$MEMORY_BUDGET_GB"
        --max-context-workers
        "$MAX_CONTEXT_WORKERS"
        --N-candidates
        "${N_CANDIDATES[@]}"
        --required-consecutive-passes
        "$REQUIRED_CONSECUTIVE_PASSES"
        --matsubara-cutoffs
        1 3 7 15 31
        --outer-cutoffs-u
        6 10 14 18 24 30 36 42
        --rtol
        "$RTOL"
        --atol-J-m2
        "$ATOL_J_M2"
    )

    printf '%q ' "${command[@]}" \
        > "$LOG_ROOT/${case_name}.command.txt"
    printf '\n' >> "$LOG_ROOT/${case_name}.command.txt"

    log "CPU affinity: $CPUSET"
    log "N ladder: ${N_CANDIDATES[*]}"

    /usr/bin/time -v \
        -o "$time_log" \
        "${command[@]}" \
        > >(tee "$stdout_log") \
        2> >(tee "$stderr_log" >&2)

    exit_code=$?

    record_status \
        "$pairing" \
        "$angle" \
        "$case_name" \
        "$exit_code" \
        "$run_dir/summary.json"

    case "$exit_code" in
        0)
            log "CONVERGED case: $case_name"
            ;;
        2)
            log "UNRESOLVED case retained: $case_name"
            ;;
        *)
            log "FAILED case: $case_name, exit_code=$exit_code"
            log "Inspect: $stderr_log"
            log "Resource report: $time_log"
            exit "$exit_code"
            ;;
    esac
}

mode="${1:-full}"

case "$mode" in
    pilot)
        pairings=(spm dwave)
        angles=(44)
        ;;
    full)
        pairings=(spm dwave)
        mapfile -t angles < <(
            seq "$ANGLE_MIN" "$ANGLE_STEP" "$ANGLE_MAX"
        )
        ;;
    *)
        echo "usage: $0 [pilot|full]" >&2
        exit 64
        ;;
esac

log "N896 Casimir scan starts"
log "mode=$mode"
log "CPUSET=$CPUSET"
log "workers=$WORKERS"
log "max_context_workers=$MAX_CONTEXT_WORKERS"
log "memory_budget_gb=$MEMORY_BUDGET_GB"
log "N_candidates=${N_CANDIDATES[*]}"
log "RTOL=$RTOL"
log "ATOL_J_m2=$ATOL_J_M2"

for pairing in "${pairings[@]}"; do
    for angle in "${angles[@]}"; do
        run_case "$pairing" "$angle"
    done
done

log "N896 Casimir scan finished: mode=$mode"
