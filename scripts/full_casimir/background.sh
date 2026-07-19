#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Fix thread policy before Python imports NumPy or a BLAS library.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MALLOC_ARENA_MAX=4

LOG_ROOT="outputs/casimir/workflow_logs"
PID_FILE="$LOG_ROOT/background.pid"
START_TICKS_FILE="$LOG_ROOT/background.start_ticks"
PGID_FILE="$LOG_ROOT/background.pgid"
MODE_FILE="$LOG_ROOT/background.mode"
COMMAND_FILE="$LOG_ROOT/background.command"
LOG_PATH_FILE="$LOG_ROOT/background.log_path"
LATEST_LOG="$LOG_ROOT/background.log"

mkdir -p "$LOG_ROOT"

usage()
{
    cat <<'EOF'
Usage:
  bash scripts/full_casimir/background.sh start pilots [workflow options]
  bash scripts/full_casimir/background.sh start scan   [workflow options]
  bash scripts/full_casimir/background.sh start all    [workflow options]
  bash scripts/full_casimir/background.sh start torque [workflow options]
  bash scripts/full_casimir/background.sh start plot   [workflow options]
  bash scripts/full_casimir/background.sh status
  bash scripts/full_casimir/background.sh logs
  bash scripts/full_casimir/background.sh stop
EOF
}

process_start_ticks()
{
    local pid="$1"
    [[ -r "/proc/$pid/stat" ]] || return 1
    local stat_line rest
    stat_line="$(cat "/proc/$pid/stat")"
    rest="${stat_line#*) }"
    awk '{print $20}' <<< "$rest"
}

clear_process_identity()
{
    rm -f "$PID_FILE" "$START_TICKS_FILE" "$PGID_FILE"
}

is_running()
{
    [[ -f "$PID_FILE" && -f "$START_TICKS_FILE" ]] || return 1
    local pid expected actual
    pid="$(cat "$PID_FILE")"
    expected="$(cat "$START_TICKS_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    actual="$(process_start_ticks "$pid" 2>/dev/null || true)"
    [[ -n "$actual" && "$actual" == "$expected" ]]
}

current_log()
{
    if [[ -f "$LOG_PATH_FILE" ]]; then
        cat "$LOG_PATH_FILE"
    else
        printf '%s\n' "$LATEST_LOG"
    fi
}

start_job()
{
    local mode="${1:-}"
    shift || true

    case "$mode" in
        pilots|scan|all|torque|plot)
            ;;
        *)
            echo "ERROR: start mode must be pilots, scan, all, torque, or plot" >&2
            usage
            exit 64
            ;;
    esac

    if is_running; then
        echo "ERROR: a background workflow is already running (PID=$(cat "$PID_FILE"))" >&2
        echo "Use: bash scripts/full_casimir/background.sh status" >&2
        exit 1
    fi
    clear_process_identity

    local python_bin
    python_bin="$(command -v python)"
    local command=(
        "$python_bin"
        -m
        scripts.full_casimir.workflow
        "$mode"
        "$@"
    )
    local run_id run_log
    run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    run_log="$LOG_ROOT/background.$run_id.log"

    printf '%q ' "${command[@]}" > "$COMMAND_FILE"
    printf '\n' >> "$COMMAND_FILE"
    printf '%s\n' "$mode" > "$MODE_FILE"
    printf '%s\n' "$run_log" > "$LOG_PATH_FILE"
    : > "$run_log"
    ln -sfn "$(basename "$run_log")" "$LATEST_LOG"

    nohup \
        setsid \
        nice -n 15 \
        ionice -c 2 -n 7 \
        "${command[@]}" \
        >> "$run_log" \
        2>&1 \
        < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    sleep 2

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: workflow exited immediately" >&2
        cat "$run_log" >&2
        clear_process_identity
        exit 1
    fi

    local ticks pgid
    ticks="$(process_start_ticks "$pid")"
    pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
    if [[ -z "$ticks" || ! "$pgid" =~ ^[0-9]+$ ]]; then
        echo "ERROR: cannot establish background process identity" >&2
        kill -TERM "$pid" 2>/dev/null || true
        clear_process_identity
        exit 1
    fi
    printf '%s\n' "$ticks" > "$START_TICKS_FILE"
    printf '%s\n' "$pgid" > "$PGID_FILE"

    echo "started: mode=$mode"
    echo "PID: $pid"
    echo "PGID: $pgid"
    echo "log: $run_log"
    echo "command: $(cat "$COMMAND_FILE")"
}

status_job()
{
    if is_running; then
        local pid pgid log_path
        pid="$(cat "$PID_FILE")"
        pgid="$(cat "$PGID_FILE" 2>/dev/null || echo unknown)"
        log_path="$(current_log)"
        echo "running"
        echo "mode: $(cat "$MODE_FILE" 2>/dev/null || echo unknown)"
        echo "PID: $pid"
        echo "PGID: $pgid"
        echo "log: $log_path"
        echo "command: $(cat "$COMMAND_FILE" 2>/dev/null || true)"
        ps -o pid,ppid,sid,pgid,ni,psr,pcpu,pmem,rss,etime,cmd -p "$pid"
        echo
        echo "active Casimir processes:"
        pgrep -af 'scripts.full_casimir.workflow|lno327.casimir|fixed_transverse_point_certification' || true
    else
        echo "not running"
        clear_process_identity
        local log_path
        log_path="$(current_log)"
        if [[ -f "$log_path" || -L "$log_path" ]]; then
            echo "last log: $log_path"
            tail -n 20 "$log_path"
        fi
    fi
}

stop_job()
{
    if ! is_running; then
        echo "no active background workflow"
        clear_process_identity
        return 0
    fi

    local pid recorded_pgid actual_pgid
    pid="$(cat "$PID_FILE")"
    recorded_pgid="$(cat "$PGID_FILE")"
    actual_pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
    if [[ "$actual_pgid" != "$recorded_pgid" ]]; then
        echo "ERROR: recorded PGID no longer matches PID; refusing to signal" >&2
        return 1
    fi

    echo "stopping process group $recorded_pgid"
    kill -TERM -- "-$recorded_pgid"

    for _ in $(seq 1 30); do
        if ! is_running; then
            echo "stopped"
            clear_process_identity
            return 0
        fi
        sleep 1
    done

    echo "process is still alive after 30 seconds" >&2
    echo "force stop with: kill -KILL -- -$recorded_pgid" >&2
    return 1
}

show_logs()
{
    local log_path
    log_path="$(current_log)"
    touch "$log_path"
    tail -f "$log_path"
}

case "${1:-}" in
    start)
        shift
        start_job "$@"
        ;;
    status)
        status_job
        ;;
    stop)
        stop_job
        ;;
    logs)
        show_logs
        ;;
    *)
        usage
        exit 64
        ;;
esac
