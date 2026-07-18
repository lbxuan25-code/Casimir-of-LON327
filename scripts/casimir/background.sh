#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="outputs/casimir/workflow_logs"
PID_FILE="$LOG_ROOT/background.pid"
MODE_FILE="$LOG_ROOT/background.mode"
COMMAND_FILE="$LOG_ROOT/background.command"
DRIVER_LOG="$LOG_ROOT/background.log"

mkdir -p "$LOG_ROOT"

usage()
{
    cat <<'EOF'
Usage:
  bash scripts/casimir/background.sh start pilots [workflow options]
  bash scripts/casimir/background.sh start scan   [workflow options]
  bash scripts/casimir/background.sh start all    [workflow options]
  bash scripts/casimir/background.sh status
  bash scripts/casimir/background.sh logs
  bash scripts/casimir/background.sh stop

Examples:
  bash scripts/casimir/background.sh start pilots
  bash scripts/casimir/background.sh start scan
  bash scripts/casimir/background.sh start all
EOF
}

is_running()
{
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$PID_FILE")"
    kill -0 "$pid" 2>/dev/null
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
        echo "Use: bash scripts/casimir/background.sh status" >&2
        exit 1
    fi

    local python_bin
    python_bin="$(command -v python)"
    local command=(
        "$python_bin"
        -m
        scripts.casimir.workflow
        "$mode"
        "$@"
    )

    : > "$DRIVER_LOG"
    printf '%q ' "${command[@]}" > "$COMMAND_FILE"
    printf '\n' >> "$COMMAND_FILE"
    printf '%s\n' "$mode" > "$MODE_FILE"

    nohup \
        setsid \
        nice -n 15 \
        ionice -c 2 -n 7 \
        "${command[@]}" \
        >> "$DRIVER_LOG" \
        2>&1 \
        < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    sleep 2

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: workflow exited immediately" >&2
        cat "$DRIVER_LOG" >&2
        exit 1
    fi

    echo "started: mode=$mode"
    echo "PID: $pid"
    echo "log: $DRIVER_LOG"
    echo "command: $(cat "$COMMAND_FILE")"
}

status_job()
{
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "running"
        echo "mode: $(cat "$MODE_FILE" 2>/dev/null || echo unknown)"
        echo "PID: $pid"
        echo "command: $(cat "$COMMAND_FILE" 2>/dev/null || true)"
        ps -o pid,ppid,sid,pgid,ni,psr,pcpu,pmem,rss,etime,cmd -p "$pid"
        echo
        echo "active Casimir processes:"
        pgrep -af 'scripts.casimir.workflow|lno327.casimir|fixed_transverse_point_certification' || true
    else
        echo "not running"
        if [[ -f "$PID_FILE" ]]; then
            echo "last PID: $(cat "$PID_FILE")"
        fi
        if [[ -f "$DRIVER_LOG" ]]; then
            echo "last log: $DRIVER_LOG"
            tail -n 20 "$DRIVER_LOG"
        fi
    fi
}

stop_job()
{
    if ! is_running; then
        echo "no active background workflow"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "stopping process group $pid"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

    for _ in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "stopped"
            return 0
        fi
        sleep 1
    done

    echo "process is still alive after 30 seconds" >&2
    echo "force stop with: kill -KILL -- -$pid" >&2
    return 1
}

show_logs()
{
    touch "$DRIVER_LOG"
    tail -f "$DRIVER_LOG"
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
