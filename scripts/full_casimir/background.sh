#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/background_runner.sh"
cd "$REPO_ROOT"
LOG_ROOT="outputs/casimir/workflow_logs"
PID_FILE="$LOG_ROOT/background.pid"
START_FILE="$LOG_ROOT/background.start_ticks"
MODE_FILE="$LOG_ROOT/background.mode"
COMMAND_FILE="$LOG_ROOT/background.command"
DRIVER_LOG="$LOG_ROOT/background.log"
EXIT_FILE="$LOG_ROOT/background.exit_code"
LOCK_FILE="$LOG_ROOT/background.lock"
mkdir -p "$LOG_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/full_casimir/background.sh start pilots [workflow options]
  bash scripts/full_casimir/background.sh start scan   [workflow options]
  bash scripts/full_casimir/background.sh start all    [workflow options]
  bash scripts/full_casimir/background.sh status|logs|stop
EOF
}

proc_start_ticks() {
  [[ -r "/proc/$1/stat" ]] || return 1
  awk '{print $22}' "/proc/$1/stat"
}

cleanup_stale() {
  rm -f "$PID_FILE" "$START_FILE" "$MODE_FILE"
}

is_running() {
  [[ -f "$PID_FILE" && -f "$START_FILE" ]] || return 1
  local pid expected actual cmdline
  pid="$(cat "$PID_FILE")"
  expected="$(cat "$START_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  actual="$(proc_start_ticks "$pid" 2>/dev/null || true)"
  [[ -n "$actual" && "$actual" == "$expected" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *"background_runner.sh"* || "$cmdline" == *"scripts.full_casimir.workflow"* ]]
}

start_job() {
  local mode="${1:-}"; shift || true
  case "$mode" in pilots|scan|all|torque|plot) ;; *) usage; exit 64;; esac

  # Never fall back to an unlocked check-then-write sequence.  Concurrent production
  # starts are unsafe because they share PID, log, cache and run-artifact paths.
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for safe background workflow startup" >&2
    exit 1
  }
  local lock_fd
  exec {lock_fd}>"$LOCK_FILE"
  if ! flock -n "$lock_fd"; then
    echo "ERROR: another background start operation is in progress" >&2
    exit 1
  fi

  if is_running; then
    echo "ERROR: workflow already running (PID=$(cat "$PID_FILE"))" >&2; exit 1
  fi
  [[ -f "$RUNNER" ]] || { echo "ERROR: missing background runner: $RUNNER" >&2; exit 1; }
  cleanup_stale
  local python_bin; python_bin="$(command -v python)"
  local command=("$python_bin" -m scripts.full_casimir.workflow "$mode" "$@")
  printf '\n[%s] START mode=%s\n' "$(date --iso-8601=seconds)" "$mode" >> "$DRIVER_LOG"
  printf '%q ' "${command[@]}" > "$COMMAND_FILE"; printf '\n' >> "$COMMAND_FILE"
  printf '%s\n' "$mode" > "$MODE_FILE"
  rm -f "$EXIT_FILE"
  local launcher=(setsid nice -n 15)
  if command -v ionice >/dev/null 2>&1; then launcher+=(ionice -c 2 -n 7); fi
  nohup "${launcher[@]}" bash "$RUNNER" "$EXIT_FILE" "${command[@]}" \
    >> "$DRIVER_LOG" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  local ticks=""
  for _ in $(seq 1 20); do
    ticks="$(proc_start_ticks "$pid" 2>/dev/null || true)"
    [[ -n "$ticks" ]] && break
    sleep 0.1
  done
  [[ -n "$ticks" ]] && printf '%s\n' "$ticks" > "$START_FILE"
  sleep 1
  if is_running; then
    echo "started: mode=$mode"; echo "PID: $pid"; echo "log: $DRIVER_LOG"; return 0
  fi
  set +e; wait "$pid"; local rc=$?; set -e
  if [[ -f "$EXIT_FILE" ]]; then rc="$(cat "$EXIT_FILE")"; fi
  cleanup_stale
  if [[ "$rc" -eq 0 ]]; then
    echo "workflow completed before background status check"; return 0
  fi
  echo "ERROR: workflow exited immediately with code $rc" >&2
  tail -n 80 "$DRIVER_LOG" >&2
  return "$rc"
}

status_job() {
  if is_running; then
    local pid; pid="$(cat "$PID_FILE")"
    echo "running"; echo "mode: $(cat "$MODE_FILE")"; echo "PID: $pid"
    echo "command: $(cat "$COMMAND_FILE" 2>/dev/null || true)"
    ps -eo pid,ppid,sid,pgid,ni,psr,pcpu,pmem,rss,etime,cmd | awk -v sid="$pid" 'NR==1 || $3==sid'
  else
    cleanup_stale
    echo "not running"
    [[ -f "$EXIT_FILE" ]] && echo "last exit code: $(cat "$EXIT_FILE")"
    [[ -f "$DRIVER_LOG" ]] && { echo "last log: $DRIVER_LOG"; tail -n 25 "$DRIVER_LOG"; }
  fi
}

stop_job() {
  if ! is_running; then cleanup_stale; echo "no active background workflow"; return 0; fi
  local pid; pid="$(cat "$PID_FILE")"
  echo "stopping process group $pid"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! is_running; then cleanup_stale; echo "stopped"; return 0; fi
    sleep 1
  done
  echo "process group still alive after 30 seconds" >&2; return 1
}

case "${1:-}" in
  start) shift; start_job "$@";;
  status) status_job;;
  logs) touch "$DRIVER_LOG"; tail -f "$DRIVER_LOG";;
  stop) stop_job;;
  *) usage; exit 64;;
esac
