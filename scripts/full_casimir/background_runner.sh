#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: background_runner.sh EXIT_FILE COMMAND [ARG ...]" >&2
  exit 64
fi

EXIT_FILE="$1"
shift

write_exit_code() {
  local rc="$1"
  local temporary="${EXIT_FILE}.tmp.$$"
  mkdir -p "$(dirname "$EXIT_FILE")"
  printf '%s\n' "$rc" > "$temporary"
  mv -f "$temporary" "$EXIT_FILE"
}

on_exit() {
  local rc=$?
  trap - EXIT
  write_exit_code "$rc"
  exit "$rc"
}

trap on_exit EXIT

"$@"
exit $?
