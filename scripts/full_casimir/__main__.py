from __future__ import annotations

from importlib import import_module
from typing import Callable, Sequence


_HELP = """\
Unified LNO327 full-Casimir command interface.

Usage:
  python -m scripts.full_casimir <command> [options]

Production commands:
  plan         Freeze a scientific policy and physical case matrix.
  run          Execute or resume a locked, SHA-confirmed formal campaign.
  resources    Show the execution resources selected for a formal run.

Read-only monitoring, proof, analysis and data commands:
  status       Inspect or watch persisted campaign progress.
  proof        Verify source identity and authoritative artifact digests.
  diagnose     Diagnose completed or interrupted campaign artifacts.
  audit        Build a convergence audit without starting production work.
  shift-audit  Replay historical three-shift evidence under the two-shift policy.
  torque       Post-process existing energy results.
  plot         Plot existing post-processed results.
  data         Catalog, archive, verify, or explicitly prune local data.
  layout       Inspect or migrate the output layout.

No other script or package command is an authorized Casimir calculation route.
Use '<command> --help' for command-specific options.
"""


_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "plan": ("scripts.full_casimir.scan", ("plan",)),
    "run": ("scripts.full_casimir.run_command", ()),
    "resources": ("scripts.full_casimir.scan", ("resources",)),
    "status": ("scripts.full_casimir.progress", ()),
    "proof": ("scripts.full_casimir.reproducibility", ()),
    "diagnose": ("scripts.full_casimir.diagnostics", ("diagnose",)),
    "audit": ("scripts.full_casimir.diagnostics", ("audit",)),
    "shift-audit": ("scripts.full_casimir.shift_audit", ()),
    "torque": ("scripts.full_casimir.analysis", ("torque",)),
    "plot": ("scripts.full_casimir.analysis", ("plot",)),
    "data": ("scripts.full_casimir.data", ()),
    "layout": ("scripts.full_casimir.layout", ()),
}


def _module_main(module_name: str) -> Callable[[Sequence[str] | None], int]:
    module = import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError(f"command module has no callable main(): {module_name}")
    return main


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        import sys

        raw = list(sys.argv[1:])
    else:
        raw = list(argv)
    if not raw or raw[0] in ("-h", "--help", "help"):
        print(_HELP)
        return 0
    command = raw[0]
    target = _COMMANDS.get(command)
    if target is None:
        print(f"unknown command: {command}\n")
        print(_HELP)
        return 2
    module_name, prefix = target
    try:
        return int(_module_main(module_name)([*prefix, *raw[1:]]))
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"COMMAND FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
