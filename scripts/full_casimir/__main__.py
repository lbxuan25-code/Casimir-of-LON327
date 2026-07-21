from __future__ import annotations

from importlib import import_module
from typing import Callable, Sequence


_HELP = """\
Unified LNO327 full-Casimir command interface.

Usage:
  python -m scripts.full_casimir <command> [options]

Primary production commands:
  plan         Resolve a single or multi-angle/multi-distance case matrix.
  run          Execute the resolved physical case matrix.
  resources    Show the CPU selection used by production runs.

Analysis and data commands:
  diagnose     Diagnose one or more completed or interrupted runs.
  audit        Build a convergence audit across runs.
  shift-audit  Replay historical three-shift histories under the formal two-shift policy.
  torque       Run the existing torque post-processing path.
  plot         Plot existing post-processed results.
  data         Catalog, archive, verify, or explicitly prune local data.
  layout       Inspect or migrate the output layout.

Frozen qualification compatibility commands:
  qualification          Run the frozen qualification command group.
  qualification-holdout  Run the parallel holdout executor.
  qualification-verify   Verify a completed frozen qualification.

Compatibility command:
  legacy-workflow  Access the previous workflow.py command surface.

Use '<command> --help' for command-specific options.
"""


_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "plan": ("scripts.full_casimir.scan", ("plan",)),
    "run": ("scripts.full_casimir.scan", ("run",)),
    "resources": ("scripts.full_casimir.scan", ("resources",)),
    "diagnose": ("scripts.full_casimir.diagnostics", ("diagnose",)),
    "audit": ("scripts.full_casimir.diagnostics", ("audit",)),
    "shift-audit": ("scripts.full_casimir.shift_audit", ()),
    "torque": ("scripts.full_casimir.workflow", ("torque",)),
    "plot": ("scripts.full_casimir.workflow", ("plot",)),
    "data": ("scripts.full_casimir.data", ()),
    "layout": ("scripts.full_casimir.layout", ()),
    "qualification": ("scripts.full_casimir.qualification", ()),
    "qualification-holdout": (
        "scripts.full_casimir.qualification_holdout",
        (),
    ),
    "qualification-verify": (
        "scripts.full_casimir.qualification_verify",
        (),
    ),
    "legacy-workflow": ("scripts.full_casimir.workflow", ()),
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
