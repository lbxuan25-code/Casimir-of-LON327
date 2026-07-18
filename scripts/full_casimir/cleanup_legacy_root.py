from __future__ import annotations

from pathlib import Path
import argparse

from .config import REPO_ROOT


LEGACY_ROOT_SCRIPTS = (
    "run_full_casimir_case_N896.py",
    "run_full_casimir_N896_scan.sh",
    "run_full_casimir_angle_sweep.sh",
    "run_full_casimir_torque_scan.sh",
    "extract_casimir_torque.py",
    "run_0deg_runtime_budget_pilots.sh",
    "summarize_0deg_runtime_budget_pilots.py",
    "run_0deg_pilots.sh",
    "start_0deg_pilots.sh",
    "stop_0deg_pilots.sh",
)


def cleanup_legacy_root_scripts(*, root: Path = REPO_ROOT) -> list[Path]:
    removed: list[Path] = []
    for name in LEGACY_ROOT_SCRIPTS:
        path = root / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove obsolete root-level Casimir helper scripts."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    removed = cleanup_legacy_root_scripts()
    if not args.quiet:
        if removed:
            for path in removed:
                print(f"removed: {path.relative_to(REPO_ROOT)}")
        else:
            print("no legacy root-level scripts were present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
