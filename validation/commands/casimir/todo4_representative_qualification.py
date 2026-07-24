"""Validation-only CLI for the frozen TODO 4 representative qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lno327.casimir.material_geometry_qualification_execution import (
    execute_campaign_geometry,
    execute_legacy_shard,
    freeze_plan,
    load_campaign,
    populate_shard,
    verify_campaign,
    write_preflight,
)

DEFAULT_MANIFEST = Path(
    "validation/configs/casimir/todo4_representative_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "validation/outputs/casimir/todo4_representative_v1"
)
DEFAULT_CACHE_ROOT = Path("validation/cache/material_response")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "preflight", "populate", "geometry", "legacy", "verify"):
        command = subparsers.add_parser(action)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if action != "plan":
            command.add_argument(
                "--cache-root",
                type=Path,
                default=DEFAULT_CACHE_ROOT,
            )
        if action in {"populate", "legacy"}:
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--shard-count", type=int, default=1)
        if action == "preflight":
            command.add_argument("--require-complete", action="store_true")
    return parser


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    campaign = load_campaign(args.manifest)

    if args.action == "plan":
        payload = freeze_plan(
            campaign,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        _print(payload["summary"])
        return

    if args.action == "preflight":
        payload = write_preflight(
            campaign,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            require_complete=bool(args.require_complete),
        )
        _print(payload["summary"])
        return

    if args.action == "populate":
        payload = populate_shard(
            campaign,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        _print(
            {
                "selected_group_count": payload["selected_group_count"],
                "total_group_count": payload["total_group_count"],
                "passed": payload["passed"],
            }
        )
        if not payload["passed"]:
            raise SystemExit("one or more response groups were unresolved")
        return

    if args.action == "geometry":
        payload = execute_campaign_geometry(
            campaign,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
        )
        _print(
            {
                "plan_count": len(payload["records"]),
                "passed": payload["passed"],
            }
        )
        if not payload["passed"]:
            raise SystemExit("scalar versus batch geometry qualification failed")
        return

    if args.action == "legacy":
        payload = execute_legacy_shard(
            campaign,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        _print(
            {
                "selected_point_count": payload["selected_point_count"],
                "total_point_count": payload["total_point_count"],
                "passed": payload["passed"],
            }
        )
        if not payload["passed"]:
            raise SystemExit("one or more matched legacy points failed")
        return

    payload = verify_campaign(
        campaign,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
    )
    _print(
        {
            "passed": payload["passed"],
            "geometry_plan_count": len(payload["geometry"]),
            "legacy_point_count": len(payload["legacy"]),
            "fixed_outer_count": len(payload["fixed_outer"]),
        }
    )
    if not payload["passed"]:
        raise SystemExit("TODO 4 representative qualification did not pass")


if __name__ == "__main__":
    main()
