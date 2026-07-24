"""Validation-only CLI for the frozen TODO 4 representative qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lno327.casimir.material_geometry_qualification_compatibility import (
    write_legacy_compatibility,
)
from lno327.casimir.material_geometry_qualification_diagnostics import (
    diagnose_unresolved_shard,
)
from lno327.casimir.material_geometry_qualification_execution import (
    execute_campaign_geometry,
    execute_legacy_shard,
    freeze_plan,
    load_campaign,
    populate_shard,
    verify_campaign,
    write_preflight,
)
from lno327.casimir.material_observable_impact_calibration import (
    write_observable_impact_calibration,
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
    for action in (
        "plan",
        "preflight",
        "populate",
        "diagnose",
        "impact",
        "geometry",
        "legacy",
        "verify",
    ):
        command = subparsers.add_parser(action)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if action not in {"plan", "impact"}:
            command.add_argument(
                "--cache-root",
                type=Path,
                default=DEFAULT_CACHE_ROOT,
            )
        if action in {"populate", "diagnose", "legacy"}:
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--shard-count", type=int, default=1)
        if action == "diagnose":
            command.add_argument(
                "--n-candidates",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "diagnostic-only N ladder; must start at the final base N "
                    "and include at least two larger even levels"
                ),
            )
        if action == "impact":
            command.add_argument(
                "--diagnostic-source-dir",
                type=Path,
                required=True,
                help=(
                    "directory containing one complete unresolved-diagnostic "
                    "ladder as shard_*.json"
                ),
            )
            command.add_argument(
                "--pairing-name",
                default="dwave",
                help="pairing represented by the diagnostic source",
            )
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
        summary = dict(payload["summary"])
        if args.require_complete:
            compatibility = write_legacy_compatibility(
                campaign,
                output_dir=args.output_dir,
                cache_root=args.cache_root,
                require_ready=True,
            )
            summary["legacy_qualification_ready"] = compatibility["summary"][
                "qualification_ready"
            ]
            summary["legacy_incompatible_pair_count"] = compatibility["summary"][
                "incompatible_pair_count"
            ]
        _print(summary)
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

    if args.action == "diagnose":
        payload = diagnose_unresolved_shard(
            campaign,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            n_candidates_override=(
                None
                if args.n_candidates is None
                else tuple(int(value) for value in args.n_candidates)
            ),
        )
        _print(
            {
                "selected_missing_group_count": payload[
                    "selected_missing_group_count"
                ],
                "total_missing_group_count": payload["total_missing_group_count"],
                "base_n_candidates": payload["base_n_candidates"],
                "diagnostic_n_candidates": payload["diagnostic_n_candidates"],
                "diagnostic_ladder_tag": payload["diagnostic_ladder_tag"],
                "unresolved_frequency_count": payload[
                    "unresolved_frequency_count"
                ],
                "established_on_diagnostic_replay_count": payload[
                    "established_on_diagnostic_replay_count"
                ],
                "error_count": payload["error_count"],
                "diagnostic_completed": payload["diagnostic_completed"],
            }
        )
        if not payload["diagnostic_completed"]:
            raise SystemExit("one or more unresolved diagnostics failed to execute")
        return

    if args.action == "impact":
        payload = write_observable_impact_calibration(
            campaign,
            output_dir=args.output_dir,
            diagnostic_source_dir=args.diagnostic_source_dir,
            pairing_name=str(args.pairing_name),
        )
        _print(
            {
                **payload["summary"],
                "diagnostic_ladder_tag": payload["source_diagnostics"][
                    "diagnostic_ladder_tag"
                ],
                "pairing_name": payload["pairing_name"],
                "observable_error_budget_calibrated": payload["contract"][
                    "observable_error_budget_calibrated"
                ],
                "diagnostic_only": payload["diagnostic_only"],
            }
        )
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
        write_legacy_compatibility(
            campaign,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            require_ready=True,
        )
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

    write_legacy_compatibility(
        campaign,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        require_ready=True,
    )
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
