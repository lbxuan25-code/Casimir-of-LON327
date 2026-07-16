"""CPU-headroom facade for the transverse-point sweet-spot engine.

All numerical implementation remains in ``transverse_point_sweet_spot_engine_legacy``.
Automatic worker selection now reserves two affinity-visible logical CPUs by default
so the host remains responsive.  Set ``LNO327_CPU_RESERVE`` to another non-negative
integer, or pass an explicit positive ``--workers`` value, to override this policy.
"""
from __future__ import annotations

import os
from typing import Sequence

from lno327.workflows.cpu_parallel import affinity_cpu_count
from validation.lib import transverse_point_sweet_spot_engine_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_legacy, _name))


def _automatic_cpu_reserve() -> int:
    raw = os.environ.get("LNO327_CPU_RESERVE", "2")
    try:
        reserve = int(raw)
    except ValueError as exc:
        raise ValueError("LNO327_CPU_RESERVE must be a non-negative integer") from exc
    if reserve < 0:
        raise ValueError("LNO327_CPU_RESERVE must be a non-negative integer")
    return reserve


def _parse_args(argv: Sequence[str] | None):
    args = _legacy._parse_args(argv)
    requested = int(args.workers)
    available = int(affinity_cpu_count())
    reserve = _automatic_cpu_reserve()
    if requested == 0:
        effective_reserve = min(reserve, max(available - 1, 0))
        args.workers = max(available - effective_reserve, 1)
        args.worker_budget_source = "cpu_affinity_minus_reserved_headroom"
        args.requested_workers_before_reserve = 0
        args.affinity_cpu_count = available
        args.reserved_affinity_cpus = effective_reserve
    else:
        args.worker_budget_source = "explicit_workers"
        args.requested_workers_before_reserve = requested
        args.affinity_cpu_count = available
        args.reserved_affinity_cpus = max(available - min(requested, available), 0)
    return args
