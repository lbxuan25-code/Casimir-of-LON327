"""Compatibility facade for fixed microscopic outer-Q preflight helpers.

Production planning, certified-point reduction, and ladder comparison live in
:mod:`lno327.casimir.fixed_outer_q`.  The legacy module remains available only for
private validation-test helpers while the migration is staged.
"""
from __future__ import annotations

from validation.lib import microscopic_outer_q_preflight_legacy as _legacy

# Preserve private validation helper imports during the staged migration.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_legacy, _name))

from lno327.casimir.fixed_outer_q import (  # noqa: E402
    OuterQGridPlan,
    OuterQGridSpec,
    OuterQNodeManifest,
    absolute_then_relative,
    aggregate_certified_outer_q,
    build_staged_grid_plan,
    build_union_node_manifest,
    compare_ladders,
)

__all__ = [
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "absolute_then_relative",
    "aggregate_certified_outer_q",
    "build_staged_grid_plan",
    "build_union_node_manifest",
    "compare_ladders",
]
