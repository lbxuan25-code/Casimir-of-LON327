"""Compatibility facade for fixed nested outer-Q planning.

The implementation lives in :mod:`lno327.casimir.fixed_outer_q`.  Validation is a
consumer of the production numerical contract and must not own a second copy.
"""
from __future__ import annotations

from lno327.casimir.fixed_outer_q import (
    OuterQGridPlan,
    OuterQGridSpec,
    OuterQNodeManifest,
    build_staged_grid_plan,
    build_union_node_manifest,
)

__all__ = [
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "build_staged_grid_plan",
    "build_union_node_manifest",
]
