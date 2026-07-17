"""Temporary import bridge after removal of the rejected GK21 driver.

The adaptive GK21 algorithm no longer exists.  The active panel controller still
imports its historically colocated workspace from this path; that import will be
updated when the split-history panel module is consolidated.
"""

from validation.lib.commensurate_orbit_workspace import (
    CompleteOrbitAggregateWorkspace,
    TransverseEvaluationBudgetExceeded,
)

__all__ = [
    "CompleteOrbitAggregateWorkspace",
    "TransverseEvaluationBudgetExceeded",
]
