#!/usr/bin/env python3
"""Retired legacy finite-q BdG Casimir pipeline entry point.

The previous implementation mixed several superseded contracts in one script:

* a zero-RHS Ward check on ``K_eff`` instead of the analytic RHS-aware identity;
* the deprecated four-orbital production route;
* positive-frequency extrapolation used as an ``n=0`` policy;
* TE/TM-amplitude trace-log plumbing instead of the common tangential-E LT path.

Keeping that script executable would allow physically invalid results to be
produced after the primitive-xy contract was introduced.  The implementation is
therefore intentionally retired.  Git history retains the old source for audit
purposes.  This path will become the thin CLI for the new two-band library
pipeline only after the static mode and quadrature contracts are complete.
"""

from __future__ import annotations

LEGACY_PIPELINE_RETIRED = True
RETIREMENT_REASON = (
    "scripts/casimir/finite_q_bdg_casimir_pipeline.py is retired: the former "
    "pipeline used an invalid zero-RHS finite-q Ward check and other superseded "
    "Casimir contracts. Use the typed primitive-xy response, RHS-aware Ward, "
    "sheet-response, LT-reflection, and signed-logdet library APIs while the "
    "new thin production CLI is being completed."
)


def main() -> None:
    raise SystemExit(RETIREMENT_REASON)


if __name__ == "__main__":
    main()
