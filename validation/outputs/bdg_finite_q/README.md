# BdG finite-q validation status contract

This directory keeps a lightweight status contract for the current finite-q
BdG validation scripts. It does not contain raw response arrays, plots, caches,
or Casimir-ready data.

- Primary validation model: `symmetry_bdg_2band`
- Secondary transfer model: `lno327_four_orbital`
- Current scripts:
  - `validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py`
  - `validation/scripts/bdg_finite_q/finite_q_ward_scan.py`
- Workspace evaluation: enabled
- `valid_for_casimir_input`: `False`

The status JSON records diagnostic outcomes only. A completed diagnostic run is
not a Casimir readiness certificate.
