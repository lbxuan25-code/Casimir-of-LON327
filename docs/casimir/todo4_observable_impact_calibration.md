# TODO 4 observable-impact calibration

## Purpose

The unresolved d-wave response diagnostics show hard physical closure at every
sample but do not satisfy the provisional response-space N/shift tolerance.  This
stage measures how that numerical spread propagates through the exact TODO 4
reflection and passive trace-log geometry before a larger microscopic ladder is
approved.

This is calibration evidence only:

```text
diagnostic_only: true
valid_for_casimir_input: false
production_casimir_allowed: false
observable_error_budget_calibrated: false
```

It does not weaken response certification, create a certified response artifact,
or authorize use of unresolved responses in the production geometry executor.

## Input boundary

The action consumes one complete unresolved-diagnostic ladder directory:

```text
<old-output>/unresolved_diagnostics/N256-384-512/shard_*.json
```

The source may come from an earlier code commit.  The calibration records both the
old diagnostic source commit/plan and the current frozen plan.  Scientific
compatibility is checked from the campaign id, exact pairing, exact q/frequency
requirements, complete N ladder, complete shift sets, hard physical status, and
sheet validation.

The action does not open the certified response cache.

## Exact geometry propagated

For every d-wave direct representative point, Matsubara index, N and distance, the
stage reconstructs the recorded static or positive-frequency sheet response and
uses the exact TODO 4 geometry formulas to build reflections and signed passive
logdet values.

Each plate has three recorded shifts.  The calibration evaluates all independent
plate combinations:

```text
3 plate-1 shifts x 3 plate-2 shifts = 9 shift pairs
```

It records:

- actual geometry logdet;
- a parallel control with `theta_2 = theta_1` at the same q_lab;
- the finite-angle logdet contrast `actual - parallel_control`;
- product and round-trip eigenvalues;
- distance-dependent gap to the log branch point;
- response-space shift spread and local-logdet shift spread;
- adjacent-N local-logdet changes;
- the finite `n=0,1` Matsubara-weighted local partial sum.

The weighted partial sum includes the zero-mode prime weight:

```text
0.5 * logdet(n=0) + 1.0 * logdet(n=1)
```

It is still a local finite-frequency diagnostic.  No outer-q quadrature weight,
`k_B T` prefactor, Matsubara tail, free-energy total, pressure or torque is claimed.

## Command

A source change requires a fresh output directory and a newly frozen plan.  The old
diagnostic directory is passed explicitly rather than copied or promoted.

```bash
python -m validation diagnostic todo4-representative-qualification impact \
  --manifest "$MANIFEST" \
  --output-dir "$OUT" \
  --diagnostic-source-dir "$DIAG_SOURCE" \
  --pairing-name dwave
```

The result is written to:

```text
$OUT/observable_impact/N256-384-512/dwave.json
```

## Output interpretation

For each local point and N:

```text
maximum_plate_response_relative_spread
actual_logdet_spread.relative_spread_to_max_abs
relative_spread_transfer_ratio
maximum_actual_round_trip_eigenvalue
minimum_actual_gap_to_log_branch
```

The transfer ratio is descriptive, not a pass/fail gate.  It indicates whether the
local trace-log geometry suppresses or amplifies the observed response shift spread.

The angular-contrast spread is especially important for torque-like observables,
but it is not itself a torque because the current representative matrix contains
only one finite angle rather than a controlled angular derivative stencil.

## Safety contract

The artifact explicitly records:

```text
microscopic_integration_performed: false
response_certification_performed: false
certified_response_cache_read_attempted: false
certified_response_cache_write_attempted: false
diagnostic_response_promoted: false
observable_tolerance_applied: false
observable_error_budget_calibrated: false
```

Consequently, this stage cannot unblock strict preflight, geometry, legacy replay or
verification.  It only supplies evidence for choosing an economical response policy
and for designing the later observable-level error budget.
