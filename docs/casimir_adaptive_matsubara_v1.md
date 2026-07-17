# Casimir adaptive Matsubara cutoff and tail v1

## Status

This branch adds a production-owned but diagnostic-only Matsubara cutoff controller on
top of the adaptive outer-Q cutoff/tail controller.

The public entry point is:

```python
from lno327.casimir import (
    AdaptiveMatsubaraCasimirConfig,
    run_adaptive_matsubara_casimir,
)

result = run_adaptive_matsubara_casimir(AdaptiveMatsubaraCasimirConfig())
```

A successful result is tail bounded in both outer-Q and Matsubara directions:

```text
status = adaptive_tail_bounded
outer_cutoff_adaptive = true
outer_tail_estimated = true
matsubara_cutoff_adaptive = true
matsubara_tail_estimated = true
production_casimir_allowed = false
```

The result remains unauthorized for production use until the chosen high-frequency
envelope and default numerical ladders are physically qualified for the intended model
and parameter region.

## Frozen boundaries

This controller does not modify:

- `run_casimir`;
- the radial panel estimator;
- the global angular-order estimator or offset audit;
- the joint radial-angular direction selector;
- the outer-Q cutoff and shell-envelope definitions;
- microscopic response, transverse-N ladders, shifts or physical gates;
- temperature, separation, plate angles, pairing definitions or material parameters;
- torque or pressure differentiation.

Every included Matsubara term must first obtain a complete outer-Q cutoff and tail bound.
An unresolved microscopic point or outer-Q result stops the frequency ladder immediately.

## Cumulative Matsubara ladder

A cutoff value `N` means the complete contiguous set:

```text
n = 0, 1, ..., N
```

The default maxima are:

```text
N = 1 -> 3 -> 7 -> 15 -> 31
```

Sparse evaluation such as only `n = 0, 1, 3, 7` is not allowed.  The high-frequency
tail window therefore always contains consecutive Matsubara terms.

The zero-frequency prime weight remains owned by the established outer quadrature.  The
stored `contributions_J_m2` are already prime weighted and include `k_B T`.

## Frequency-extendable certified-point cache

The v1 point provider cache includes the requested Matsubara set in its fingerprint and
is unchanged.

The new `FrequencyExtendableCertifiedOuterQProvider` uses a separate v2 cache schema.
Its entry key is:

```text
(pairing, n, qx.hex(), qy.hex())
```

The requested Matsubara set is removed from the v2 policy fingerprint, but all physical
inputs remain fingerprinted, including:

- pairing set;
- temperature and gap parameters;
- plate angles and separation;
- transverse-N candidates;
- shifts and scheduling controls;
- Ward, conditioning, static and logdet gates.

Changing any of those inputs rejects the cache.  Growing the Matsubara set under the
same fingerprint is allowed.  When a cumulative set grows, only the missing indices are
submitted to the production certifier.  Previously certified `(pairing, n, q)` entries
are not recomputed.

The v1 and v2 cache schemas are deliberately incompatible, preventing accidental reuse
of a fixed-frequency cache as a frequency-extendable cache.

## Unified total free-energy budget

For each pairing, let the finite Matsubara partial sum be:

```text
S_N = sum_{n=0}^N F_n
```

and let `E_outer,n` be the complete outer-Q error bound for term `n`, including finite
radial/angular error, offset audit and omitted outer-Q tail.

The total tolerance is:

```text
T_total = max(total_free_energy_atol,
              total_free_energy_rtol * abs(S_N))
```

It is split into:

```text
T_finite = finite_matsubara_fraction * T_total
T_tail   = matsubara_tail_fraction * T_total
```

with positive fractions summing to one.

The accumulated finite-frequency error is bounded by:

```text
E_finite = sum_{n=0}^N E_outer,n
```

The inner outer-Q controller receives a conservative fixed per-term share based on the
largest configured Matsubara cutoff.  Final acceptance still requires the actual sum
`E_finite` to fit the finite-Matsubara allocation; per-term success alone is not enough.

## High-frequency envelope

For every included term and every pairing, define:

```text
A_n = abs(F_n) + E_outer,n
```

This absolute envelope prevents sign cancellation between Matsubara terms from hiding a
large high-frequency contribution.

For the final consecutive high-frequency window:

```text
r_n = A_n / A_{n-1}
```

with `0/0` treated as zero and a positive value following zero treated as unresolved.
All observed ratios must satisfy:

```text
r_n <= tail_ratio_max < 1
```

The omitted high-frequency tail is then bounded conservatively using the configured
ratio cap rather than the smaller observed ratio:

```text
E_Matsubara_tail <= A_N * tail_ratio_max / (1 - tail_ratio_max)
```

Acceptance requires, separately for every pairing:

```text
E_finite <= T_finite
E_Matsubara_tail <= T_tail
E_finite + E_Matsubara_tail <= T_total
```

The total partial sum is never used to substitute for the absolute term envelope.

## Result and audit surface

The result records:

- every attempted Matsubara cutoff;
- the complete contiguous index set at each cutoff;
- the selected outer-Q cutoff for each frequency run;
- every prime-weighted Matsubara contribution;
- every per-term outer-Q error bound;
- absolute high-frequency term envelopes;
- observed tail ratios and their maximum;
- finite-frequency, tail and total error budgets;
- the selected final Matsubara cutoff;
- q-coordinate and `(n,q)` point-cache statistics;
- the explicit termination reason.

A successful pairing result has:

```text
status = integrated_with_outer_and_matsubara_tail_bounds
```

and includes a finite partial-sum center with an absolute total error bound.  No signed
value is assigned to the omitted tail.

## Fail-closed termination reasons

Important unresolved outcomes include:

```text
outer_tail_run_unresolved: ...
matsubara_microscopic_point_entry_budget_exhausted
matsubara_tail_window_not_established
matsubara_tail_decay_ratio_not_established
finite_matsubara_outer_budget_not_met
matsubara_tail_budget_not_met
total_free_energy_budget_not_met
matsubara_cutoff_ladder_exhausted
point_provider_failure: ...
matsubara_result_contract_failure: ...
```

Missing, malformed, uncertified or non-finite terms are never treated as zero.  Failure
to establish the geometric envelope at the maximum cutoff remains unresolved.
