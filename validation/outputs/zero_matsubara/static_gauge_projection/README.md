# Exact-static longitudinal gauge projection

This directory documents the validation contract for the optional exact
zero-Matsubara longitudinal projection.

## Scope

The policy is permitted only for:

- exact `xi_eV == 0`;
- nonzero in-plane `q_model`;
- the collective-corrected `EffectiveEMKernel`;
- normal Schur inversion (`inv`), not diagnostic pseudoinversion;
- a passed mixed absolute-relative Ward validation.

It is forbidden for positive Matsubara frequencies.

## Policies

```text
raw_fail_closed
project_after_validated_ward
```

`raw_fail_closed` preserves the original static-sheet validation behavior.

`project_after_validated_ward` first builds and validates the raw local
`(A0,L,T)` response.  The raw response must pass reality,
density-transverse-mixing, passivity, Ward, and Schur-condition gates, and its
aggregate longitudinal residual must satisfy

```text
raw_relative_longitudinal_gauge_residual <= 1e-5
```

by default.  Only then is

```text
P_static = diag(1, 0, 1)
K_projected = P_static @ K_raw @ P_static
```

applied.

The production longitudinal target remains `1e-7`; the `1e-5` value is only a
ceiling on how much raw pure-gauge quadrature leakage may be removed.  A raw
response above that ceiling still fails closed.

## Audit trail

The returned `StaticSheetResponse` contains the projected kernel.  Its metadata
retains:

- the raw and projected local kernels;
- the projection matrix;
- raw and projected longitudinal residuals;
- the relative projection correction norm;
- the raw ceiling and target tolerance;
- every projection prerequisite and its status.

The physical `(A0,T)` block is required to be preserved exactly.  Consequently
`chi_bar`, `Dbar_T`, and the current static reflection adapter are unchanged by
the projection.

## API

```python
from lno327.electrodynamics import (
    PROJECT_AFTER_VALIDATED_WARD,
    static_matsubara_kernel_to_sheet_response_with_policy,
)

response = static_matsubara_kernel_to_sheet_response_with_policy(
    kernel,
    ward_validation,
    longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
    projection_raw_longitudinal_ceiling=1e-5,
    longitudinal_tolerance=1e-7,
)
```

This policy reduces the need to choose very large `nk` solely to force an
analytically pure-gauge row and column below the final static tolerance.  It
does not remove the need to converge the retained density and transverse
physical channels over the Casimir `q`, angle, and Matsubara grids.
