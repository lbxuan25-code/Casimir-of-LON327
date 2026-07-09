# Primitive extended translation collective robustness scan note

This diagnostic scans the robustness of the primitive extended BdG translation/contact RHS.

Script:

```text
sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_robustness_scan.py
```

It repeatedly calls:

```text
sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_audit.py
```

across selected pairings, Matsubara indices, q values, nk values, and shifted meshes.

## Identity being tested

The single-point audit found that after adding the mixed collective Ward term, the primitive residual is explained by:

```text
u K_SS + W K_etaS = translation_forward + qM_mid
```

or equivalently:

```text
missing_to_close = - (translation_forward + qM_mid)
```

The top-ranked candidate should therefore be:

```text
minus_translation_plus_qM
```

## Pass criterion

For both left and right sides, each scan row must satisfy:

```text
top candidate = minus_translation_plus_qM
diff/missing <= diff_tol
fit_res/missing <= fit_res_tol
|fit_alpha - 1| <= alpha_tol
```

Default tolerances are `1e-9` to allow modest quadrature noise.

## Interpretation

If all rows pass, the primitive extended BdG translation/contact RHS is numerically robust over the scanned window.

If some rows fail, inspect the failed rows before writing a frozen analytic derivation or moving to Schur-level Ward closure.

This scan is diagnostic-only and does not validate Casimir input.
