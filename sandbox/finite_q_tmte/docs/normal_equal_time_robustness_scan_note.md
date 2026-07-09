# Normal equal-time robustness scan note

This diagnostic scans the robustness of the normal finite-q equal-time / translation RHS before freezing an analytic derivation.

Script:

```text
sandbox/finite_q_tmte/scripts/debug_normal_equal_time_robustness_scan.py
```

It repeatedly calls:

```text
sandbox/finite_q_tmte/scripts/debug_normal_equal_time_ward_audit.py
```

across selected Matsubara indices, q values, nk values, and shifted meshes.

## Purpose

The single-point equal-time audit found that the missing term in the normal Ward identity is exactly explained by `minus_translation_forward` with unit coefficient and machine precision mismatch.

This scan asks whether that result is stable over a small parameter window before writing a frozen analytic derivation.

## Pass criterion

For each row, the scan checks whether the top-ranked candidate is one of:

```text
minus_translation_forward
minus_translation_direct
```

and whether:

```text
diff/missing <= diff_tol
fit_res/missing <= fit_res_tol
|fit_alpha - 1| <= alpha_tol
```

The default tolerances are diagnostic and may be tightened or relaxed after observing realistic quadrature behavior.

## Interpretation

If all rows pass, the normal finite-q translation/equal-time RHS is numerically robust over the scan window.

If some rows fail, inspect the failed rows before writing the analytic derivation. Failure can mean either a real convention issue or simply insufficient quadrature for that parameter point.

This scan is not a production fix and does not make any Casimir input valid.
