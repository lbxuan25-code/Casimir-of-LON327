# finite-q transverse adaptive GK21 diagnostic record

## Status

The SciPy `quad_vec(..., quadrature="gk21")` implementation is **rejected as a
production transverse integrator**. It remains in the repository only as a
reproducible diagnostic record while the deterministic panel-adaptive replacement
is validated.

```text
diagnostic_only = True
production_candidate_active = False
production_reference_established = False
valid_for_casimir_input = False
```

It must not be restored as a runtime fallback and must not be selected for special
q directions.

## Invariants that remain valid

The rejected driver still established reusable physical and software contracts:

- every transverse node evaluates a complete exact commensurate q orbit, including
  complementary origins when required;
- only primitive electromagnetic, collective, mixed, and Ward-RHS blocks are
  integrated;
- nearest-neighbour bond metric, amplitude/phase Schur, sheet, reflection, and
  logdet operations occur only after a complete global primitive integral;
- all requested Matsubara frequencies share the same transverse nodes;
- Ward RHS shares the quadrature but remains a monitor rather than a refinement
  driver;
- the optimized batched microscopic q-workspace has no q-direction dispatch and no
  scalar runtime fallback.

These invariants are retained by the replacement integrator.

## Rejection evidence

The optimized real-`nk=1256` callback reduced complete-orbit wall time by roughly
6.2--6.9 times, so the rejection is not caused by the old scalar microscopic hot
loop.

The fixed 256-unique-node acceptance failed for all three decisive cases before the
primary integral completed:

```text
reference (6,4):     transverse_evaluation_budget_exceeded
diagonal_mid (6,6):  transverse_evaluation_budget_exceeded
diagonal_min (1,1):  transverse_evaluation_budget_exceeded
```

A larger bounded diagnostic allowed `quad_vec` to return complete finite estimates:

```text
case                    unique t    primary error ratio
reference (6,4)              651          10.733645
diagonal_mid (6,6)           483         132.922892
diagonal_min (1,1)           567         135.686475
```

The tightened audit did not start in any case. Ward validation passed for Matsubara
indices `1,2,4,8` on every available primary estimate, so the blocker is transverse
quadrature rather than gauge closure.

Root-panel frozen scaling underestimated the low-frequency electromagnetic groups
by factors up to about 5.8 on the diagonal cases, but only about 1.25 on the
reference case. Correcting that scaling alone cannot explain error ratios of
approximately 133, and it cannot explain the reference failure.

The reported error was strongly localized in a small number of symmetric
subintervals. This supports local panel refinement, but `quad_vec` exposes only one
black-box vector error and does not provide deterministic operation-boundary budget
control. In practice its internal batched subdivision also exceeded the intuitive
panel `limit` count.

## Why the driver is not repaired by parameter tuning

The following responses are forbidden:

- increasing the production cap until the current driver happens to pass;
- weakening `epsabs`, `epsrel`, or the physical audit gates;
- switching `norm=max` to another norm only for difficult q points;
- adding a periodic, fixed-Gauss, or direction-specific runtime fallback;
- restoring pointwise metric or Schur operations.

The failure requires a common redesign of the transverse controller rather than a
q-specific parameter sweep.

## Replacement

The active replacement candidate is documented in:

```text
docs/finite_q_transverse_panel_adaptive_contract.md
```

It uses an explicit deterministic panel queue, nested Clenshaw-Curtis 9/17/33
rules, physical-group error accounting, shared primary/audit state, and budget
checks before complete p- or h-refinement operations.
