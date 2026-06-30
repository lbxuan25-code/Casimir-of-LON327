# Best-effort finite-q Casimir plumbing route

## Purpose

This document records a temporary route for running the downstream finite-q response-to-reflection-to-Casimir plumbing before the superconducting finite-q Ward residual is fully reduced.

The route is intentionally **not** a production physics path.  It is a best-effort diagnostic candidate for exercising the full software chain while preserving the current Ward/gauge validation boundary.

## Current decision

For the next plumbing step, use the highest-refinement Ward-oriented integration strategy already present in the repository as the numerical default for finite-q diagnostic runs:

```text
coarse_grid = 32
adaptive_level = 5
gauss_order = 5
fermi_window_eV = 0.12
eta_eV = 1e-10
```

This is the upper end of the targeted Ward-refinement strategy used by `validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py`, rather than the small uniform `nk=3` smoke grid used by the active BdG finite-q status command.

## Response convention to carry downstream

Use the current finite-q BdG engine with the same convention stack used in the active Ward diagnostics:

```text
phase_vertex = bond_endpoint_gauge
current_vertex = peierls
collective_mode = amplitude_phase
collective_counterterm = goldstone_gap_equation
include_phase_phase_direct = True
selected_response = amplitude_phase_schur
```

The selected response is a **diagnostic candidate response**.  It should carry its Ward residual metadata into every downstream artifact.

## Required implementation discipline

1. Build a q-specific adaptive Brillouin-zone quadrature grid for each external q.  Do not reuse a single adaptive grid for all q values unless it is explicitly audited.
2. Evaluate `finite_q_bdg_response_from_ansatz(...)` on that q-specific grid.
3. Use `amplitude_phase_schur` as the candidate gauge-restored matrix for plumbing only.
4. Convert the spatial block to model sheet conductivity using the existing bilayer convention.
5. Apply the existing SI sheet and reflection-dimensionless conversion chain exactly once.
6. Feed the converted tensor into the reflection adapter.
7. Run only small-grid or explicitly labeled best-effort Casimir integration until Ward closure is solved.
8. Preserve Ward residuals, q-grid coverage, unit-conversion status, `n=0` policy, and `valid_for_casimir_input=False` in all outputs.

## Boundary

All artifacts produced by this route must keep:

```text
diagnostic_only = True
best_effort_plumbing = True
ward_identity_closed = False
valid_for_casimir_input = False
not_final_casimir_conclusion = True
```

This route may be used to discover downstream interface, unit, reflection, grid, cache, and integration problems.  It must not be used to claim formal conductivity, final Casimir energy, force, or torque.

## Later return point

After the full plumbing path runs end-to-end, return to the upstream finite-q Ward blocker and reduce the superconducting finite-q residual without response-level fitting, residual projection, or LSQ repair.
