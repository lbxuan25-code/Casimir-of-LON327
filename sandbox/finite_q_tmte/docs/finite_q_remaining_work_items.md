# Finite-q remaining work items

This short note records the non-Ward issues that remain after the sandbox finite-q Ward handoff.

Status boundary:

```text
Ward closure: complete at sandbox diagnostic level
Main validation flow: unchanged
valid_for_casimir_input: False
```

Detailed Ward handoff:

```text
sandbox/finite_q_tmte/docs/finite_q_ward_final_handoff.md
```

---

## 1. Limit and special-point problems

```text
exact q = 0 definition
q -> 0 continuity from finite q
order of limits: q -> 0 vs xi -> 0
n = 0 static Matsubara mode
large-n / high-frequency tail
```

---

## 2. Numerical convergence problems

```text
internal BZ nk convergence
shifted-mesh convergence and practical shift strategy
external q-grid convergence
Matsubara cutoff convergence
K_etaeta conditioning and near-zero collective modes
```

---

## 3. Response-to-Casimir mapping problems

```text
which K_eff components or projections are consumed by the future Casimir path
primitive (A0, L, T) basis -> physical EM response
physical EM response -> TE/TM or reflection basis
anisotropy, q-direction, and rotation/torque geometry
units, prefactors, and SI/eV/lattice normalization
```

---

## 4. Physical sanity checks

```text
normal-state limit
Delta0 -> 0 limit
pairing-symmetry comparison sanity checks
static stiffness / susceptibility checks
Hermiticity, reciprocity, and symmetry checks
```

---

## 5. Production and migration problems

```text
separate diagnostic-only fields from future production fields
keep contact_scale and scalar alpha projections diagnostic-only
reproducible output schema: model, pairing, q, n, nk, shift, eta, commit
old-main-flow regression / migration strategy
final valid_for_casimir_input criteria and error budget
```

---

## 6. Engineering optimization problems

```text
caching and reuse of expensive blocks
vectorization and parallel scan execution
large nk memory/runtime control
shifted5 cost control
output-size control for large scans
CI coverage for sandbox diagnostics
```

---

## Recommended next focus

The next stage should start from physical response convergence and Casimir-consumed response definition, not from additional Ward residual fitting.
