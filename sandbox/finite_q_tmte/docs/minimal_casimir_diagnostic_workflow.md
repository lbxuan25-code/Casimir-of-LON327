# Minimal Casimir diagnostic workflow

This is the current workflow document for the sandbox finite-q TMTE minimal Casimir diagnostics.

Status boundary:

```text
diagnostic-only workflow
unified CLI entry point
primary scan mode: no-shift
positive-n budget checkpoint documented
health-report credibility gate added
not a production Casimir pipeline
not a Casimir energy
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Current authoritative docs

Only these docs should be treated as current sandbox status for main-flow integration work:

```text
finite_q_ward_final_handoff.md
finite_q_ward_closure_status.md
finite_q_bdg_schur_ward_derivation.md
finite_q_remaining_work_items.md
minimal_casimir_diagnostic_workflow.md
minimal_casimir_n_budget_checkpoint_note.md
minimal_casimir_health_report_note.md
```

Other old sandbox notes have been removed to avoid conflicting historical status.

---

## 2. Unified diagnostic CLI

Use:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py
```

Supported subcommands:

```text
theta-scan
phi-scan
q-scan
shift-scan
n-scan
n-tail-fit
n-budget
health-report
```

The CLI remains diagnostic-only. It must not be interpreted as a production Casimir runner.

---

## 3. Current numerical policy

```text
q=0:
  excluded from finite-q path

n=0:
  excluded; no static Matsubara policy yet

shift:
  default no-shift
  shifted meshes are stress/pathology probes, not default accuracy policy

nk:
  nk=13 is scouting-level
  suspicious local points require nk convergence probes, e.g. nk=17

logdet_abs:
  diagnostic-only quantity
  not a physical signed Casimir integrand

valid_for_casimir_input:
  always False in current sandbox outputs
```

---

## 4. Current positive-n checkpoint

The current positive-n diagnostic checkpoint is recorded in:

```text
minimal_casimir_n_budget_checkpoint_note.md
```

Summary:

```text
positive-n diagnostic budget: 0.007431760613385256 to 0.007449706575429743
midpoint validation: globally passed, about 1.2% budget shift
n>500 tail: controlled, about 4.2e-05 to 6.0e-05
local nk=13 pathology: n=12, q=0.04, removed by nk=17
valid_for_casimir_input: False
```

---

## 5. Health report

The current artifact credibility checker is documented in:

```text
minimal_casimir_health_report_note.md
```

Purpose:

```text
read existing JSON/CSV artifacts
classify numerical health as pass / needs_review / fail
flag R_norm, Rdiff, phi-range, kappa, and nonfinite issues
do not run BdG
do not schedule scans
do not define Casimir policy
```

This tool is intended for future formal data-production result checking, not for bulk sandbox scanning.

---

## 6. Remaining blockers before main-flow Casimir input

The current remaining-work list is:

```text
finite_q_remaining_work_items.md
```

Highest-priority blockers:

```text
Casimir-consumed response contract not fixed
physical EM -> TE/TM/reflection production mapping not fixed
signed/physical logdet policy not fixed
units / prefactors / q-phi measure not fixed
n=0 static Matsubara mode policy not fixed
q=0 / q->0 / xi->0 limit policy not fixed
health-report reviewed override mechanism not implemented
final valid_for_casimir_input criteria not defined
```

Engineering optimization is deliberately not first priority. Caching, parallel execution, vectorization, output-size control, and large-nk runtime control should come after the sandbox flow and main-flow contract are fixed.

---

## 7. Recommended next non-scanning work

Next work should focus on process clarity and main-flow readiness, not more data production:

```text
1. reviewed override manifest for health-report findings
2. Casimir-consumed response contract
3. signed/physical logdet policy
4. q=0/n=0/order-of-limits policy
5. valid_for_casimir_input criteria
```
