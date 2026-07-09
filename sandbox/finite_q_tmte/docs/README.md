# finite_q_tmte documentation map

This directory contains both current sandbox documentation and historical audit notes.  Many files were written during exploratory Ward-identity and minimal-Casimir development, so this README is the entry point that defines document precedence and status.

Status boundary:

```text
sandbox documentation map
non-destructive cleanup pass
old notes retained for provenance
current entry points explicitly marked
valid_for_casimir_input: False
```

---

## 1. How to read this directory

Use this precedence order when documents appear to conflict:

```text
1. This README
2. Current workflow / checkpoint / remaining-work docs
3. Final Ward handoff and closure-status docs
4. Current minimal-Casimir diagnostic docs
5. Low-level audit notes
6. Legacy planning / scratch notes
```

Do not treat older audit notes as current project status unless they are referenced by a current entry-point document.

Do not infer Casimir readiness from any file in this directory.  Current sandbox status remains:

```text
valid_for_casimir_input: False
```

---

## 2. Current entry points

Read these first:

| File | Status | Purpose |
| --- | --- | --- |
| `minimal_casimir_diagnostic_workflow.md` | active workflow | Unified diagnostic CLI, current scan order, no-shift default, health-report entry point. |
| `finite_q_remaining_work_items.md` | active remaining-work list | Non-Ward blockers before main-flow/Casimir readiness. |
| `minimal_casimir_n_budget_checkpoint_note.md` | active checkpoint | Positive-n budget checkpoint, midpoint validation, high-n tail, nk=13 artifact classification. |
| `minimal_casimir_health_report_note.md` | active tooling note | Offline artifact credibility gate for future result checking. |
| `finite_q_ward_final_handoff.md` | final Ward handoff | Final sandbox Ward/RHS handoff; use this instead of intermediate Ward audit notes for current Ward status. |
| `finite_q_ward_closure_status.md` | current Ward status | Compact Ward closure status boundary. |

---

## 3. Current minimal-Casimir diagnostic notes

These describe currently available sandbox diagnostic components.  They are implementation notes, not production Casimir documentation.

| File | Status | Purpose |
| --- | --- | --- |
| `minimal_casimir_path_note.md` | component note | Minimal q=(q,0) single-point path. |
| `minimal_casimir_qvec_path_note.md` | component note | Arbitrary q-vector path. |
| `minimal_casimir_theta_path_note.md` | component note | Two-plate theta/path diagnostic. |
| `minimal_casimir_theta_scan_note.md` | component note | Theta scan diagnostic. |
| `minimal_casimir_phi_scan_note.md` | component note | Phi scan diagnostic. |
| `minimal_casimir_q_scan_note.md` | component note | q scan diagnostic and q-weighted radial budget. |
| `minimal_casimir_shift_scan_note.md` | component note | Shift scan / R-norm pathology diagnostic. |
| `minimal_casimir_n_scan_note.md` | component note | Positive-Matsubara n scan diagnostic. |
| `minimal_casimir_n_tail_fit_note.md` | component note | Offline high-n tail fit. |
| `minimal_casimir_n_budget_note.md` | component note | Offline positive-n budget aggregation. |
| `minimal_casimir_health_report_note.md` | active tooling note | Offline artifact health report. |

Read `minimal_casimir_n_budget_checkpoint_note.md` for the current positive-n checkpoint instead of piecing together state from the individual component notes.

---

## 4. Current Ward/RHS status docs

Use these for current Ward interpretation:

| File | Status | Purpose |
| --- | --- | --- |
| `finite_q_ward_final_handoff.md` | canonical Ward handoff | Final sandbox finite-q Ward/RHS result. |
| `finite_q_ward_closure_status.md` | canonical status | Compact Ward closure status. |
| `finite_q_bdg_schur_ward_derivation.md` | derivation support | Schur/BdG Ward derivation reference. |
| `finite_q_matsubara_ward_convention_note.md` | convention support | Matsubara Ward convention note. |
| `finite_q_matsubara_inverse_green_ward_block.md` | derivation block | Inverse-Green Ward block reference. |
| `rhs_aware_finite_q_validation_note.md` | validation support | RHS-aware finite-q validation status. |

Current interpretation:

```text
zero-RHS Ward at finite q is not the production criterion
finite-q RHS-aware / Schur-projected closure is the sandbox diagnostic interpretation
valid_for_casimir_input remains False
```

---

## 5. Historical audit notes

These are retained for provenance and debugging.  They should not be used as the first source of current status.

| File | Status | Notes |
| --- | --- | --- |
| `inverse_green_ward_audit_note.md` | historical audit | Earlier inverse-Green Ward audit. |
| `normal_equal_time_ward_audit_note.md` | historical audit | Normal equal-time Ward audit. |
| `normal_equal_time_robustness_scan_note.md` | historical audit | Normal equal-time robustness scan. |
| `normal_contact_ward_control_note.md` | historical audit | Normal contact/Ward control. |
| `normal_response_convention_audit_note.md` | historical audit | Normal response convention audit. |
| `contact_formula_audit_note.md` | historical audit | Contact formula audit. |
| `pairing_contact_missing_audit_note.md` | historical audit | Pairing contact missing-term audit. |
| `primitive_response_ward_audit_note.md` | historical audit | Primitive response Ward audit. |
| `primitive_response_ward_decomposition_note.md` | historical audit | Primitive response decomposition. |
| `primitive_response_closure_suite_note.md` | historical audit | Primitive closure suite. |
| `primitive_em_translation_rhs_audit_note.md` | historical audit | Primitive EM translation/RHS audit. |
| `primitive_extended_translation_collective_audit_note.md` | historical audit | Extended translation/collective audit. |
| `primitive_extended_translation_collective_robustness_scan_note.md` | historical audit | Extended translation/collective robustness scan. |
| `schur_effective_translation_rhs_audit_note.md` | historical audit | Schur effective translation RHS audit. |

If one of these notes conflicts with `finite_q_ward_final_handoff.md`, the final handoff wins.

---

## 6. Legacy planning and scratch notes

These are useful for context, but should not guide current main-flow integration without review.

| File | Status | Notes |
| --- | --- | --- |
| `derivation.md` | legacy planning / derivation | Early derivation context. |
| `implementation_plan.md` | legacy planning | Early implementation plan. |
| `migration_plan.md` | legacy planning | Early migration plan. |
| `validation_protocol.md` | legacy planning | Early validation protocol. |
| `open_questions.md` | legacy planning | Older open questions; use `finite_q_remaining_work_items.md` as current remaining-work list. |
| `finite_q_matsubara_ward_derivation_scratch.md` | scratch | Scratch derivation; not current status. |

---

## 7. Compatibility / maintenance notes

| File | Status | Purpose |
| --- | --- | --- |
| `q_scan_numpy2_compat_note.md` | compatibility note | NumPy 2 q-scan trapezoid compatibility fix. |

---

## 8. Current unresolved main-flow blockers

The short version is:

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

Engineering optimization is deliberately not first priority.  Caching, parallelism, vectorization, output-size control, and large-nk runtime control should come after the sandbox flow and main-flow contract are fixed.

---

## 9. Documentation cleanup policy

Non-destructive cleanup rules:

```text
1. Do not delete or rewrite historical audit notes just to make the directory smaller.
2. Add or update entry-point docs first.
3. Every new doc should declare one of: active workflow, active checkpoint, component note, historical audit, legacy planning, compatibility note.
4. Every current-status doc must explicitly state valid_for_casimir_input.
5. If a historical note is superseded, mark that in this README rather than silently editing the old note.
6. Main-flow integration should read only active/canonical docs unless debugging provenance.
```

Future cleanup can move historical files into an archive directory, but that should be a separate mechanical change after this README has been used for at least one integration pass.
