# Minimal Casimir diagnostic workflow

This document defines the current lightweight diagnostic workflow for the sandbox finite-q TMTE Casimir path.

The goal is to reduce script sprawl without promoting the sandbox diagnostics to a production Casimir pipeline.

Status:

```text
diagnostic-only workflow
unified CLI entry point
old debug scripts retained for reproducibility
default shift policy: no-shift
positive-n budget checkpoint documented
health-report credibility gate added
not a full q/phi/n Casimir integral
not a torque calculation
valid_for_casimir_input: False
```

Current checkpoint note:

```text
sandbox/finite_q_tmte/docs/minimal_casimir_n_budget_checkpoint_note.md
```

Health-report note:

```text
sandbox/finite_q_tmte/docs/minimal_casimir_health_report_note.md
```

Checkpoint summary:

```text
positive-n diagnostic budget: 0.007431760613385256 to 0.007449706575429743
midpoint validation: globally passed, about 1.2% budget shift
n>500 tail: controlled, about 4.2e-05 to 6.0e-05
local nk=13 pathology: n=12, q=0.04, removed by nk=17
valid_for_casimir_input: False
```

---

## 1. Unified entry point

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

The old individual debug scripts are still present and can reproduce previous runs.  New exploratory commands should prefer the unified CLI.

---

## 2. Shift policy

The unified CLI defaults to:

```text
--shift-fractions 0.0
```

Reason:

```text
single-shift scans showed reflection-norm pathologies for shifted meshes
no-shift has been cleaner in the current n=1 diagnostics
shifted5 should be treated as a stress test, not as the default high-accuracy setting
```

Recommended use:

```text
primary diagnostic scans:
  no-shift

quick auxiliary check:
  shifted2, e.g. 0.0 0.5

stress/pathology check:
  shift-scan over 0.0 0.2 0.4 0.6 0.8
```

---

## 3. Current diagnostic order

Suggested research flow:

```text
1. theta-scan
2. phi-scan
3. q-scan
4. shift-scan / R_norm guard when suspicious
5. n-scan
6. n-tail-fit, offline CSV-only
7. n-budget, offline CSV-only
8. n-budget checkpoint note
9. health-report, offline artifact credibility gate
10. q-phi-n diagnostic sum/checkpoint, not implemented yet
```

Important distinction:

```text
health-report is intended for future formal result credibility checks.
It should read outputs after a run and classify their trustworthiness.
It should not be used as an excuse to continue bulk sandbox scanning.
```

Do not skip the status flags in the output JSON.  All current outputs must retain:

```text
valid_for_casimir_input: False
```

---

## 4. Example commands

### 4.1 theta scan

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  theta-scan \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --phi-deg 30 \
  --theta-values 0 15 30 45 60 75 90 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/diag_theta_scan_dwave_n1_q002_phi30_theta0_90_noshift
```

### 4.2 phi scan

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  phi-scan \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --phi-values 0 30 60 90 120 150 180 210 240 270 300 330 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/diag_phi_scan_dwave_n1_q002_theta45_phi12_noshift
```

### 4.3 q scan

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  q-scan \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q-values 0.00125 0.0025 0.005 0.0075 0.01 0.015 0.02 0.04 0.08 \
  --phi-values 0 30 60 90 120 150 180 210 240 270 300 330 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/diag_q_scan_dwave_n1_theta45_phi12_noshift_refined
```

### 4.4 shift scan

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  shift-scan \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.04 \
  --phi-values 0 15 30 45 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-values 0.0 0.2 0.4 0.6 0.8 \
  --r-norm-warning-threshold 2.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/diag_shift_scan_dwave_n1_q004_theta45_phi4_shifts002468
```

### 4.5 n scan

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  n-scan \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-indices 100 150 200 300 500 \
  --temperature-K 10.0 \
  --q-values 0.00125 0.0025 0.005 0.0075 0.01 0.015 0.02 0.04 0.08 \
  --phi-values 0 30 60 90 120 150 180 210 240 270 300 330 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n100_500_sparse_theta45_phi12_qrefined_noshift
```

### 4.6 n tail fit

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  n-tail-fit \
  --input-csv sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n100_500_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
  --models power_n power_xi \
  --fit-min-n 100 \
  --tail-start-n-exclusive 500 \
  --output-dir sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n100_500_theta45_phi12_qrefined_noshift
```

### 4.7 n budget

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  n-budget \
  --input-csv \
    sandbox/finite_q_tmte/outputs/diag_n_scan_dwave_n1_5_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
    sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n10_100_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
    sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n100_500_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
  --tail-fit-json \
    sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n100_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json \
    sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n150_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json \
    sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n200_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json \
  --output-dir sandbox/finite_q_tmte/outputs/diag_n_budget_dwave_theta45_phi12_qrefined_noshift
```

### 4.8 health report

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  health-report \
  --input-json \
    sandbox/finite_q_tmte/outputs/diag_n_budget_with_midpoints_dwave_theta45_phi12_qrefined_noshift/minimal_casimir_n_budget.json \
    sandbox/finite_q_tmte/outputs/diag_theta_probe_compare_dwave_theta0_45_90_phi12_qrefined_noshift/theta_probe_compare_summary.json \
  --input-csv \
    sandbox/finite_q_tmte/outputs/diag_phi_scan_dwave_n12_q004_theta45_phi24_noshift/minimal_casimir_phi_scan.csv \
    sandbox/finite_q_tmte/outputs/diag_phi_scan_dwave_n12_q004_theta45_phi24_noshift_nk17/minimal_casimir_phi_scan.csv \
  --output-dir sandbox/finite_q_tmte/outputs/diag_health_report_current_checkpoint
```

---

## 5. What remains deliberately unintegrated

The following remain separate for now:

```text
minimal single-point path scripts
q-vector single-point scripts
theta-path single-point script
legacy debug scripts used in earlier handoff notes
```

They are useful for reproducing older checkpoints and should not be removed during this lightweight consolidation.

---

## 6. Next planned diagnostic tooling

The next diagnostic-tooling step should be a reviewed override manifest for health-report findings:

```text
unresolved needs_review
reviewed finite-grid artifact
hard fail
```

After that, a manifest-based controlled runner can call scans and then immediately run health-report, while still keeping:

```text
n >= 1 partial sums only until n=0 and tail policies are defined
no production energy
valid_for_casimir_input: False
```
