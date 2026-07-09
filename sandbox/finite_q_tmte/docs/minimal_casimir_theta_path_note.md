# Minimal sandbox Casimir theta path

This note documents the first diagnostic-only plate-rotation step built on top of the q-vector minimal response-to-Casimir path.

Status:

```text
single-point diagnostic path only
lab q_vec supported
plate rotation geometry supported at one point
theta-dependent mixed trace-log supported at one point
main validation flow unchanged
main Casimir pipeline unchanged
valid_for_casimir_input: False
```

---

## 1. Purpose

The goal is to evaluate a two-plate trace-log point at a fixed lab-frame external momentum and Matsubara frequency:

```text
q_lab = (qx, qy)
plate 1 angle = theta1
plate 2 angle = theta2
```

The geometric convention is

```text
q_crystal = R(-theta_plate) q_lab,
```

where `theta_plate` is the angle of a plate's crystal axes relative to lab axes.

Each plate is evaluated in its own crystal frame using the q-vector path:

```text
plate 1: q1_crystal = R(-theta1) q_lab -> R1_TE_TM
plate 2: q2_crystal = R(-theta2) q_lab -> R2_TE_TM
```

Then the diagnostic computes one mixed trace-log point:

```text
log det[I - exp(-2 kappa d) R1_TE_TM R2_TE_TM].
```

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_theta_path.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_path.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_theta_path.py
```

---

## 3. Scope and restrictions

Supported:

```text
n >= 1
q_lab != 0
single q_lab point
single Matsubara point
single separation
single plate-angle pair
mixed trace-log at that point
```

Not supported yet:

```text
n = 0
q = 0
phi integration
q-grid integration
Matsubara summation
finite-difference torque
Casimir energy/force/torque claim
production validation
```

This path uses the q-vector LT-native reflection path for each plate.  The output reflection matrices are treated as living in the common lab propagation TE/TM basis.  This is the intended diagnostic convention for the square in-plane lattice case.

---

## 4. Ward guard status

The q-vector path inherits the current RHS-aware Ward guard behavior:

```text
if q_crystal_y = 0:
  scalar q-along-x RHS-aware validation is run

if q_crystal_y != 0:
  the payload records that the scalar-q Ward guard was not run
```

Therefore a nonzero-angle plate can have `rhs_aware_ward_closed = None` even though the response-to-reflection chain ran successfully.

---

## 5. Example command

Representative diagnostic point:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_path.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --phi-deg 30.0 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_n1_q002_phi30_theta45_nk13_minimal_casimir_theta_shifted5
```

Equivalent explicit lab-q input:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_path.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --qx 0.01732050807568877 \
  --qy 0.01 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_n1_q002_explicit_qvec_theta45_nk13_minimal_casimir_theta_shifted5
```

---

## 6. Interpretation

A successful run means only:

```text
the sandbox can evaluate plate-specific q_crystal responses, convert them to TE/TM reflection matrices, and form a finite mixed trace-log point at one lab q and one Matsubara frequency.
```

It does not mean:

```text
Casimir result is converged
torque has been computed
finite-difference theta derivative is reliable
arbitrary-q Ward guard is complete
n=0 is handled
q=0 is handled
phi/q/n integration is correct
valid_for_casimir_input can be True
```

---

## 7. Next step

The next step should be a tiny theta scan at fixed `q_lab`, `n`, `nk`, and separation.  That scan should compare the mixed trace-log at a few angles and eventually enable a diagnostic finite-difference torque-like quantity, still without claiming production readiness.
