# Inverse-Green Ward audit note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_inverse_green_ward_audit.py
```

It is not a production convention proposal.

## Motivation

The primitive vertex audit showed that all primitive EM vertices and collective vertices are Hermitian in the current code convention, while source/observable signs differ between density and spatial current:

```text
A0 observable = + A0 source^dagger
L/T observable = - L/T source^dagger
```

A direct matrix comparison of `H(k+q/2)-H(k-q/2)` against `q Gamma_L` is therefore too crude. The BdG/Nambu Ward identity should be checked against inverse Green functions and may require tau3 insertions, e.g.

```text
tau3 G^{-1}_+ - G^{-1}_- tau3
```

rather than only

```text
G^{-1}_+ - G^{-1}_-
```

## What the audit computes

At a single representative k point, the audit builds

```text
G^{-1}_-(z_-, k-q/2) = z_- I - H_BdG(k-q/2)
G^{-1}_+(z_+, k+q/2) = z_+ I - H_BdG(k+q/2)
```

and reports multiple diagnostic references:

```text
plain_delta_Ginv_plus_minus_minus
nambu_tau_left_Gplus_minus_Gminus_tau_right
nambu_Gplus_tau_right_minus_tau_left_Gminus
left_tau_plain_delta
plain_delta_right_tau
```

It then compares the same Ward-like vertex combinations used in the vertex convention audit against each reference.

## Frequency conventions

The audit reports several transfer-frequency conventions:

```text
matsubara_i_transfer
matsubara_minus_i_transfer
real_transfer_debug
```

These are diagnostic probes. They are not accepted conventions.

## Non-goal

Do not select a production convention from the smallest residual in this audit. A production convention can only be accepted after the analytic imaginary-time BdG/Nambu Ward derivation predicts it and response-level q/n/nk sweeps verify it.
