# Normal response convention audit note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_normal_response_convention_audit.py
```

It is not a production convention proposal.

## Motivation

The normal contact Ward control showed that the normal Peierls vertex identity closes at machine precision in absolute error, but the normal response-level bubble plus contact does not close. The implemented normal contact contraction is parallel to the Ward-required contact contraction but has a scalar mismatch.

This means the blocker is already present in the normal response-level assembly before BdG pairing, phase vertices, collective channels, or Schur completion.

## What the audit scans

The audit keeps the normal Peierls vertex algebra fixed and scans response-level assembly conventions:

```text
band-vertex orientation:
  forward_minus_plus
  direct_minus_plus

Kubo factor:
  minus_plus
  denominator_flipped
  fully_reversed

source/observable current signs:
  observable_minus_source_plus
  both_plus
  observable_plus_source_minus
  both_minus

contact sign:
  minus_expectation
  plus_expectation

contact evaluation point:
  mid
  plus
  minus
  sym_pm

Ward contraction vector:
  standard
  right_spatial_plus
  left_spatial_minus_right_plus
  temporal_minus
```

The default run evaluates a small targeted list of candidates. `--full-grid` evaluates the Cartesian product and may be slower.

## What it reports

For each candidate it reports:

```text
bubble/contact/total Ward residuals
contact_required/contact_current scalar alpha
left/right alpha consistency
vertex identity absolute and relative errors
```

It also ranks candidates by total Ward residual over a reference response norm.

## Interpretation

A low-residual candidate is not automatically accepted. It is a pointer to which analytic convention should be derived and checked next.

If a contact-sign candidate alone closes the normal Ward identity, inspect the sign of the diamagnetic/direct term.

If a Kubo or band-orientation candidate closes the normal Ward identity, inspect the response assembly convention and band-basis storage before returning to BdG pairing.

If no candidate closes, the normal response may require a missing finite-q equal-time/contact term or a more careful derivation beyond the scanned controls.

## Non-goal

This audit does not test d-wave pairing and does not modify production response code. It only diagnoses normal response-level conventions.
