# Qualification holdout selection correction

The official preparation entrypoint is:

```bash
python -m scripts.full_casimir.qualification_prepare
```

It replaces the initial preparation entrypoint for holdout selection.  Acceptance-status
changes (`not_established` to `established`) are mandatory.  Earlier stopping under the
same accepted status is recorded but sampled by quadrature-weighted importance.  The
explicit `--max-holdout-points` cap is enforced; mandatory overflow fails closed rather
than silently truncating boundary evidence.
