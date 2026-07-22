# Casimir production route

## Operational surface

The only user-facing calculation route is:

```bash
python -m scripts.full_casimir plan ...
python -m scripts.full_casimir run ... --fresh|--resume
```

`plan` freezes the scientific policy, physical case matrix, Git source identity and
plan SHA. `run` accepts only that confirmed plan and executes it under one campaign
owner lock, with explicit stale takeover, bounded retries and atomic-cache recovery.
There is no package command, installed console command, single-case command, pilot
runner, qualification runner, background wrapper, or cache-extension calculation
route.

The Python objects exported by `lno327.casimir` are internal library components used
by the unified dispatcher and tests. Calling those functions from ad hoc scripts is
not an authorized production workflow.

## Layer order

```text
immutable top-level plan
→ FrequencyExtendableCertifiedOuterQProvider
→ adaptive radial panels
→ global periodic angular order and offset audit
→ joint radial/angular direction selection
→ cumulative u cutoff and certified outer tail
→ cumulative dyadic Matsubara blocks and holdout tail certificate
→ total error-budget and production-authorization gate
```

Every layer preserves pairing- and Matsubara-channel error evidence. If any
microscopic point, finite-domain integral, tail certificate, or total budget remains
unresolved, the case is not marked `completed`.

Completed attempts produce source, plan, environment and authoritative-artifact proof
records. Verify them without calculation through:

```bash
python -m scripts.full_casimir proof --campaign <campaign-id>
```

## Documents

- `numerical_contract.md`: error budgets and certification rules;
- `operations.md`: planning, locks, fresh execution, resume, recovery and proof;
- `progress_reporting.md`: persisted runtime events and status display;
- `todo5_numerical_and_execution_policy.md`: frozen numerical/execution policies;
- `legacy_fixed_reference.md`: isolated historical library reference only.
