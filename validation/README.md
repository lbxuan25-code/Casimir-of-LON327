# Validation

`validation/` contains reproducible checks for Ward identities, response conventions,
numerical stability, and independent quadrature contracts. It is not part of the
production calculation implementation.

The dependency direction is:

```text
validation -> src/lno327
```

Code under `src/lno327` must never depend on the top-level `validation` package.

## Fixed Casimir chain

The complete fixed microscopic Casimir chain is production-owned:

```python
from lno327.casimir import FixedCasimirConfig, run_casimir

result = run_casimir(FixedCasimirConfig())
```

The former validation compatibility facades, legacy numerical copies,
`microscopic-outer-q-preflight` command, and `transverse-point-sweet-spot` command
have been removed. Validation no longer provides an alternate entry point for the
fixed Casimir calculation.

The immutable qualified `spm`, `n=0,1` golden fixture remains under
`validation/references/casimir/` as regression evidence. It is not a runtime output
and does not authorize a complete production Casimir result.

## Runtime outputs

All files generated under `validation/outputs/`, `validation/logs/`, and
`validation/cache/` are local reproducible artifacts and are ignored by version
control. Results that must become long-lived contracts should be reduced to a small,
reviewed fixture under `validation/references/` or documented in `docs/`.

## Retained command surface

```bash
python -m validation <group> <command> [options]
```

The retained CLI exposes independent Ward, Matsubara, numerical, diagnostic, and
outer-measure checks listed by `python -m validation --help`. These commands never
authorize production Casimir output.
