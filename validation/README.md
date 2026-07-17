# Validation

`validation/` contains reproducible checks for the current two-band finite-q response
and fixed Casimir-chain contracts. It is not part of the production calculation
implementation.

The dependency direction is:

```text
validation -> src/lno327
```

Validation commands may consume production numerical implementations. Code under
`src/lno327` must never depend on `validation`.

Current migration status:

- production owns the Matsubara energy helper;
- production owns the active finite-q microscopic model adapter;
- production owns the fixed transverse-point engine and universal point-certification controller;
- production owns fixed nested outer-q planning, node reuse, reduction, and ladder comparisons;
- validation retains compatibility commands, diagnostics, report schemas, and regression evidence.

Validation outputs are diagnostic evidence only and never authorize a production
Casimir result by themselves.

## Public command surface

```bash
python -m validation <group> <command> [options]
```

Main retained and blocking commands include the Ward, static, Matsubara, diagnostic,
and Casimir preflight groups exposed by `python -m validation --help`.
