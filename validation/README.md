# Validation

`validation/` contains reproducible checks for the current two-band finite-q response contract. It is not the full Casimir outer integrator.

The dependency direction for the fixed Casimir chain is:

```text
validation -> src/lno327
```

Validation commands may consume production numerical implementations. Code under
`src/lno327` must never depend on `validation`.

Current migration status:

- production owns the Matsubara energy helper;
- production owns the active finite-q microscopic model adapter;
- production owns fixed nested outer-q planning, node reuse, reduction, and ladder comparisons;
- validation retains compatibility commands, diagnostics, and report schemas;
- the transverse-point certification engine remains the next mechanical migration boundary.

Validation outputs are diagnostic evidence only and never authorize a production
Casimir result by themselves.

## Public command surface

```bash
python -m validation <group> <command> [options]
```

Main retained and blocking commands include the Ward, static, Matsubara, diagnostic,
and Casimir preflight groups exposed by `python -m validation --help`.
