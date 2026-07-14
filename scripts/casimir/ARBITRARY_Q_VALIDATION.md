# Arbitrary-q validation

This flow separates inexpensive structural checks from formal evidence. The smoke checks are diagnostic-only and may not authorize outer integration.

## Setup

```bash
python -m pip install -e ".[dev]"
```

The runner sets all BLAS/OpenMP thread variables before starting Python.

## Performance smoke

```bash
bash scripts/casimir/run_arbitrary_q_validation.sh performance-smoke
```

Default workload: `spm`, `N=128`, four q tasks/four workers, Matsubara `0,1,2,4,8`, runtime chunks `4096/16384`.

Output:

```text
validation/outputs/matsubara/arbitrary_q_validation/<head>/performance_smoke.json
```

The JSON reports:

- material-cache build time;
- q-workspace, Kubo-factor, contraction, primitive-pack, operator-Ward and Kahan-accumulation times;
- each timing component's fraction of reported response time;
- runtime-chunk and shifted-eigensystem call counts;
- short/full Matsubara eigensystem reuse;
- serial runtime-chunk equality;
- process wall time, worker time, pool startup/shutdown, memory and IPC metadata.

This command checks optimization structure, not formal speedup thresholds.

## Physics smoke

```bash
bash scripts/casimir/run_arbitrary_q_validation.sh physics-smoke
```

Default workload: `spm/dwave`, `N=128,192`, axis/generic/near-diagonal/exact-diagonal/17-degree rotated q, Matsubara `0,1,8`, and the true two-plate `0/17 degree` common-lab observable.

Output:

```text
validation/outputs/matsubara/arbitrary_q_validation/<head>/physics_smoke.json
```

The smoke pass requires:

- all Peierls operator identities;
- all integrated RHS-aware Ward gates;
- all positive-frequency sheet/reflection/logdet pipelines;
- finite exact-zero strict-static metrics;
- all positive-frequency two-plate logdets.

It deliberately does **not** require small-N zero-mode strict-static convergence. The JSON records every strict-static metric at each N and its trend.

Run both smoke checks with:

```bash
bash scripts/casimir/run_arbitrary_q_validation.sh diagnostics
```

## Formal performance preflight

After the smoke checks are understood, use a clean checkout:

```bash
git status --porcelain --untracked-files=all
bash scripts/casimir/run_arbitrary_q_validation.sh performance-preflight
```

Output:

```text
validation/outputs/matsubara/arbitrary_q_validation/<head>/performance_preflight.json
```

This creates the formal 12-cell performance evidence matrix:

```text
spm/dwave × runtime 4096/16384 × outer/primary/audit
```

Do not modify source after this succeeds.

## Periodic-BZ qualification

```bash
git status --porcelain --untracked-files=all
bash scripts/casimir/run_arbitrary_q_validation.sh qualification
```

Output:

```text
validation/outputs/matsubara/arbitrary_q_validation/<head>/periodic_bz_qualification.json
```

This runs the existing public clean-source gate with `N=256,384,512`, complete-orbit C384 references, A/B and primitive-paired shifts, and the final two-plate logdet convergence gates.

## q-envelope

No generic command is provided yet. Before implementing it, freeze:

```text
separation interval
angle interval
q quadrature family
q cutoff rule
tail tolerance
```

Then derive the required `q_max` and build a dedicated envelope/tail manifest. Periodic-BZ qualification alone does not authorize a full outer integral.
