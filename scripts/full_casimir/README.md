# Casimir operational workflow

All run and post-processing helpers live under `scripts/full_casimir/`.

## Runtime-v3 pilot

```bash
python -m scripts.full_casimir.workflow resources
bash scripts/full_casimir/background.sh start pilots
```

The pilot reserves six logical CPUs (maximum 26 workers), runs SPM then d-wave,
uses `logdet_rtol=1.5e-3`, keeps `logdet_atol=1e-6` and two consecutive accepted
transitions, and uses 512-q certifier batches.  When v2 pilot caches are present,
the workflow safely reassesses their stored raw N/shift histories under the relaxed
relative logdet criterion and seeds the v3 cache.  It never alters the v2 evidence.

Inspect or stop the background job:

```bash
bash scripts/full_casimir/background.sh status
bash scripts/full_casimir/background.sh logs
bash scripts/full_casimir/background.sh stop
```

Run the padded scan after both v3 pilots are reviewed:

```bash
bash scripts/full_casimir/background.sh start scan
```

Run scan, torque extraction, and plotting as one fail-closed workflow:

```bash
bash scripts/full_casimir/background.sh start all
```

`all` stops before post-processing if any energy is unresolved or an engineering
failure occurs.  Torque uses the five-point centered derivative with angle in
radians, so the default energy scan includes -4° and 94° padding.

Outputs are under `outputs/casimir/runs`, logs under
`outputs/casimir/workflow_logs`, and post-processing under
`outputs/casimir/postprocessed/runtime_budget_v3`.
