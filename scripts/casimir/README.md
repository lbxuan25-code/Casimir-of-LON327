# Casimir operational workflow

All repository-local run and post-processing helpers live in this directory. No
Python or shell workflow scripts should be added to the repository root.

## Commands

Show the CPU reservation without starting a calculation:

```bash
python -m scripts.casimir.workflow resources
```

Run the SPM and d-wave 0° pilots sequentially in the background:

```bash
bash scripts/casimir/background.sh start pilots
```

Run the padded energy scan, -4° to 94° in 2° steps:

```bash
bash scripts/casimir/background.sh start scan
```

Run the complete scan and then automatically extract torque and create plots:

```bash
bash scripts/casimir/background.sh start all
```

Inspect or stop the background job:

```bash
bash scripts/casimir/background.sh status
bash scripts/casimir/background.sh logs
bash scripts/casimir/background.sh stop
```

Post-process an already completed scan in the foreground:

```bash
python -m scripts.casimir.workflow torque
python -m scripts.casimir.workflow plot
```

## Defaults

- T = 10 K
- d = 20 nm
- pairings = SPM, d-wave
- N ladder = 128, 192, 256, 384, 512, 640, 768, 896
- Matsubara cutoffs = 1, 3, 7, 11, 15, 23, 31
- total free-energy tolerance = 0.5% relative and 1e-12 J/m² absolute
- at most 28 workers, with 4 logical CPUs reserved
- single-threaded BLAS/OpenMP inside each worker
- low CPU and I/O scheduling priority for background jobs

## Outputs

Energy run artifacts:

```text
outputs/casimir/runs/<case>/
```

Workflow logs:

```text
outputs/casimir/workflow_logs/
```

Post-processing tables and figures:

```text
outputs/casimir/postprocessed/runtime_budget_v2/
```

Torque uses a five-point centered derivative in radians. The default energy scan
therefore includes -4° and 94° padding so torque can be evaluated on 0°–90°.
