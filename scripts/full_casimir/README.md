# Casimir operational workflow

All repository-local run and post-processing helpers live in this directory. No
Python or shell workflow scripts should be added to the repository root.

The numerical route remains fail closed. A converged numerical artifact still carries
`production_casimir_allowed = false` until the separate physical qualification is closed.

## Commands

Show the CPU reservation without starting a calculation:

```bash
python -m scripts.full_casimir.workflow resources
```

Run the SPM and d-wave 0° v3 pilots sequentially in the background:

```bash
bash scripts/full_casimir/background.sh start pilots
```

Run the padded energy scan, -4° to 94° in 2° steps:

```bash
bash scripts/full_casimir/background.sh start scan
```

Run the complete scan and then automatically extract torque diagnostics and plots:

```bash
bash scripts/full_casimir/background.sh start all
```

Inspect, follow, or stop the background job:

```bash
bash scripts/full_casimir/background.sh status
bash scripts/full_casimir/background.sh logs
bash scripts/full_casimir/background.sh stop
```

Post-process an already completed scan in the foreground:

```bash
python -m scripts.full_casimir.workflow torque
python -m scripts.full_casimir.workflow plot
```

## Audited v3 defaults

- T = 10 K
- d = 20 nm
- pairings = SPM, d-wave
- N ladder = 128, 192, 256, 384, 512, 640, 768, 896
- required consecutive accepted transitions = 2
- microscopic logdet tolerance = 1.5e-3 relative and 1e-6 absolute
- certifier q batch size = 384
- Matsubara cutoffs = 1, 3, 7, 11, 15, 23, 31
- total free-energy tolerance = 0.5% relative and 1e-12 J/m² absolute
- at most 26 workers, with 6 logical CPUs reserved
- one live material context at high N
- single-threaded BLAS/OpenMP before any numerical import
- low CPU and I/O scheduling priority for background jobs

The v3 point policy intentionally uses new case names and caches. It must not resume the
old `0deg_pilot_v2` artifacts because the microscopic logdet tolerance is part of the
cache fingerprint.

## Outputs

Energy run artifacts:

```text
outputs/casimir/runs/<case>/
```

Timestamped workflow logs and the `background.log` symlink to the latest run:

```text
outputs/casimir/workflow_logs/
```

Post-processing tables and figures:

```text
outputs/casimir/postprocessed/runtime_budget_v3/
```

Torque uses a five-point centered derivative in radians. The default energy scan includes
-4° and 94° padding so torque can be evaluated on 0°–90°. The propagated energy-error
term is a formal bound from the available energy bounds. The difference between the
five-point and three-point derivatives is recorded separately as a stencil-sensitivity
diagnostic; their sum is not promoted to a formal error bound.
