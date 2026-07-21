# TODO 5 numerical and execution policy

Status: **microscopic policy frozen; execution policy frozen; outer-Q and Matsubara endpoints remain qualification candidates**

## Scope

TODO 5 has two different responsibilities:

1. freeze numerical inputs that already have sufficient qualification evidence;
2. freeze the engineering route used to realize those inputs efficiently.

The transverse Brillouin-zone policy is frozen now.  The outer-Q and Matsubara
ladders are conservative candidate ceilings for the fresh 0-degree run after TODO
8.  That run may stop before the ceiling.  The final economical outer and
Matsubara endpoints are recorded only after the complete local run closes the
TODO 4 error budget.

## Frozen microscopic policy

Policy ID: `full-casimir-microscopic-policy-v1`

```text
pairing policy                 = pairing blind
N candidates                   = 128, 192, 256, 384, 512, 640,
                                 768, 896, 1024, 1152, 1280
shifts                         = (0.5,0.5), (0.25,0.75), (0.75,0.25)
logdet rtol                    = 2.0e-3
logdet atol                    = 1.0e-6
required consecutive passes   = 2
hard physical gates           = unchanged and fail closed
point-specific early stopping = enabled
```

This is the least-relaxed common policy selected by the completed 0-degree
transverse audit.  SPM and d-wave may stop at different N values, but they use
the same ladder and acceptance rules.

## Qualification candidate outer-Q policy

Policy ID: `full-casimir-outer-candidate-v1`

```text
cutoff u ladder = 6, 10, 14, 18, 24, 30, 36, 42, 48, 54, 60
tail start u    = 24
shell window    = 3
geometric ratio = 0.8 (diagnostic/numerical certificate path)
```

The passive-vacuum analytic certificate is attempted at every cutoff.  Therefore
the conservative ceiling at 60 does not force calculation to 60 when the finite
domain and analytic tail budgets already pass at 18 or 24.

The finite radial/angular orders, panel refinement history and selected cutoff
from the fresh 0-degree run must be retained.  Their final economical defaults
remain pending until that run is complete.

## Qualification candidate Matsubara policy

Policy ID: `full-casimir-matsubara-candidate-v1`

```text
cutoff maxima = 1, 3, 7, 15, 31, 63
blocks        = 0-1, 2-3, 4-7, 8-15, 16-31, 32-63
holdout       = final complete dyadic block
```

The candidate maximum is not a prediction that every case requires n=63.  The
formal dyadic-block certificate may stop at an earlier complete block.  The
fresh 0-degree run determines whether the candidate ceiling and holdout window
are economical enough for later production.

## Frozen execution policy

Policy ID: `full-casimir-execution-policy-v1`

The qualified process structure is:

```text
parallel layer          = one process layer only
primary strategy        = q-parallel persistent fork pool
BLAS/OpenMP threads     = one per process
canonical reduce block  = 4096
runtime chunk           = 16384
q certification batch  = 512
nested process pools    = forbidden
```

These choices are supported by the existing arbitrary-q formal performance
policy and performance preflight.  In particular, the 16384 runtime chunk
reduces repeated q-workspace construction while preserving the canonical 4096
reduction order.

### Local workstation profile

```text
profile name         = local-workstation-v1
reserved logical CPU = 6
worker cap           = 26
parallel mode        = q
memory budget        = 16 GiB
max context workers  = 1
q batch size         = 512
```

### Server throughput profile

```text
profile name         = server-throughput-v1
reserved logical CPU = 2
worker cap           = no artificial cap; use affinity minus reserve
parallel mode        = q
memory budget        = automatic host budget
max context workers  = 1
q batch size         = 512
```

The server profile intentionally preserves the same numerical process shape as
the local profile.  Only execution resources change, so the same scientific
cache identity remains valid.

## Performance evidence required from the fresh 0-degree run

The provider telemetry must report at least:

- requested, new and cache-hit q and point counts;
- certifier wall time and batch count;
- material-cache build time;
- context evaluation time;
- cache load/save time and file size;
- selected N distribution;
- selected outer cutoff and radial/angular refinement work;
- Matsubara block expansion history;
- process strategy, worker count and memory records.

The run must be reviewed for repeated material-context construction and cache
full-rewrite cost.  If either becomes a dominant fraction of total time, the run
is paused and the engineering policy is revised before the wider scan.

## Completion boundary

TODO 5 is considered complete when:

- the transverse policy is frozen in production defaults and tests;
- local and server execution profiles are fixed;
- the candidate outer-Q and Matsubara ladders are recorded in the policy;
- the full contract suite and engineering smoke tests pass.

Final economical outer-Q and Matsubara endpoints are deliberately deferred to
the post-TODO-8 fresh 0-degree run.  This deferred measurement is part of the
qualification campaign, not an unfinished transverse or execution-policy task.

After TODO 5, the repository undergoes one engineering cleanup pass.  Cleanup may
reorganize modules and remove obsolete compatibility clutter, but it may not
change the policy values or acceptance semantics without a new policy identity.
