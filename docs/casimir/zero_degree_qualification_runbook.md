# Zero-degree qualification v5 runbook

This runbook executes the frozen SPM/d-wave 0° qualification policy defined in
`zero_degree_qualification_preflight.md`.  The numerical values are constants in the
qualification command; the user cannot accidentally change one pairing without changing
the other.

## Frozen policy

```text
profile                       = 0deg_qualification_v5
source profile                = 0deg_pilot_v4
logdet_rtol                   = 2.0e-3
logdet_atol                   = 1.0e-6
required_consecutive_passes   = 2
N candidates                  = 128,192,256,384,512,640,768,896,1024,1152,1280
radial/angular budget         = 0.80 / 0.20
outer cutoff u                = 6,10,14,18,24,30,36,42,48,54,60
Matsubara cutoff              = 1,3,7,11,15,23,31
total free-energy tolerance   = 5.0e-3 relative, 1.0e-12 J/m^2 absolute
holdout safety factor         = 2.0
```

All hard microscopic physical gates remain unchanged.  SPM and d-wave use the same
controller structure, budget split, ladders and tail certification logic.

## 1. Tests

```bash
python -m pytest -q \
  tests/test_zero_degree_qualification.py \
  tests/test_zero_degree_qualification_projection.py \
  tests/test_casimir_runtime_policy.py
```

## 2. Prepare v5 caches and freeze the holdout plan

```bash
python -m scripts.full_casimir.qualification prepare \
  --pairings spm dwave \
  --source-profile 0deg_pilot_v4 \
  --profile 0deg_qualification_v5 \
  --audit-report outputs/casimir/reports/convergence_audit.compact.json \
  --holdout-plan outputs/casimir/catalog/0deg_qualification_v5_holdout_plan.json \
  --max-holdout-points 32 \
  --reserve-cpus 6 \
  --worker-cap 26 \
  --memory-budget-gb 16 \
  --max-context-workers 1 \
  --parallel-mode q \
  --certifier-q-batch-size 512
```

This command does not execute new microscopic work.  It:

- hashes both immutable v4 source runs;
- projects the complete stored histories under `logdet_rtol=0.002`;
- retains only points established by the frozen policy and contraction contract;
- writes new v5 policy fingerprints and projection reports;
- never modifies v4;
- writes a SHA-bound independent high-N holdout plan.

The targets are:

```text
outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_qualification_v5
outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_qualification_v5
```

## 3. Execute the independent high-N holdout

```bash
HOLDOUT_PLAN=outputs/casimir/catalog/0deg_qualification_v5_holdout_plan.json
HOLDOUT_SHA="$(python - <<'PY'
from pathlib import Path
import json
p = json.loads(Path('outputs/casimir/catalog/0deg_qualification_v5_holdout_plan.json').read_text())
print(p['plan_sha256'])
PY
)"

python -m scripts.full_casimir.qualification holdout \
  --plan "$HOLDOUT_PLAN" \
  --confirm-plan-sha256 "$HOLDOUT_SHA" \
  --output outputs/casimir/reports/0deg_qualification_v5_holdout.json
```

The candidate has already been frozen.  Holdout results cannot retune it.  Every selected
point and both predeclared higher-N levels must satisfy

```text
maximum shiftwise absolute delta <= 2 * predicted local uncertainty
```

with all hard physical gates passing.

## 4. Generate the launch preflight

```bash
python -m scripts.full_casimir.qualification preflight \
  --source-profile 0deg_pilot_v4 \
  --profile 0deg_qualification_v5 \
  --holdout-report outputs/casimir/reports/0deg_qualification_v5_holdout.json \
  --output outputs/casimir/catalog/0deg_qualification_v5_preflight.json
```

Preflight requires a clean tracked Git worktree, verifies both projection reports and
source hashes, requires the holdout to pass, and checks that the two configs are
pairing-blind after physical identity is removed.

## 5. Run both 0° qualification cases

```bash
PREFLIGHT=outputs/casimir/catalog/0deg_qualification_v5_preflight.json
PREFLIGHT_SHA="$(python - <<'PY'
from pathlib import Path
import json
p = json.loads(Path('outputs/casimir/catalog/0deg_qualification_v5_preflight.json').read_text())
print(p['preflight_sha256'])
PY
)"

python -m scripts.full_casimir.qualification run \
  --preflight "$PREFLIGHT" \
  --confirm-preflight-sha256 "$PREFLIGHT_SHA"
```

The runner resumes the already seeded v5 directories.  It never starts from an empty
cache and verifies the v4 source hashes before and after each pairing.  The ordinary
geometric tail path is attempted first.  The analytic passive-vacuum fallback is allowed
only if the actual accepted cache states satisfy the power-metric contraction contract
and the analytic bound fits the same common tail budget.

## 6. Verify the completed qualification

```bash
python -m scripts.full_casimir.qualification_verify \
  --preflight "$PREFLIGHT" \
  --confirm-preflight-sha256 "$PREFLIGHT_SHA" \
  --output outputs/casimir/reports/0deg_qualification_v5_final.json
```

A successful final report has

```text
status = qualification_passed
```

and verifies both run manifests, the frozen Git commit, microscopic closure, finite-domain
closure, a valid geometric or analytic outer-tail certificate, Matsubara-tail closure,
total error within tolerance, policy parity, holdout validity and unchanged v4 sources.

`production_casimir_allowed` intentionally remains false.  This qualification establishes
the frozen numerical closure contract; broader scientific authorization for downstream
Casimir use remains a separate explicit decision.
