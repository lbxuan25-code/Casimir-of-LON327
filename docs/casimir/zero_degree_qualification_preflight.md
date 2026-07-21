# SPM / d-wave 0° qualification rerun preflight

Status: **numerical policy frozen; preparation code implemented; local holdout and run not yet executed**  
Target profile: `0deg_qualification_v5`  
Source evidence: the existing immutable `0deg_pilot_v4` runs  
Production authorization: **false until the local qualification and final verifier pass**

## 1. Purpose

This report freezes the numerical strategy for the next 0° qualification rerun of the SPM and d-wave pairings. The two pairings must use the same numerical structure, acceptance gates, adaptive ladders and error allocations. Pairing identity may change only the material response values, not the numerical decision rules.

The rerun must reuse the existing v4 certified-point caches. It must not start from an empty cache and must not modify either v4 source run.

Source runs:

```text
outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_pilot_v4
outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_pilot_v4
```

Target runs:

```text
outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_qualification_v5
outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_qualification_v5
```

## 2. Frozen physical case

Both runs use:

```text
temperature_K       = 10.0
separation_nm       = 20.0
plate_angles_deg    = (0.0, 0.0)
delta0_eV           = 0.1
eta_eV              = 1e-8
degeneracy          = 1.0
```

No physical model parameter is changed by the numerical qualification.

## 3. Frozen microscopic certification policy

The selected candidate is the least-relaxed stored-history candidate that closes the known d-wave boundary points in the completed convergence audit:

```text
logdet_rtol                  = 2.0e-3
logdet_atol                  = 1.0e-6
required_consecutive_passes  = 2
```

The following remain unchanged:

- all hard-physical response, Ward, sheet, passivity and determinant gates;
- cross-shift, adjacent-N and three-level oscillatory-envelope structure;
- the requirement for two consecutive accepted transitions;
- fail-closed handling of malformed, missing or non-finite evidence.

The global candidate comparison is therefore fixed as:

```text
strict reference  = 1.5e-3
selected moderate = 2.0e-3
aggressive audit  = 2.5e-3, not selected
stress boundary   = 3.0e-3, not selected
```

`2.5e-3` and `3.0e-3` must not be substituted after the run starts. A failure of the selected candidate triggers a new audit and a new holdout set rather than retuning against the same holdout evidence.

### Unified N ladder

Both pairings use the union ladder already represented by the v4 evidence:

```text
N_candidates = (
    128, 192, 256, 384, 512, 640, 768, 896,
    1024, 1152, 1280,
)
```

A point may stop earlier when the same gates pass. The code may not assign pairing-specific ladders.

## 4. Frozen finite-domain and total-energy policy

Both pairings use the same controller and budget split:

```text
radial_budget_fraction   = 0.80
angular_budget_fraction  = 0.20

total_free_energy_rtol        = 5.0e-3
total_free_energy_atol_J_m2   = 1.0e-12
```

The former pairing-dependent production default (`0.85` for SPM and `0.75` otherwise) has been removed. `build_full_casimir_config` now accepts one explicit pairing-blind radial budget and defaults to `0.80`.

The finite-domain ledger uses the already-combined radial/angular/offset bound. Radial, angular and offset diagnostic components must not be added a second time.

## 5. Frozen outer-Q policy

Both pairings use the same cutoff ladder and geometric settings:

```text
cutoff_u_values = (6, 10, 14, 18, 24, 30, 36, 42, 48, 54, 60)
tail_start_u = 24
tail_window_shells = 3
tail_ratio_max = 0.8
```

The unified tail certificate has two possible paths:

1. **Resolved geometric path**: the central shell signal is numerically resolved and the final window contracts under the common ratio bound.
2. **Analytic passive-vacuum path**: the actual accepted cache states satisfy the determinant-preserving power-metric contraction contract, after which the pairing-independent analytic tail integral may be used.

Rules:

- `below_finite_domain_resolution` is diagnostic evidence only and never certifies the omitted tail by itself;
- a path participates only when all of its premises are certified;
- if the geometric path passes, it remains the primary route;
- the analytic path is a fail-closed fallback, not a shortcut around finite-domain convergence;
- if neither path is valid, the run remains unresolved;
- SPM and d-wave use exactly the same logic.

The qualification runner re-reads the target cache before using the analytic path. Positive-Matsubara accepted plate states use the validated passive-sheet vacuum-admittance similarity theorem; zero-Matsubara states require a persisted conservative reflection contraction bound. Missing evidence leaves the run unresolved.

## 6. Frozen Matsubara policy

Both pairings use:

```text
matsubara_cutoff_values = (1, 3, 7, 11, 15, 23, 31)
matsubara_tail_start_n = 8
matsubara_tail_window_terms = 4
matsubara_tail_ratio_max = 0.8
```

No pairing-specific Matsubara ladder or tail rule is permitted.

## 7. Cache reuse contract

### 7.1 Source immutability

Before and after every preparation, holdout and qualification step, record and compare SHA-256 for:

```text
<source-v4>/config.json
<source-v4>/manifest.json
<source-v4>/result.json
<source-v4>/summary.json
<source-v4>/cache/certified_points.json
```

Any source change aborts the workflow.

### 7.2 Audited policy projection

The ordinary `prepare-pilot-extension` path still permits only an unchanged microscopic point policy or a strict N-ladder prefix extension. It is deliberately not used for this rtol change.

The implemented qualification preparation command instead:

1. validates the complete v4 source policy and hashes;
2. retains the complete stored N/shift history;
3. replays the frozen v5 gates with `logdet_rtol=2.0e-3`;
4. preserves every hard-physical result unchanged;
5. writes a new v5 sweet spot only from stored evidence;
6. omits entries that remain unresolved or lack contraction evidence, allowing only those identities to trigger new microscopic work;
7. writes the v5 policy payload and fingerprint;
8. records source/target SHA-256, retained and omitted identities, and every changed decision;
9. refuses to overwrite a partial or incompatible target.

This is reuse by verified history projection, not blind byte copying.

### 7.3 Expected reuse

- SPM v4 has established microscopic entries and should seed essentially all previously requested points.
- d-wave v4 should seed all strict-established entries and reclassify the four known boundary histories under the frozen moderate policy when the stored replay supports establishment.
- New microscopic work is allowed only for genuinely missing q/n identities, independent holdout levels, or points omitted because the frozen policy still does not establish them.
- Expanding the common outer cutoff to `u=60` may require new d-wave q points. Those are incremental additions, not a restart.

## 8. Independent high-N holdout

The candidate policy is frozen before the holdout is executed. The SHA-bound plan includes:

- all points whose acceptance decision changes under projection, including the known d-wave boundary set when present;
- the largest quadrature-weighted uncertainty contributors from the compact convergence audit;
- representatives for pairing and Matsubara strata;
- ordinary easy controls.

Each selected point is evaluated at two predeclared N levels above its highest stored valid N. Acceptance is

```text
maximum shiftwise abs(L_holdout - L_accepted)
    <= 2.0 * predicted_local_uncertainty
```

for every selected point and both holdout levels, with all hard-physical gates passing. The factor `2.0` is frozen before execution.

Holdout values are not used to retune `logdet_rtol`, the N ladder, or the safety factor. Failure rejects the candidate and requires a new candidate plus a new independent holdout.

## 9. Real-work benchmark

The holdout uses the real strict transverse certifier and records:

- wall time by grouped batch;
- full level-resolved execution telemetry;
- hard-gate status and shiftwise errors for each point and N level;
- source and target cache immutability.

The subsequent qualification run retains the existing provider telemetry for new versus cache-hit q and point evaluations, cache growth and certifier timings. Cache-only replay time and the N-squared work proxy remain diagnostics rather than real-work evidence.

## 10. Implemented launch gates

The repository now contains:

1. pairing-blind radial/angular configuration;
2. an audited v4-to-v5 history-projection cache command;
3. a SHA-bound holdout planner and real strict-certifier holdout executor;
4. a cached power-metric contraction certificate and a premise-checked analytic tail fallback;
5. identical tail-controller logic for both pairings;
6. a preflight manifest containing the frozen policy, source hashes, target hashes, holdout hash and Git commit;
7. a policy-parity test and final two-run verifier.

Code readiness is now:

```text
ready_to_prepare_v5_cache = true
ready_to_execute_holdout = true
ready_to_generate_preflight = true
ready_to_run_v5 = conditional_on_local_holdout_and_preflight
production_change_authorized = false
```

## 11. Execution order

The exact commands are maintained in `zero_degree_qualification_runbook.md`:

1. run focused tests;
2. project both v4 caches into v5 and freeze the holdout plan;
3. execute the independent high-N holdout;
4. generate the SHA-bound preflight;
5. run SPM and d-wave 0° through the qualification runner;
6. execute the final qualification verifier;
7. regenerate diagnostics/catalog only after the final report is written;
8. keep v4 as frozen evidence until v5 is formally accepted.

## 12. Final acceptance contract

Each pairing must independently satisfy:

```text
all_microscopic_points_established = true
independent_high_N_holdout_passed = true
finite_domain_budget_passed = true
outer_tail_bound_certified = true
matsubara_tail_bound_certified = true
nonduplicating_total_error_within_tolerance = true
source_v4_cache_unchanged = true
```

The joint decision additionally requires:

```text
pairing_blind_numerical_policy = true
same_tail_controller_structure = true
same_error_allocation = true
real_work_benchmark_recorded = true
```

A failure in one pairing does not authorize a pairing-specific exception. It leaves that pairing unresolved and requires a new common-policy audit.

## 13. Frozen decision

```text
profile                       = 0deg_qualification_v5
logdet_rtol                   = 2.0e-3
logdet_atol                   = 1.0e-6
required_consecutive_passes   = 2
N_candidates                  = 128..1280 unified ladder
radial/angular budget         = 0.80 / 0.20
outer cutoff ladder           = 6..60 unified ladder
Matsubara ladder              = 1,3,7,11,15,23,31
hard physical gates           = unchanged
source cache                  = each pairing's immutable 0deg_pilot_v4 cache
cache strategy                = audited full-history projection, never empty restart
```

This is a frozen qualification policy and executable preparation path. It is not yet a downstream production-Casimir authorization.
