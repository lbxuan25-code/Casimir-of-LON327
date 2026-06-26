# Validation Summary

This summary preserves current validation conclusions at topic level. It does
not replace detailed topic reports under `validation/outputs/**`; it points to
the relevant reproduction entry points and records whether a topic blocks the
main pipeline.

## Numerical Stability

Purpose: check sampling sensitivity, imaginary-axis convergence, high-Nk trends,
BDG normal-limit behavior, static response behavior, and n=0 policy sensitivity.

Current conclusion: numerical stability scripts support using the current local
response and response-kernel conventions as diagnostic benchmarks. Large arrays
and plots are regenerable and are not needed in Git as long as summary markdown
is retained.

Production relevance: indirect. These checks support confidence in response
inputs but do not define material production outputs.

Diagnostic-only: yes for most convergence scans.

Reproduction entry: `validation/scripts/numerical_stability/*.py`.

Blocking status: not a main-flow blocker; rerun targeted scripts when changing
integration grids, response units, or n=0 policy.

## Response Convention And Ward Diagnostics

Purpose: test Peierls vertices, contact terms, density/current Ward sectors,
response-level Ward conventions, and normal-state finite-q closure.

Current conclusion: normal-state and response-convention audits identify the
consistent density/current/contact structure used by the response pipeline.
Detailed compact tables and figures are reproducible from scripts; durable
conclusions live in markdown summaries under `validation/outputs/response/**`.

Production relevance: supports response conventions, but validation reports do
not modify responses.

Diagnostic-only: yes.

Reproduction entry: `validation/scripts/response/audit_*.py`,
`validation/scripts/response/diagnose_*.py`, and
`validation/scripts/response/verify_response_level_ward_conventions.py`.

Blocking status: not currently blocking the local response flow.

## Finite-q BdG Diagnostic Status

Purpose: audit finite-q BdG pairing, collective vertices, Schur restoration,
Goldstone/counterterm behavior, and Ward residuals for superconducting ansaetze.

Current conclusion: the minimal separable pairing ansatz and generic finite-q
engine are structurally separated. Existing finite-q superconducting diagnostics
remain diagnostic-only. The `dwave` finite-q Schur Ward issue is not promoted to
production input, and LSQ references remain non-production diagnostics.

Production relevance: not production-ready for formal Casimir input.

Diagnostic-only: yes.

Reproduction entry: `validation/scripts/response/stageSC_*.py` and shared
helpers under `validation/scripts/response/*_common.py`.

Blocking status: blocks using raw finite-q BdG kernels as formal Casimir input;
does not block local q=0 response workflows.

## Local-response Casimir Convergence / Benchmark

Purpose: benchmark local-response Casimir integration, cutoff choices, cache
behavior, and convergence scaffolding.

Current conclusion: local-response convergence scripts are benchmark and
planning tools. They are not final material conclusions. Response caches are
regenerable and should not be tracked.

Production relevance: supports workflow planning only unless a specific report
states otherwise.

Diagnostic-only: yes for current validation outputs.

Reproduction entry:
`validation/scripts/casimir/refine_casimir_local_convergence_blockers.py`,
`validation/scripts/casimir/converge_casimir_local_response_integral.py`, and
`validation/scripts/casimir/run_casimir_local_convergence_final.py`.

Blocking status: not a code blocker; final convergence claims require explicit
fresh runs and retained summary reports.

## Unit / Reflection Input Audits

Purpose: check model-response to sheet-conductivity conversion, dimensionless
reflection input formatting, TE/TM adapter conventions, and q-grid mapping.

Current conclusion: stage-5 response/unit reports document the current
candidate conventions. Raw arrays and figures are not needed for long-term
review when markdown/json summaries are retained.

Production relevance: candidate/scaffold level unless a downstream production
report explicitly consumes validated, unit-converted tensors.

Diagnostic-only: mostly yes.

Reproduction entry:
`validation/scripts/response/stage5_*.py` and
`validation/scripts/units/audit_casimir_q_grid_to_model_q.py`.

Blocking status: raw diagnostic tensors remain blocked from formal Casimir
input without explicit Ward validation, unit conversion, and n=0 policy.

## Smoke / Plumbing Checks

Purpose: ensure scripts run, output paths are writable, cache behavior works,
and minimal smoke calculations do not regress.

Current conclusion: smoke tests and pytest cover the operational path. Generated
smoke outputs are scratch artifacts unless summarized.

Production relevance: operational only.

Diagnostic-only: yes.

Reproduction entry: `validation/scripts/smoke/*.py` and repository tests.

Blocking status: failures block developer confidence but do not by themselves
define physics conclusions.
