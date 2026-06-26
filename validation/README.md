# Validation Guide

`validation/` stores validation logic, lightweight conclusions, and reproduction
entry points. It is not a long-term home for raw numerical artifacts.

## Directory Layout

- `scripts/`: reproducible validation, diagnostic, convergence, and smoke-entry scripts.
- `outputs/`: lightweight summaries, reports, command records, and small machine-readable metrics.
- `cache/`: regenerable response tensors or intermediate arrays used to speed up validation.
- `reports/`: curated inventory and cross-topic validation summaries.

## Artifact Policy

Long-term Git-tracked evidence should be compact:

- README files;
- summary markdown;
- small summary JSON or CSV;
- command scripts or reproduction command records;
- validation report documents.

Generated artifacts are ignored by default:

- `.npz` / `.npy`;
- raw, expanded, or large data CSV files;
- cache tensors;
- intermediate outputs;
- repeated benchmark figures;
- scratch logs.

`validation/cache/` is always regenerable. `validation/outputs/` may contain
small summaries, but large data products should be recreated by running the
corresponding script.

Any large artifact that must be kept should be justified in
`validation/reports/validation_artifact_inventory.md` or in the report that
uses it. New validation tasks should write raw artifacts to ignored paths and
emit a compact summary markdown/json/csv next to them.

## Reading Order

1. `validation/reports/validation_summary.md`
2. `validation/reports/validation_artifact_inventory.md`
3. Topic-specific summaries under `validation/outputs/**`
4. Reproduction scripts under `validation/scripts/**`

## Reproduction

Reports list their primary script entry points. Existing `command.sh` files or
command snippets should be kept when available. If an ignored artifact is
needed for inspection, rerun the corresponding script and regenerate it locally.
