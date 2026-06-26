# Validation Artifact Inventory

This inventory is a maintenance policy for `validation/`, not a file-by-file
manifest. The goal is to preserve conclusions and reproducibility while keeping
large generated artifacts out of version control.

## Keep In Git

Keep artifacts that are useful for review without rerunning heavy jobs:

- `validation/README.md` and report documents;
- topic README files;
- summary markdown files;
- small summary JSON or CSV files;
- command scripts or command snippets used for reproduction;
- validation scripts and tests.

Small machine-readable metrics are allowed when they are summaries rather than
raw arrays. If a JSON summary grows into a large raw dump, split it into a small
summary and an ignored raw artifact.

## Do Not Keep In Git

The following are regenerable and should be ignored or removed from tracking:

- `validation/cache/**/*.npz`, `.npy`, `.csv`, `.jsonl`;
- `validation/outputs/**/data/*.npz` and `.npy`;
- `validation/outputs/**/data/*.csv` unless it is a compact summary CSV;
- `*expanded*.csv`, `*raw*.csv`, and raw/intermediate output directories;
- repeated figures under `validation/outputs/**/figures`;
- benchmark scratch outputs and logs.

Figures are optional evidence. Keep only a small number when a report explicitly
references them and text/table summaries are not enough. Prefer moving such
figures near the report or under `docs/assets/validation/` with provenance.

## Cache Versus Outputs

`validation/cache/` contains reusable intermediate tensors. It exists to speed
up local validation and is always safe to regenerate.

`validation/outputs/` contains script outputs. Only lightweight summaries and
reproduction records should be tracked. Data arrays and plots in subdirectories
such as `data/`, `raw/`, `intermediate/`, and `figures/` are local artifacts.

## Regeneration Policy

When a raw artifact is needed for review:

1. Read the topic summary under `validation/outputs/**`.
2. Find the corresponding script under `validation/scripts/**`.
3. Recreate outputs locally, usually with an `--output-prefix` or default path.
4. Commit only updated summaries or reports unless there is a documented reason
   to keep a larger artifact.

Common entry points:

- numerical stability: `validation/scripts/numerical_stability/*.py`;
- response and Ward diagnostics: `validation/scripts/response/*.py`;
- local-response Casimir convergence: `validation/scripts/casimir/*.py`;
- unit/reflection audits: `validation/scripts/units/*.py` and response stage-5 scripts;
- smoke/plumbing checks: `validation/scripts/smoke/*.py`.

## Current Gitignore Policy

`.gitignore` ignores validation cache tensors, binary data arrays, raw/expanded
CSV files, raw/intermediate directories, and repeated figures. It explicitly
allows README files, reports, summary markdown/json/csv, and command records.

The policy is intentionally conservative: preserve scripts and conclusions,
regenerate bulky evidence on demand.

## Cleanup Snapshot

On 2026-06-26, the repository policy changed from tracking bulky validation
artifacts to tracking lightweight evidence. The cleanup removed tracked
regenerable artifacts in these categories:

- cache tensors under `validation/cache/`;
- binary arrays and CSV data tables under `validation/outputs/**/data/`;
- repeated figures under `validation/outputs/**/figures/`;
- scratch logs under `validation/logs/`.

The retained evidence is the scripts, README files, markdown reports, summary
JSON/CSV where present, and command records. The deleted artifacts can be
regenerated from the script entry points listed above and from the topic-level
summaries under `validation/outputs/**`.
