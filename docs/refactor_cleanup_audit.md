# Refactor Cleanup Audit

## Executive Summary

The main model, response, electrodynamics, collective, and Casimir building-block code has been split into clearer package boundaries, but the repository is not ready for old top-level module deletion yet.

The original audit found active `src/lno327` imports of old top-level modules, many regression tests that intentionally compare new code against old references, and a substantial validation/script layer that still imports old modules directly. A follow-up cleanup has since removed the selected active new-src imports of old `conductivity.py`, `bdg_response.py`, `finite_q_primitives.py`, and `response_conventions.py`; remaining selected old imports in `src/lno327` are the root compatibility facade or old reference modules themselves. Current response objects still correctly carry `valid_for_casimir_input=False`; this audit does not promote any response path to Casimir-ready status.

Priority counts from this audit:

| priority | count | summary |
| --- | ---: | --- |
| P0 | 1 | Root compatibility facade policy still blocks final deletion of several old modules. |
| P1 | 5 | Safe cleanup and compatibility debt before deletion. |
| P2 | 5 | Cleanup during deletion, mostly facade/test/metadata ergonomics. |
| P3 | 3 | Optional ergonomic cleanup after deletion. |

Immediate answer: do not start deleting old top-level modules as a batch. The safest first deletion candidates are old modules with no active new-src dependencies and only regression/validation references, such as `nonlocal_response.py`, `bdg_nonlocal_response.py`, `tb_fourier.py`, `ward_response.py`, `ward_validation.py`, `reflection_input.py`, and old `casimir.py`, but only after the tests and validation/scripts that still reference them are migrated or explicitly archived. For `conductivity.py`, `bdg_response.py`, `finite_q_primitives.py`, and `response_conventions.py`, the remaining deletion blocker inside `src/lno327` is now the root compatibility facade or old reference files, not active new modules.

## Current Migration Status

The new structure is active in these areas:

| area | current owner | status |
| --- | --- | --- |
| Models | `src/lno327/models/` | Four-orbital and two-band specs exist; hopping/Peierls support moved into model packages. |
| BdG helpers | `src/lno327/bdg/` | Nambu, spectrum, finite-q vertex helpers are active. |
| Response core | `src/lno327/response/` | Local, nonlocal, finite-q, occupation, static-policy, and validation helpers exist. |
| Collective/Ward | `src/lno327/collective/` | Active Ward validation and Schur helpers exist. |
| Electrodynamics | `src/lno327/electrodynamics/` | Conductivity, unit conventions, and reflection adapters exist. |
| Casimir building blocks | `src/lno327/casimir/` | New package is active for `lno327.casimir`; old `src/lno327/casimir.py` remains as regression reference. |

Important status flags remain conservative:

| path | relevant status |
| --- | --- |
| `src/lno327/response/local_interface.py` | `LocalSheetResponse.valid_for_casimir_input=False` |
| `src/lno327/response/static_policy.py` | `StaticResponseResult.valid_for_casimir_input=False` |
| `src/lno327/response/finite_q_bdg.py` | metadata keeps `valid_for_casimir_input=False` |
| `src/lno327/casimir/__init__.py` | `casimir_layer_metadata()["valid_for_casimir_input"] is False` |
| `validation/lib/finite_q_diagnostics.py` | raises if diagnostic finite-q response is marked Casimir-ready |

## Old Top-Level Module Inventory

Legend:

| label | meaning |
| --- | --- |
| `active_src_blocker` | Active `src/lno327` code outside the old file itself still imports this old top-level module. |
| `regression_reference` | Tests intentionally import old modules or load old files to compare behavior. |
| `historical_validation_dependency` | Validation scripts/libs still import old modules. |
| `external_script_dependency` | User-facing scripts still import old modules. |

| old_file | new_owner | active_src_refs | test_refs | validation_refs | script_refs | delete_readiness | blockers | recommended_action |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `src/lno327/conductivity.py` | `response/config.py`, `response/occupations.py`, `response/local_normal.py`, `electrodynamics/conductivity.py`, `numerics/` | root/old refs only | 23 | 38 | 1 | high risk | Root facade, old reference modules, many tests and validation scripts. | P0: decide root facade strategy; then migrate validation/scripts and old-reference tests. |
| `src/lno327/bdg_response.py` | `response/local_bdg.py`, `bdg/nambu.py`, `response/containers.py` | root/old refs only | 9 | 1 | 0 | high risk | Root facade and old reference modules; several regression tests. | P0: decide root facade strategy; then migrate old-reference tests. |
| `src/lno327/nonlocal_response.py` | `response/nonlocal_normal.py` | 0 | 1 | 2 | 0 | medium | Regression test and numerical-stability scripts. | P1/P2: migrate tests and scripts, then delete. |
| `src/lno327/bdg_nonlocal_response.py` | `response/nonlocal_bdg.py` | 0 | 1 | 1 | 0 | medium | Regression test and one diagnostic script. | P1/P2: migrate or archive diagnostics, then delete. |
| `src/lno327/finite_q_primitives.py` | `bdg/finite_q.py`, `response/finite_q.py`, `response/finite_q_bdg.py`, model Peierls specs | root facade only | 8 | 2 | 0 | high risk | Root facade, finite-q regression tests, and validation scripts. | P0: decide root facade strategy; then migrate tests and validation scripts. |
| `src/lno327/tb_fourier.py` | `models/lno327_four_orbital/peierls.py`, `models/hopping.py` | 0 | 4 | 11 | 0 | medium | Validation scripts and regression tests. | P1: migrate validation imports to model package Peierls APIs; keep old tests until migrated. |
| `src/lno327/ward_response.py` | `collective/ward.py`, `response/normal_density_current.py` | 0 | 2 | 22 | 0 | medium/high | Heavy validation dependency. | P1: migrate/retire validation scripts before deletion; keep old file until diagnostics are explicitly archived. |
| `src/lno327/ward_validation.py` | `collective/validation.py` | 0 | 3 | 2 | 0 | medium | Regression tests and finite-q validation libs. | P1: migrate validation lib imports; then delete with tests adjusted. |
| `src/lno327/response_interface.py` | `response/local_interface.py` | 1 | 2 | 0 | 0 | high risk | Root facade still imports it. | P0: stop exporting old response-interface objects from root facade or move compatibility to explicit legacy namespace. |
| `src/lno327/static_response.py` | `response/static_policy.py` | 1 | 2 | 0 | 0 | high risk | Root facade still imports it. | P0: stop root facade dependency after downstream API plan is set. |
| `src/lno327/response_conventions.py` | `electrodynamics/conventions.py`, `electrodynamics/units.py` | root/old refs only | 6 | 7 | 1 | high risk | Root facade, old reference modules, validation/scripts and tests. | P0/P1: decide root facade strategy; then migrate validation/scripts to electrodynamics modules. |
| `src/lno327/reflection_input.py` | `electrodynamics/reflection.py` | 0 | 3 | 4 | 1 | medium | Regression tests, validation scripts, and Casimir pipeline script. | P1/P2: migrate scripts to `electrodynamics.reflection`; then delete. |
| `src/lno327/casimir.py` | `casimir/setup.py`, `casimir/reflection.py`, `casimir/lifshitz.py`, `casimir/torque.py` | 0 old-file refs | 1 path-loaded old reference plus active package tests | 0 old-file refs | 0 old-file refs | low/medium | New `lno327.casimir` references resolve to package, not old file; old file is still loaded by path in regression. | P2: after regression no longer needs path reference, delete old file. |

The raw audit script reports `lno327.casimir` references under the `casimir` row, but those imports now resolve to the new package. Manual classification above treats only direct file-path loading of `src/lno327/casimir.py` as old-reference usage.

## Active Import Boundary Audit

Static commands found active old-module imports in these places:

| old boundary | active src locations | classification |
| --- | --- | --- |
| `from .conductivity` / `lno327.conductivity` | `src/lno327/__init__.py`, `finite_q_quadrature.py`, `gap_analysis.py`, `normal_sampling.py`, `models/lno327_four_orbital/collective.py` | active_src_blocker |
| `from .bdg_response` | `src/lno327/__init__.py`, `bdg_q0_conventions.py` | active_src_blocker |
| `from .finite_q_primitives` | `src/lno327/__init__.py`, `bdg_q0_conventions.py` | active_src_blocker |
| `from .response_interface` | `src/lno327/__init__.py` | active_src_blocker |
| `from .static_response` | `src/lno327/__init__.py` | active_src_blocker |
| `from .response_conventions` | `src/lno327/__init__.py`, `normal_sampling.py` | active_src_blocker |
| `lno327.nonlocal_response`, `lno327.bdg_nonlocal_response`, `lno327.tb_fourier`, `lno327.ward_response`, `lno327.ward_validation`, `lno327.reflection_input` | no active new-src imports found outside old files themselves | regression_reference and historical_validation_dependency |

Resolved follow-up item: `src/lno327/models/lno327_four_orbital/collective.py` now imports `fermi_function` from `lno327.response.occupations`.

Root `src/lno327/__init__.py` is still a broad compatibility facade and imports several old modules. That is acceptable as a transitional facade, but it blocks file deletion.

## Public API and __init__ Surface Audit

| file | public names | exports too wide? | mixes generic and LNO327-specific? | exports old reference objects? | circular import risk | recommended strategy |
| --- | ---: | --- | --- | --- | --- | --- |
| `src/lno327/api.py` | 39 | medium | yes, but intentionally user-facing | no direct old module imports found | medium | Keep as stable convenience API; avoid adding more internals. |
| `src/lno327/__init__.py` | 73 | high | yes | yes, via old top-level modules | high | Treat as historical compatibility facade; later shrink or split into explicit legacy imports. |
| `src/lno327/response/__init__.py` | 53 | high | mostly generic response, but very broad | no old top-level imports found | medium | Prefer explicit submodule imports; consider exporting only config, containers, and primary public entry points. |
| `src/lno327/electrodynamics/__init__.py` | 39 | medium/high | mostly generic electrodynamics | no | low/medium | Keep core tensor/unit/reflection exports; avoid metadata-test workarounds. |
| `src/lno327/collective/__init__.py` | 13 | reasonable | collective/Ward specific | no | low | Acceptable. |
| `src/lno327/casimir/__init__.py` | 6 | narrow | Casimir building blocks only | no; old file is path-loaded only in tests | low | Acceptable; metadata clearly says not Casimir-ready. |
| `src/lno327/models/__init__.py` | 3 | narrow | generic base only | no | low | Good boundary. |

Specific findings:

- `response/__init__.py` is useful for tests but too broad for long-term public API. It exposes core kernels, local/nonlocal response, finite-q assembly, and static policy together.
- `api.py` should continue as a stable convenience API, but internal code should keep using explicit submodule imports.
- `electrodynamics.conventions.SheetConductivityConversion` is currently a hand-written `__slots__` class because dependency boundary tests made the direct `valid_for_casimir_input` field awkward. This is P2 API ergonomics debt.
- `electrodynamics.conventions` still has `"valid_for_" + "casi" + "mir_input"`. This is P2 style/static-test debt. Boundary tests should forbid imports/references to `lno327.casimir`, not ordinary metadata strings.

## Performance and Repeated-Work Audit

| file | function | suspected_repetition | loop_context | cost_level | is_physics_required | safe_optimization | risk | recommended_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `src/lno327/response/local_bdg.py` | `bdg_local_total_kernel_imag_axis` | Paramagnetic and diamagnetic paths each call `bdg_local_eigensystem_from_model`, so the same k-point BdG Hamiltonian can be diagonalized twice. | Local BdG total kernel over k-points. | high | no | Add a shared per-k eigensystem iterator used by para and dia. | needs_regression | P1 |
| `src/lno327/response/local_normal.py` | `kubo_conductivity_imag_axis_from_model`, `kubo_conductivity_real_axis_from_model` | Imaginary and real-axis functions separately diagonalize if both are requested for same mesh/config. | User may compute both axes in scripts/tests. | medium | no | Optional cache keyed by `(spec identity, kx, ky, config occupation params)`, or expose eigensystem precompute helper. | needs_regression | P3 |
| `src/lno327/response/nonlocal_normal.py` | `normal_current_current_kernel_imag_axis_from_model` | q=0 branch shares one eigensystem; finite-q branch necessarily has minus/plus eigensystems. | Nonlocal loop over k-points. | medium | partly | Current q=0 sharing is good; possible optimization is shared midpoint velocity construction. | safe | P3 |
| `src/lno327/response/nonlocal_bdg.py` | `bdg_current_current_kernel_imag_axis_from_model` | Always diagonalizes minus and plus; unlike normal path, no q=0 shared-eigenbasis shortcut. | Nonlocal BdG loop. | high | no for q=0 | Add q=0 shared basis path similar to finite-q engine and normal nonlocal response. | needs_regression | P1 |
| `src/lno327/response/finite_q_bdg.py` | `finite_q_bdg_response_from_model_ansatz` | q=0 shares eigensystem; finite q computes minus/plus once and reuses across multiple bubbles. Vertices are rebuilt once per k. | Main model-driven finite-q BdG loop. | high | mostly yes | Current sharing is good. Possible future cache for Peierls vector/contact vertices within k-loop. | physics_sensitive | P2 |
| `src/lno327/response/normal_density_current.py` | `_finite_q_band_bubble_imag_axis` plus callers | `normal_density_current_response_imag_axis_from_model` computes `occupations_minus/plus`, then helper recomputes occupations internally. | Diagnostic finite-q normal density/current loop. | medium | no | Pass occupations into helper or remove unused precompute. | safe | P1 |
| `src/lno327/response/normal_density_current.py` | `_normal_physical_components_legacy_compatible` | Explicit `hamiltonian` fallback uses zero current vertices and zero direct contact, so it is not equivalent to old Ward behavior. | Legacy compatibility path. | medium | no | Either implement full legacy velocity/contact support or remove fallback with a clear error. | needs_regression | P1 |
| `src/lno327/response/normal_density_current.py` | `normal_physical_density_current_response_components_imag_axis_from_model` | For each k, diagonizes k-q/2, k+q/2, and midpoint for direct term. | Physical normal density/current loop. | high | midpoint direct term is required | Cache midpoint bands if q=0 or if direct and bubble share same Hamiltonian in a future formulation. | physics_sensitive | P2 |
| `src/lno327/finite_q_engine.py` | `finite_q_bdg_response_from_ansatz` | Constructs `LNO327FourOrbitalSpec(pairing_amplitudes=amp)` per public call, not per k. | Public wrapper. | low | acceptable | No action unless high-volume caller constructs repeatedly. | safe | P3 |
| `src/lno327/response/local_interface.py` | `local_response_imag_axis` | Constructs one `LNO327FourOrbitalSpec` per call. | Public local response call. | low | acceptable | Optional `spec` injection if scripts batch many calls. | safe | P3 |
| `src/lno327/models/lno327_four_orbital/spec.py` | `__init__` | Caches hopping terms once; avoids repeated `normal_state_hopping_terms()`. | Spec construction. | low | good | Keep. | safe | P3 |
| `src/lno327/models/symmetry_bdg_2band/spec.py` | `__init__` | Caches hopping terms once; avoids repeated generation. | Spec construction. | low | good | Keep. | safe | P3 |
| `src/lno327/models/hopping.py` | Peierls helpers | Hopping terms are passed explicitly; no obvious active repeated generation. | Model Peierls calls. | medium | no | Keep explicit tuple/list passing. | safe | P3 |
| `src/lno327/casimir/` | Lifshitz/torque functions | Torque finite difference evaluates energy twice by definition. | Single integrand evaluation. | low | yes | No optimization unless adding analytic torque later, which would be physics-sensitive. | physics_sensitive | P3 |
| `validation/scripts/response/stage4_16_full_response_adaptive_ward_diagnostic.py` | adaptive Ward integration helpers | Repeated diagonalizations and occupation calculations in validation-only loops. | Historical validation scans. | high | partly | Leave until validation scripts are migrated or archived; not active library path. | needs_regression | P2 |
| `validation/scripts/bdg_finite_q/normal_finite_q_ward_audit.py` | large diagnostic class/functions | Repeated hopping-term and diagonalization patterns. | Historical finite-q Ward audit. | high | partly | Archive or migrate to response helpers before deletion. | needs_regression | P2 |

Known issues confirmed:

1. `normal_density_current.py` still recomputes occupations inside `_finite_q_band_bubble_imag_axis`; classify as P1 safe cleanup.
2. `_normal_physical_components_legacy_compatible` still uses zero vector vertices for explicit Hamiltonian fallback; classify as P1 compatibility debt. It must not be presented as equivalent to old `ward_response.py`.
3. `electrodynamics.conventions` still contains the `"casi" + "mir"` metadata workaround; classify as P2 style/static-test debt.
4. `SheetConductivityConversion` is still not a dataclass; classify as P2 API ergonomics debt.
5. `response/__init__.py` exports 53 names and is too broad for long-term API stability.
6. `api.py` remains a mixed compatibility/user-facing convenience surface, but it no longer imports old response/conductivity modules directly.

## Validation Scripts and Historical Outputs Audit

| area | finding | action |
| --- | --- | --- |
| `validation/scripts/response/` | Many historical response scripts import `lno327.conductivity`, `lno327.tb_fourier`, `lno327.ward_response`, `lno327.response_conventions`, and `lno327.reflection_input`. | Migrate scripts only after active library deletion blockers are resolved; preserve scripts with physical diagnostic value. |
| `validation/scripts/bdg_finite_q/` | Finite-q Ward scripts still import old `conductivity`, `finite_q_primitives`, `tb_fourier`, `ward_response`, and `ward_validation`. | Keep until finite-q validation contracts are moved to new `response`/`collective` APIs. |
| `validation/lib/finite_q_diagnostics.py` | Imports old `conductivity` and `ward_validation`. | P1 migration target because scripts depend on it. |
| `validation/scripts/numerical_stability/` | Several diagnostic scripts import old response/conductivity modules. | Archive or migrate after deciding which diagnostics are still active. |
| `scripts/casimir/finite_q_bdg_casimir_pipeline.py` | Imports old `conductivity`, `reflection_input`, and `response_conventions`. | P1 if pipeline remains user-facing; do not modify in this audit. |
| `scripts/casimir/local_response_integral.py` | Imports new `lno327.casimir` package for Matsubara frequency; this is not an old `casimir.py` dependency. | No deletion blocker for old `casimir.py`. |
| historical numbered script names | Many validation scripts have historical numbered names. | Do not rename during deletion; consider archival directory later. |
| tracked validation outputs | 34 files under `validation/outputs/` are tracked. | Keep as historical artifacts for now; later replace large run outputs with small summaries/contracts if desired. |

Validation scripts that seem replaced or partly replaced by tests include response convention/unit conversion, electrodynamics reflection adapters, finite-q engine boundary checks, local/nonlocal response regression, and Ward metadata regression. Scripts that still have diagnostic value include adaptive Ward scans, finite-q Ward audits, real-material reflection grids, and numerical-stability probes.

## Test Suite Audit

| test_file | purpose | depends_on_old_reference | can_remove_after_old_deletion | can_merge_with | priority |
| --- | --- | --- | --- | --- | --- |
| `tests/test_response_core_occupations.py` | New occupation helpers vs old `conductivity.py`. | yes | yes | `tests/test_response_core_occupations.py` can become pure new-behavior tests. | P2 |
| `tests/test_electrodynamics_conductivity_tools.py` | New electrodynamics conductivity vs old conductivity tools. | yes | partly | Keep new behavior tests; remove old comparison rows. | P2 |
| `tests/test_numerics_grids_matsubara.py` | Numerics grid/Matsubara helpers vs old conductivity utilities. | yes | partly | Keep numerics-only tests. | P2 |
| `tests/test_response_config.py` | `KuboConfig` migration regression. | yes | yes | Merge into response config tests. | P2 |
| `tests/test_response_local_normal_regression.py` | New local normal response vs old conductivity. | yes | yes | Keep smaller structural tests. | P2 |
| `tests/test_response_local_bdg_regression.py` | New local BdG response vs old BdG response. | yes | yes | Keep local BdG behavior tests. | P2 |
| `tests/test_response_nonlocal_normal_regression.py` | New nonlocal normal response vs old nonlocal response. | yes | yes | Keep nonlocal behavior tests. | P2 |
| `tests/test_response_nonlocal_bdg_regression.py` | New nonlocal BdG response vs old BdG nonlocal response. | yes | yes | Keep nonlocal BdG behavior tests. | P2 |
| `tests/test_response_finite_q_primitives_regression.py` | New finite-q primitives vs old finite-q primitives. | yes | yes | Merge into finite-q core tests. | P2 |
| `tests/test_bdg_finite_q_primitives_regression.py` | Legacy finite-q primitive regression. | yes | yes | Merge with model Peierls/finite-q core tests. | P2 |
| `tests/test_lno327_four_orbital_peierls_regression.py` | Model Peierls vs old `tb_fourier.py`. | yes | yes | Keep model Peierls invariants. | P2 |
| `tests/test_collective_validation_regression.py` | New collective validation vs old `ward_validation.py`. | yes | yes | Keep collective validation behavior tests. | P2 |
| `tests/test_collective_ward_metadata_regression.py` | Ward metadata vs old finite-q primitive metadata. | yes | yes | Merge into collective/Ward metadata tests. | P2 |
| `tests/test_response_static_policy_regression.py` | Static policy vs old static response. | yes | yes | Keep static-policy behavior tests. | P2 |
| `tests/test_response_local_interface_regression.py` | Local interface vs old response interface. | yes | yes | Keep local interface behavior tests. | P2 |
| `tests/test_electrodynamics_units_regression.py` | New unit helpers vs old response conventions. | yes | partly | Keep unit conversion tests. | P2 |
| `tests/test_electrodynamics_conventions_regression.py` | New conventions vs old response conventions. | yes | partly | Keep convention tests. | P2 |
| `tests/test_electrodynamics_reflection_regression.py` | New reflection adapter vs old reflection input. | yes | yes | Keep adapter behavior tests. | P2 |
| `tests/test_casimir_package_regression.py` | New Casimir package vs old `casimir.py` path-loaded reference. | yes | yes | Keep package smoke tests after old file deletion. | P2 |
| `tests/test_*dependency_boundaries.py` | Static dependency boundaries. | no old execution, but many string checks | no | Consolidate once deletion is complete. | P2 |

The test suite has many dependency-boundary tests. They are useful during migration, but some are string-fragile and encourage unnatural source text such as `"casi" + "mir"`. Later boundary tests should focus on import statements and module paths, not metadata words.

Full pytest was green before this audit work and must be rerun after adding audit files.

## Package Size and Consolidation Audit

| metric | count |
| --- | ---: |
| `src/lno327` Python files | 100 |
| `src/lno327` packages (`__init__.py`) | 11 |
| `tests` Python files | 97 |
| `validation/scripts` Python files | 67 |
| `scripts` Python files | 13 |
| tracked files under `validation/outputs` | 34 |

File growth mainly comes from:

- model package split (`models/lno327_four_orbital`, `models/symmetry_bdg_2band`, shared model helpers);
- generic response core split (`response/`, `bdg/`, `collective/`);
- electrodynamics split (`conductivity`, units, conventions, reflection);
- Casimir package split;
- migration regression tests and dependency boundary tests.

Reasonable splits to keep:

- `models/`, `bdg/`, `response/`, `collective/`, `electrodynamics/`, and `casimir/` should not be merged back.
- Model-specific Peierls code should remain inside model packages.
- `response/occupations.py`, `response/bubble.py`, and `bdg/spectrum.py` are good small generic units.

Likely deletion/consolidation after cleanup:

- old top-level reference modules listed in the inventory;
- old-vs-new regression tests once old modules are deleted;
- duplicated boundary tests after imports are stable;
- historical validation scripts that are superseded by tests and have no active diagnostic role.

## Immediate Risks

| risk | severity | detail | recommended mitigation |
| --- | --- | --- | --- |
| Deleting old modules now breaks active src imports. | high | Root facade and several active modules still import old paths. | Clear P0 active imports first. |
| Validation/scripts may silently diverge if old modules are removed without migration. | high | Many validation scripts still import old modules directly. | Migrate or archive scripts before deletion. |
| `normal_density_current` legacy fallback can be mistaken for old Ward behavior. | medium | Explicit Hamiltonian fallback uses zero current vertices. | Document and then fix/remove fallback with regression. |
| Broad root facade masks old dependency usage. | medium | `src/lno327/__init__.py` imports old modules. | Split compatibility facade from stable `api.py`. |
| String-based boundary tests distort source style. | low/medium | `casi` + `mir` workaround exists. | Refine boundary tests to check imports rather than metadata words. |

## Deletion Readiness Table

| old_file | readiness | first required action |
| --- | --- | --- |
| `src/lno327/casimir.py` | closest to deletion | Remove path-loaded regression dependency after new package behavior is trusted. |
| `src/lno327/nonlocal_response.py` | close, validation blocked | Migrate one regression test and two numerical-stability scripts. |
| `src/lno327/bdg_nonlocal_response.py` | close, validation blocked | Migrate one regression test and one diagnostic script. |
| `src/lno327/tb_fourier.py` | validation blocked | Migrate validation scripts to model Peierls APIs. |
| `src/lno327/ward_response.py` | validation blocked | Migrate/retire Ward validation scripts. |
| `src/lno327/ward_validation.py` | validation blocked | Migrate validation lib and tests to `collective.validation`. |
| `src/lno327/reflection_input.py` | validation/script blocked | Migrate scripts to `electrodynamics.reflection`. |
| `src/lno327/response_interface.py` | active root facade blocked | Stop root facade import; migrate tests. |
| `src/lno327/static_response.py` | active root facade blocked | Stop root facade import; migrate tests. |
| `src/lno327/bdg_response.py` | active src blocked | Migrate `bdg_q0_conventions.py` and root facade. |
| `src/lno327/finite_q_primitives.py` | active src blocked | Migrate `bdg_q0_conventions.py` and root facade. |
| `src/lno327/response_conventions.py` | active src and scripts blocked | Migrate `normal_sampling.py`, root facade, validation/scripts. |
| `src/lno327/conductivity.py` | most blocked | Migrate active src imports, root facade, tests, validation/scripts, and user-facing script. |

## Recommended Cleanup Plan

### P0: must fix before deletion

1. Decide root facade policy for `src/lno327/__init__.py`: either keep as compatibility facade until the very end or stop importing old modules before deletion.
2. Keep active new-src boundary tests in place so old `conductivity.py`, `bdg_response.py`, `finite_q_primitives.py`, and `response_conventions.py` imports do not return outside the root facade and old reference modules.
3. Before deletion, migrate or retire old reference modules that still import each other internally.
4. Before deletion, migrate validation/scripts and old-vs-new regression tests.

### P1: safe cleanup before deletion

1. Migrate validation lib and high-value validation scripts to new `response`, `collective`, `electrodynamics`, and model package APIs.
2. Migrate `scripts/casimir/finite_q_bdg_casimir_pipeline.py` away from old `conductivity.py`, `reflection_input.py`, and `response_conventions.py` if it remains user-facing.
3. Continue performance cleanup only with regression tests; `response/local_bdg.py` now shares para/dia eigensystems in the total-kernel path, and `response/nonlocal_bdg.py` now shares the q=0 eigensystem.
4. Keep `_normal_physical_components_legacy_compatible` disabled with a clear error unless a full explicit-Hamiltonian physical-current implementation is restored.
5. Audit diagnostic `normal_density_current_response_imag_axis_from_model` separately before changing its occupation calculations; its current precomputed occupations are used by its local loop.

### P2: cleanup during deletion

1. Delete old-vs-new regression tests or convert them to pure new-behavior tests as each old module is removed.
2. Consolidate dependency-boundary tests after imports stabilize.
3. Refine boundary tests so metadata strings like `casimir` are allowed while imports of forbidden modules are not.
4. Restore `SheetConductivityConversion` as a frozen dataclass after tests are adjusted.
5. Archive or clearly label historical validation scripts and tracked outputs that are no longer active validation contracts.

### P3: optional ergonomic cleanup

1. Split `api.py` documentation into stable user-facing convenience API vs explicit internal submodule imports.
2. Narrow `response/__init__.py` exports to core types and primary entry points.
3. Add optional precomputed eigensystem/cache APIs for high-volume scripts, without changing formulas.

Recommendation: do not immediately start broad deletion. Start with P0 active-src import migrations, then migrate validation/scripts, then delete the lowest-risk old files in small batches with targeted regression removal.

## Commands Run

Static audit commands run during this audit:

```bash
python tools/audit_refactor_debt.py
rg 'lno327\.(conductivity|bdg_response|nonlocal_response|bdg_nonlocal_response|finite_q_primitives|tb_fourier|ward_response|ward_validation|response_interface|static_response|response_conventions|reflection_input)' src/lno327 tests validation scripts
rg 'from \.(conductivity|bdg_response|nonlocal_response|bdg_nonlocal_response|finite_q_primitives|tb_fourier|ward_response|ward_validation|response_interface|static_response|response_conventions|reflection_input)' src/lno327 tests validation scripts
rg 'np\.linalg\.eigh|diagonalize_hermitian|normal_eigensystem_from_model|bdg_eigensystem_from_model|fermi_function|negative_fermi_derivative|hopping_terms\(|normal_state_hopping_terms\(|LNO327FourOrbitalSpec\(|SymmetryBdG2BandSpec\(|PairingAmplitudes\(|spec\.normal_hamiltonian|spec\.bdg_hamiltonian|spec\.velocity_operator|spec\.mass_operator|spec\.peierls_hamiltonian_vector_vertex|spec\.peierls_hamiltonian_contact_vertex|transform_operator_to_band_basis' src/lno327 tests validation scripts
rg 'valid_for_casimir_input|Casimir-ready|casimir_ready|casi.*mir' src/lno327 tests validation scripts
rg 'from lno327\.casimir|import lno327\.casimir|from \.casimir' src/lno327 tests validation scripts
git ls-files validation/outputs
```

Counts collected:

```text
src/lno327 python files: 100
src/lno327 packages: 11
tests python files: 97
validation/scripts python files: 67
scripts python files: 13
tracked validation output files: 34
```

Test commands run after adding the audit files:

```bash
python -m pytest tests/test_refactor_cleanup_audit.py
# 3 passed

python -m pytest
# 508 passed, 16 warnings
```

The warnings are the existing finite-q global phase-correction runtime warnings; no physics implementation was changed by this audit.
