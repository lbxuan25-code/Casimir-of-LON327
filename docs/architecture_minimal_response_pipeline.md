# Minimal Response Pipeline Architecture

This document records the current structural split between model inputs and
the generic response pipeline. The refactor preserves the existing minimal
separable fixed-form-factor pairing model; it does not introduce new physics
corrections.

## Model Input Layer

The model input layer provides:

- normal-state Hamiltonian `H0(k)`;
- pairing ansatz `Delta(k)`;
- collective vertices for retained order-parameter fields;
- Hubbard-Stratonovich / Goldstone counterterm provider;
- metadata describing gauge and Ward conventions.

The finite-q BdG pipeline consumes these inputs through `PairingAnsatz`.
`pairing_ansatz.py` and `pairing_bonds.py` are the model-input /
order-parameter-vertex preprocessing layer, not the response engine.
The currently retained collective fields are only:

- `eta1`: amplitude;
- `eta2`: phase.

For this minimal model, the pairing remains separable with fixed form factors.
For nonlocal pairing diagnostics, the endpoint gauge form factor is available as
the gauge-aware finite-q form factor. Endpoint gauge, midpoint, and symmetric
`k±q/2` phase vertices are all input-layer choices.

Bond-resolved extra collective modes are intentionally not part of this
refactor. The omitted `xi`-like modes are also not introduced here.

## Generic Finite-q Engine

The standard generic engine lives in `lno327.finite_q_engine`. It accepts a pairing
ansatz and computes:

- BdG eigensystems at `k-q/2` and `k+q/2`;
- density/current bubbles;
- direct/contact terms;
- EM-collective mixed kernels;
- collective kernels and counterterms;
- Schur-complement responses.

The engine is not supposed to branch on pairing names such as `spm` or `dwave`.
Pairing-specific structure belongs in `lno327.pairing_ansatz`.

Generic low-level numerical primitives shared by the engine and the legacy
facade live in `lno327.finite_q_primitives`. The dependency direction is:

`pairing_ansatz` -> model preprocessing
`finite_q_primitives` -> shared generic helpers
`finite_q_engine` -> primitives plus ansatz
`bdg_finite_q_response` -> compatibility facade around the engine

The legacy public function `bdg_finite_q_response_imag_axis` remains available
as a compatibility wrapper. It builds the appropriate ansatz and delegates the
generic calculation to the engine. Its default `phase_vertex="symmetric_kpm"`
is retained for legacy compatibility; new ansatz-based workflows should choose
the phase vertex explicitly.

## Ward Validation Layer

Ward validation is diagnostic-only. The module `lno327.ward_validation` reports
left and right Ward residuals, norms, pass/fail status, tolerance, and convention
metadata.

Ward validation must not:

- mutate the response tensor;
- apply LSQ or response-level fitting;
- choose new counterterms;
- repair a response to make it pass.

## Casimir Layer

The Casimir layer must consume only response tensors that are explicitly marked
as validated and unit-converted. Raw local kernels and raw finite-q diagnostic
BdG responses are not Casimir-ready.

The finite-q diagnostic response metadata keeps `valid_for_casimir_input=False`
unless all of the following have been made explicit:

- Ward validation has passed;
- unit conversion policy is documented and applied;
- Matsubara `n=0` policy is documented and applied.

This refactor does not promote any current finite-q output to formal Casimir
input. Raw local q=0 responses and raw finite-q diagnostic responses remain
`valid_for_casimir_input=False`.

## Production Policy

The current production model is the minimal separable pairing ansatz. LSQ
corrections and response-level fitting are not allowed in production. They may
appear only in diagnostic validation scripts as references for identifying
missing structure.

The refactor is architectural: it preserves existing formulas for pairing,
collective vertices, Goldstone counterterm construction, and Schur complements
while making ownership boundaries explicit.
