# Stage 5.1 Response-to-conductivity convention audit

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no reflection / Casimir
- no Casimir-ready claim

## Existing conductivity-related code paths

| category | count |
| --- | --- |
| conductivity_related_files | 149 |
| existing_conductivity_helpers | 81 |
| existing_si_conversion_helpers | 56 |
| reflection_or_casimir_consumers | 114 |

## Response convention

$$\Pi_{\mu\nu}=\frac{\delta\langle J_\mu\rangle}{\delta a_\nu},\quad a_\nu=(\phi,A_x,A_y).$$

$$J=(\rho,j_x,j_y)=(\rho,-V_x,-V_y),\qquad P=(\rho,V_x,V_y).$$

The spatial block is interpreted as $\Pi_{ij}=\delta\langle j_i\rangle/\delta A_j$.

## Candidate conductivity conventions

| id | E/A relation | formula |
| --- | --- | --- |
| A_plus_xi | E_j(i xi)=+xi A_j(i xi) | sigma_ij(i xi)=Pi_ij(i xi)/xi |
| B_minus_xi | E_j(i xi)=-xi A_j(i xi) | sigma_ij(i xi)=-Pi_ij(i xi)/xi |
| C_iOmega | E_j(iOmega)=iOmega A_j(iOmega) | sigma_ij(iOmega)=Pi_ij(iOmega)/(iOmega) |

## Selected or ambiguous convention

| quantity | value |
| --- | --- |
| status | AMBIGUOUS |
| formula | CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE |
| reason | Existing local Kubo code computes sigma(i xi) directly, and response_units documents model-response-to-sheet normalization, but the finite-q physical Pi_ij to sigma_ij Euclidean E/A sign is not uniquely fixed by code alone. |

## Unit audit

| quantity | symbol | code variable | current unit | target unit | status |
| --- | --- | --- | --- | --- | --- |
| finite-q physical response | Pi_ij | response[1:3,1:3] | dimensionless model response from normalized BZ weights | conductivity kernel before SI sheet scaling | INFERRED |
| Matsubara energy | hbar xi_n | omega_eV | eV | eV for diagnostic conversion; rad/s for SI electrodynamics if needed | KNOWN |
| model conductivity | sigma_model | spatial_response_to_conductivity(...) | Pi_model / eV for candidates A/B or Pi_model/(i eV) for C | model sheet conductivity convention | AMBIGUOUS |
| SI sheet conductivity scaling | sigma_sheet | model_response_to_sheet_conductivity | model response | S | INFERRED |
| 2D versus 3D conductivity | sigma_2D / sigma_3D | SheetConductivityConvention | 2D sheet convention | reflection input needs sheet conductivity or dimensionless sheet normalization | KNOWN |
| explicit lattice geometry | a_parallel, layer spacing | lattice_constant_m, unit_cell_area_m2 | optional and inactive by default | needed only for future explicit 3D/bulk normalization | MISSING |

## Lightweight sanity check

| n | omega_eV | Pi_xx | Pi_yy | relative_offdiag | quad points |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.624329e-02 | -3.169999e-01-2.034158e-18j | -3.170924e-01+3.692941e-18j | 2.647158e-01 | 544 |
| 2 | 3.248658e-02 | -3.448624e-01+1.670675e-18j | -3.449146e-01-5.393035e-18j | 1.361614e-01 | 544 |
| 4 | 6.497316e-02 | -3.676981e-01+4.084667e-18j | -3.677445e-01+2.432113e-18j | 4.391329e-02 | 544 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| conductivity_convention_status | CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE |
| unit_status | UNIT_CHAIN_AMBIGUOUS |
| recommended_next_action | NOT_READY_NEEDS_CONVENTION_DECISION: Decide the Euclidean E/A convention and sheet-vs-bulk normalization before Stage 5.2 numerical conductivity sanity. |

## Recommended next step

Confirm the Euclidean electric-field/vector-potential convention and the sheet-vs-bulk normalization before Stage 5.2 numerical conductivity sanity. This audit is not reflection/Casimir input.
