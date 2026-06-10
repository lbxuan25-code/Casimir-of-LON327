# Best-convention spatial Ward term decomposition

This is Stage 4.5 diagnostic output.
It decomposes spatial Ward residual terms under the best Stage 4.4 convention candidates.
It is not final finite-q conductivity, not reflection/Casimir input, and not a material conclusion.

run_command = `python validation/scripts/response/decompose_best_convention_spatial_ward_terms.py`
quick_mode=False
response_computed=True
conductivity_computed=False
casimir_computed=False
normal_state_only=True
not_final_finite_q_conductivity=True
not_final_casimir_conclusion=True

## Convention cases
- physical_current_q_plus_contact_minus: current_vertex_multiplier=-1.0, contact_sign_convention=minus, ward_q_sign=1.0; physical current j_i=-delta H/delta A_i with Q_phys and contact minus.
- hamiltonian_vertex_q_minus_contact_minus: current_vertex_multiplier=1.0, contact_sign_convention=minus, ward_q_sign=-1.0; Hamiltonian derivative vertex Gamma_i^H with Q_H and contact minus.

All cases use Peierls current vertices. The contact-only block is extracted as Pi(finite_q_peierls, plus) - Pi(none), then assigned the case contact sign.

## Max residual and alpha by case
- physical_current_q_plus_contact_minus: max residual_abs = 0.0414927
- physical_current_q_plus_contact_minus: small-q residual alpha = 0.999993
- physical_current_q_plus_contact_minus: term_iomega_abs alpha = 1.0003
- physical_current_q_plus_contact_minus: term_bubble_abs alpha = 1.00073
- physical_current_q_plus_contact_minus: term_contact_abs alpha = 0.999995
- hamiltonian_vertex_q_minus_contact_minus: max residual_abs = 0.0414927
- hamiltonian_vertex_q_minus_contact_minus: small-q residual alpha = 0.999993
- hamiltonian_vertex_q_minus_contact_minus: term_iomega_abs alpha = 1.0003
- hamiltonian_vertex_q_minus_contact_minus: term_bubble_abs alpha = 1.00073
- hamiltonian_vertex_q_minus_contact_minus: term_contact_abs alpha = 0.999995

## Best case q trend: physical_current_q_plus_contact_minus
- q_model=0.001: max term_iomega_abs=3.49552e-05, max term_bubble_abs=0.000188881, max term_contact_abs=0.000590357, max density_bubble_partial_abs=0.00022295, max residual_abs=0.000416418
- q_model=0.005: max term_iomega_abs=0.000174802, max term_bubble_abs=0.000944837, max term_contact_abs=0.00295178, max density_bubble_partial_abs=0.00111481, max residual_abs=0.00208208
- q_model=0.01: max term_iomega_abs=0.000349834, max term_bubble_abs=0.00189241, max term_contact_abs=0.00590349, max density_bubble_partial_abs=0.0022301, max residual_abs=0.0041641
- q_model=0.05: max term_iomega_abs=0.0156024, max term_bubble_abs=0.0320134, max term_contact_abs=0.0295076, max density_bubble_partial_abs=0.0345741, max residual_abs=0.0208111
- q_model=0.1: max term_iomega_abs=0.0194106, max term_bubble_abs=0.0726336, max term_contact_abs=0.0589536, max density_bubble_partial_abs=0.0753469, max residual_abs=0.0414927

## Interpretation
- Best case by max residual_abs: physical_current_q_plus_contact_minus.
- Dominant leftover classification in the best case: contact_overcorrects.
- Contact role: contact partly reduces the residual coefficient but does not remove the O(q) leftover.
- Source diagnosis: The leftover is contact-sensitive, so contact normalization/factor/sign and equal-time terms should be checked next.
- Longitudinal max residual_abs = 0.0414927.
- Transverse max residual_abs = 9.40801e-17.
- Mixed-angle max residual_abs = 0.0294166.
- Longitudinal residual is larger than transverse.
- Left/right comparison: left/right residuals differ; check response index order and conjugation; max log-ratio deviation = 0.736135.

## Next-step rules
- If density_bubble_partial is already small but residual is contact-dominated, check contact factor/sign.
- If density_bubble_partial is O(q), check Kubo bubble sign, denominator, and matrix-element order.
- If contact cancels part of the residual but not enough, check equal-time/commutator term or contact normalization.
- If left/right differ strongly, check response index order and Hermitian conjugation.

## Output files
- compact_csv = `validation/outputs/response/best_convention_spatial_ward_terms/data/best_convention_spatial_ward_terms_compact.csv`

## Figures
- `validation/outputs/response/best_convention_spatial_ward_terms/figures/best_case_terms_vs_q.png`
- `validation/outputs/response/best_convention_spatial_ward_terms/figures/residual_vs_q_by_case.png`
- `validation/outputs/response/best_convention_spatial_ward_terms/figures/cancellation_ratios_vs_q.png`
- `validation/outputs/response/best_convention_spatial_ward_terms/figures/longitudinal_transverse_residuals.png`

## Explicit boundary
This stage is diagnostic only.
It is not final finite-q conductivity.
It is not reflection/Casimir input.
It is not a material conclusion.
