# Stage 5.1b Bilayer sheet conductivity convention

## Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_reflection_casimir: True
- not_casimir_ready_claim: True

## Selected convention

- electric_field_relation: E_j(i xi) = - xi A_j(i xi)
- response_to_conductivity_formula: sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV
- frequency_variable: omega_eV = hbar xi
- normalization: bilayer-normalized 2D sheet conductivity
- si_scaling_applied: False
- bulk_3d_conductivity: False
- single_layer_conductivity: False

## Analytic derivation

With real-time convention $f(t)\sim e^{-i\omega t}$, $E_j(\omega)=i\omega A_j(\omega)$ in transverse/optical gauge. Analytic continuation $\omega\to i\xi$ gives $E_j(i\xi)=-\xi A_j(i\xi)$. Since $j_i=\Pi_{ij}A_j=\sigma_{ij}E_j$, the model convention is $\sigma^{model}_{ij}(i\Omega)=-\Pi_{ij}(i\Omega)/\Omega_{eV}$.

## Bilayer-normalized 2D sheet interpretation

The response is computed from the full bilayer Hamiltonian, including interlayer hopping, hybridization, bonding/antibonding structure, and bilayer pairing information. The output is the in-plane sheet response of one bilayer unit, not a 3D bulk conductivity and not a single-layer conductivity. Final SI sheet scaling is not applied here.

## Synthetic check

- input_pi_spatial: [[-0.3, 0.01], [0.02, -0.4]]
- omega_eV: 0.02
- output_sigma_model: [[15, -0.5], [-1, 20]]
- diagonal_positive_for_negative_pi: True
- status: PASS

## Diagnostic decision

- conductivity_convention_status: CONVENTION_FIXED
- normalization_status: BILAYER_SHEET_MODEL_FIXED
- recommended_next_action: Proceed to Stage 5.2 numerical conductivity sanity scan; still do not enter reflection/Casimir.

## Next step

Proceed to Stage 5.2 numerical conductivity sanity scan; still do not enter reflection/Casimir.
