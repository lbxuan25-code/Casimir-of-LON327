# Peierls contact vertex audit

This is a finite-q Peierls contact vertex audit.
It is not a Pi_mu_nu response calculation.
It is not conductivity.
It is not Casimir.
This result is a contact vertex-level validation only and is not yet connected to response.

Formula audited:
$\Lambda_{ij}^P(k,q)=-\sum_R R_iR_j t_R e^{i k\cdot R}\,\mathrm{sinc}(q\cdot R/2)^2$.
The minus sign follows from the existing convention H0(k)=sum_R t_R exp(i k.R), so q=0 gives d2H0/dk_i dk_j.
The formula comes from the second-order Peierls phase expansion along the same straight-bond hopping path.

run_command = `python validation/scripts/response/audit_peierls_contact_vertex.py`
quick_mode=False
expanded_data_written=False
mesh_n=24
random_num=64
random_seed=12345
q_list = 0 0.001 0.005 0.01 0.05 0.1 0.2 0.5 1
q_angle_list = 0 0.392699 0.785398 1.1781 1.5708
directions = xx xy yx yy

## Audit status
- q0_mass_limit_passed = True
- max_abs_q0_mass_error = 6.33005e-16
- max_rel_q0_mass_error = 3.87127e-15
- hermiticity_passed = True
- max_hermiticity_error = 5.1563e-16
- xy_yx_index_symmetry_passed = True
- max_index_symmetry_error = 0

## q_model trends
- max relative mass-limit error: q=0:3.87127e-15, q=0.001:4.89129e-06, q=0.005:0.000122282, q=0.01:0.000489126, q=0.05:0.0122262, q=0.1:0.0488803, q=0.2:0.195131, q=0.5:1.20262, q=1:4.5766
- max Hermiticity error: q=0:3.5774e-16, q=0.001:5.1563e-16, q=0.005:3.96914e-16, q=0.01:3.96702e-16, q=0.05:3.56662e-16, q=0.1:4.01737e-16, q=0.2:3.92646e-16, q=0.5:5.13384e-16, q=1:3.77248e-16
- max xy/yx index-symmetry error: q=0:0, q=0.001:0, q=0.005:0, q=0.01:0, q=0.05:0, q=0.1:0, q=0.2:0, q=0.5:0, q=1:0

## Scope flags
response_computed=False
conductivity_computed=False
ward_identity_checked=False
casimir_computed=False
not_final_finite_q_contact=True
not_final_finite_q_conductivity=True
not_final_casimir_conclusion=True

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/peierls_contact_vertex_audit/data/peierls_contact_vertex_audit_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/peierls_contact_vertex_audit/figures/peierls_contact_q0_mass_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/peierls_contact_vertex_audit/figures/peierls_contact_hermiticity_error_vs_q.png
