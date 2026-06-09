# Peierls vertex Ward identity diagnostic

This is a vertex-level Peierls current vertex Ward identity check.
It is not a Pi_mu_nu response calculation.
It is not conductivity.
It is not Casimir.
The contact term is not involved in this vertex-level check.

Formula checked:
$\Gamma_i^P(k,q)= i \sum_R R_i t_R e^{i k\cdot R}\,\mathrm{sinc}(q\cdot R/2)$.
$q_x\Gamma_x^P+q_y\Gamma_y^P=H_0(k+q/2)-H_0(k-q/2)$.

run_command = `python validation/scripts/response/diagnose_peierls_vertex_ward_identity.py`
quick_mode=False
expanded_data_written=False
mesh_n=16
random_num=32
random_seed=12345

## Sign convention summary
- best sign_convention = plus
- minus: max relative error = 2, median relative error = 2
- plus: max relative error = 1.14316e-11, median relative error = 4.66038e-15

## Scope flags
response_computed=False
conductivity_computed=False
ward_identity_checked=True
casimir_computed=False
not_final_casimir_conclusion=True

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/peierls_vertex_ward_identity/data/peierls_vertex_ward_identity_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/peierls_vertex_ward_identity/figures/peierls_vertex_ward_error_vs_q.png
