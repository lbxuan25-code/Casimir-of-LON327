# TB Fourier reconstruction audit

This is an H0(k) representation reconstruction audit.
It is not a response calculation, not conductivity, not a Ward identity check, and not Casimir.
The hopping/Fourier representation and the trigonometric representation are the same Hamiltonian written in equivalent forms.
The hopping/Fourier representation is not a new model and not a higher-precision model.

run_command = `python validation/scripts/response/audit_tb_fourier_reconstruction.py`
quick_mode=False
expanded_data_written=False
mesh_n=24
random_num=64
random_seed=12345

## Reconstruction status
- passed = True
- num_k_points = 640
- max absolute reconstruction error = 1.30491e-15
- max relative reconstruction error = 6.14328e-16
- max hopping Hermiticity error = 0

## Scope flags
response_computed=False
conductivity_computed=False
ward_identity_checked=False
casimir_computed=False
not_final_casimir_conclusion=True

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/tb_fourier_reconstruction_audit/data/tb_fourier_reconstruction_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/tb_fourier_reconstruction_audit/figures/tb_fourier_reconstruction_error.png
