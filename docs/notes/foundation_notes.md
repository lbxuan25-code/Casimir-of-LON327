# Foundation Notes

This repository currently implements only low-level algebraic pieces. It does
not run physical simulations or claim final numerical predictions.

## Ground-State Model

The normal-state Hamiltonian uses the four-orbital basis
`(dz1, dx1, dz2, dx2)`:

`H(k) = [[H_parallel, H_perp], [H_perp, H_parallel]] - mu I`.

The implemented coefficients are the adopted values for `Tz_k`, `Tx_k`,
`Tz_perp,k`, `Tx_perp,k`, `V_k`, and `V'_k`. The exchange/filling parameters
are stored in `GroundStateExchangeParameters`:

- `mu = 0.05`
- `J_perp = 0.135 eV`
- `J_parallel = 0.084 eV`
- `J_xz = 0.03 eV`
- `J_H = 1 eV`
- `n_z = 0.8`, `n_x = 0.58`

The `s_pm` pairing follows the A1g structure in Eqs. (A6)-(A7). The simple
`d_wave` pairing is intentionally minimal and exists only as a comparison
channel for later theory work.

All Hamiltonian, pairing, velocity-vertex, and Kubo-response energies are in
eV. The velocity operator used by Kubo is `dH/dk_alpha`, also in eV because
`kx, ky` are dimensionless lattice momenta. Kubo conductivity multiplies the
dimensionless band response by `e^2/hbar` when SI output is requested. Bosonic
Matsubara energies are represented as `hbar xi_n = 2 pi n kBT` in eV.

## Dai and Jiang

The Casimir utilities implement the process skeleton:

1. conductivity tensor at imaginary frequency,
2. rotation by plate/in-plane angle,
3. reflection matrix,
4. Lifshitz energy integrand,
5. torque integrand from `-partial_theta E`.

The Kubo conductivity is defined as a band-basis imaginary-frequency response
using the ground-state velocity vertices. It is still intentionally low-level:
callers provide k-points and weights, so later theory corrections can change the
integration strategy without touching the Casimir geometry code.
