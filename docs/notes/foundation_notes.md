# Foundation Notes

This repository currently implements only low-level algebraic pieces. It does
not run physical simulations or claim final numerical predictions.

## Qiu et al.

The normal-state Hamiltonian uses the four-orbital basis
`(dz1, dx1, dz2, dx2)` from Qiu et al. Appendix A:

`H(k) = [[H_parallel, H_perp], [H_perp, H_parallel]] - mu I`.

The implemented coefficients are the Appendix-A values for `Tz_k`, `Tx_k`,
`Tz_perp,k`, `Tx_perp,k`, `V_k`, and `V'_k`. The exchange/filling parameters
quoted in the main text are stored in `QiuExchangeParameters`:

- `mu = 0.05`
- `J_perp = 0.135 eV`
- `J_parallel = 0.084 eV`
- `J_xz = 0.03 eV`
- `J_H = 1 eV`
- `n_z = 0.8`, `n_x = 0.58`

The `s_pm` pairing follows the A1g structure in Eqs. (A6)-(A7). The simple
`d_wave` pairing is intentionally minimal and exists only as a comparison
channel for later theory work.

## Dai and Jiang

The Casimir utilities implement the process skeleton:

1. conductivity tensor at imaginary frequency,
2. rotation by plate/in-plane angle,
3. reflection matrix,
4. Lifshitz energy integrand,
5. torque integrand from `-partial_theta E`.

No Kubo conductivity calculation is implemented yet. The placeholder is explicit
so that future theory corrections can define the response without touching the
Casimir geometry code.
