# normal-subtracted smoke-pilot figures

These figures are pure post-processing outputs derived from the existing runtime CSV files.

- raw Delta E = E_pairing - E_normal may contain an angle-independent energy offset
- angle-independent offsets do not produce torque
- plotted energy is δΔE = ΔE - <ΔE>_θ
- torque is Δτ = -∂θδΔE
- subtracting <ΔE>_θ leaves torque and max-min anisotropy amplitudes unchanged relative to raw ΔE
- therefore `excess_torque_vs_angle.png` and `excess_anisotropy_amplitude_vs_distance.png` can have the same numerical curves as raw-ΔE-derived plots; their labels document that the torque-producing angular part is used
- theta derivatives use radians
- torque is computed by applying numpy.gradient(theta_rad) to δΔE after subtracting the angular mean
- endpoint values use raw finite-difference one-sided gradients and are not strict physical endpoint torque claims
- smoke-pilot diagnostic only
- valid_for_formal_casimir_claim = false
- reference distance = 50 nm

Generated figures:

- `excess_anisotropic_energy_vs_angle.png`
- `excess_torque_vs_angle.png`
- `excess_anisotropy_amplitude_vs_distance.png`

Optional extra raw ΔE figure is generated only with `--plot-raw-excess-energy`:

- `extra/raw_excess_energy_vs_angle.png`
