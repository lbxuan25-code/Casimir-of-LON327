# LNO327 Casimir Torque Foundations

Low-level Python scaffolding for studying whether Casimir torque can distinguish
`s_pm` and `d_wave` superconducting pairing symmetry in La3Ni2O7.

Current scope:

- Qiu et al. four-orbital bilayer normal-state Hamiltonian.
- Seed `s_pm` and simple `d_wave` pairing matrices.
- BdG matrix assembly.
- Conductivity tensor rotation and anisotropy helpers.
- Dai/Jiang-style reflection matrix plus Casimir energy/torque integrands.
- Smoke-test scripts and pytest coverage.

This is intentionally not a numerical simulation layer yet.

Run tests:

```bash
pytest
```

Inspect one momentum point:

```bash
python scripts/inspect_qiu_blocks.py --kx 0.0 --ky 0.0
```
