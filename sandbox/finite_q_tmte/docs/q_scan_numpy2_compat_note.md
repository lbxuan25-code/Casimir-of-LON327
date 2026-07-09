# q scan NumPy 2 compatibility note

The q scan originally used `np.trapz` for the q-direction trapezoid diagnostic.

In NumPy 2.x this alias may be unavailable, so the q scan now uses a small local trapezoid accumulator instead of relying on `np.trapz` or `np.trapezoid`.

This change affects only diagnostic q-direction trapezoid summaries.  It does not change the response calculation, reflection calculation, phi scan, or Ward logic.

Status remains:

```text
q scan diagnostic only
not a full q/phi/n Casimir integral
not a torque calculation
valid_for_casimir_input: False
```
