"""Exact-static validation commands.

Importing this command package installs the experimental paired-translation,
Ward-aware integrand used by the canonical static d-wave Gauss/adaptive route.
The library-level primitive evaluator and commensurate Ward commands remain
unchanged unless this command package is imported.
"""

from validation.lib.dwave_ward_aware_quadrature import (
    install_dwave_ward_aware_quadrature,
)

install_dwave_ward_aware_quadrature()
