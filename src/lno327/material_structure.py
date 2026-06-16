"""Material and structure conventions for LNO327 calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InPlaneLatticeConvention:
    """In-plane lattice convention for sheet-conductivity normalization."""

    name: str
    lattice_a_x_m: float
    lattice_a_y_m: float
    unit_cell_area_m2: float
    source_note: str
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        if self.lattice_a_x_m <= 0.0:
            raise ValueError("lattice_a_x_m must be positive")
        if self.lattice_a_y_m <= 0.0:
            raise ValueError("lattice_a_y_m must be positive")
        if self.unit_cell_area_m2 <= 0.0:
            raise ValueError("unit_cell_area_m2 must be positive")


LNO327_THIN_FILM_SLAO_IN_PLANE = InPlaneLatticeConvention(
    name="LNO327_thin_film_SrLaAlO4_clamped",
    lattice_a_x_m=3.754e-10,
    lattice_a_y_m=3.754e-10,
    unit_cell_area_m2=(3.754e-10) * (3.754e-10),
    source_note=(
        "Default in-plane lattice constant for coherently strained thin-film "
        "LNO327 / (La,Pr)327-type films on SrLaAlO4-like substrate. "
        "Use as a thin-film working value, not as relaxed bulk La3Ni2O7."
    ),
    is_placeholder=False,
)


LEGACY_PLACEHOLDER_IN_PLANE_385_PM = InPlaneLatticeConvention(
    name="legacy_placeholder_3p85_angstrom_square",
    lattice_a_x_m=3.85e-10,
    lattice_a_y_m=3.85e-10,
    unit_cell_area_m2=(3.85e-10) * (3.85e-10),
    source_note="Legacy placeholder/testing in-plane lattice scale; not the default LNO327 thin-film config.",
    is_placeholder=True,
)
