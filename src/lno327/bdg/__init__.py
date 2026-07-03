"""Model-independent BdG response core helpers."""

from lno327.bdg.kinematics import MomentumTransfer, shifted_momenta
from lno327.bdg.hamiltonian import bdg_hamiltonian_from_blocks, bdg_hamiltonian_from_model_pairing
from lno327.bdg.finite_q import (
    bdg_block_diagonal_vertex,
    bdg_finite_q_vertex_from_normal_blocks,
    density_vertex,
    phase_phase_direct_vertex,
    phase_vertex,
)
from lno327.bdg.nambu import charge_current_vertex_from_model, diamagnetic_vertex_from_model
from lno327.bdg.spectrum import (
    bdg_eigensystem_from_model,
    diagonalize_hermitian,
    normal_eigensystem_from_model,
    transform_operator_to_band_basis,
)

__all__ = [
    "MomentumTransfer",
    "shifted_momenta",
    "bdg_hamiltonian_from_blocks",
    "bdg_hamiltonian_from_model_pairing",
    "bdg_block_diagonal_vertex",
    "bdg_finite_q_vertex_from_normal_blocks",
    "phase_vertex",
    "phase_phase_direct_vertex",
    "density_vertex",
    "charge_current_vertex_from_model",
    "diamagnetic_vertex_from_model",
    "diagonalize_hermitian",
    "normal_eigensystem_from_model",
    "bdg_eigensystem_from_model",
    "transform_operator_to_band_basis",
]
