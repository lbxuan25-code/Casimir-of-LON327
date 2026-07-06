"""Normal-state density/current response prototypes for Ward diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from lno327.bdg.spectrum import diagonalize_hermitian
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.finite_q import add_band_bubble
from lno327.response.occupations import fermi_function

HamiltonianBuilder = Callable[[float, float], np.ndarray]
VelocityBuilder = Callable[[float, float, str], np.ndarray]
MassBuilder = Callable[[float, float, str, str], np.ndarray]


@dataclass(frozen=True)
class NormalDensityCurrentWorkspaceEntry:
    weight: float
    kx: float
    ky: float
    qx: float
    qy: float
    shared_eigenbasis_q0: bool
    energies_minus: np.ndarray
    energies_plus: np.ndarray
    states_minus: np.ndarray
    states_plus: np.ndarray
    occupations_minus: np.ndarray
    occupations_plus: np.ndarray
    observable_vertices_band: tuple[np.ndarray, ...]
    source_vertices_band: tuple[np.ndarray, ...]
    direct_contact_contribution: np.ndarray


@dataclass(frozen=True)
class NormalDensityCurrentWorkspace:
    spec: object
    k_points: np.ndarray
    k_weights: np.ndarray
    q: np.ndarray
    config: KuboConfig
    vertex_scheme: str
    contact_scheme: str
    contact_sign_convention: str
    hopping_terms: object
    shared_eigenbasis_q0: bool
    response_convention: str
    entries: tuple[NormalDensityCurrentWorkspaceEntry, ...]


def _validate_inputs(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    if weights is None:
        normalized_weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != (points.shape[0],):
            raise ValueError("k_weights must have shape (n,)")
    return points, normalized_weights, q_vector


def _finite_q_band_bubble_imag_axis(
    energies_minus: np.ndarray,
    states_minus: np.ndarray,
    energies_plus: np.ndarray,
    states_plus: np.ndarray,
    observable_vertices: Sequence[np.ndarray],
    source_vertices: Sequence[np.ndarray],
    config: KuboConfig,
) -> np.ndarray:
    if len(observable_vertices) != len(source_vertices):
        raise ValueError("observable_vertices and source_vertices must have the same length")
    occupations_minus = fermi_function(
        energies_minus,
        config.fermi_level_eV,
        config.temperature_eV,
    )
    occupations_plus = fermi_function(
        energies_plus,
        config.fermi_level_eV,
        config.temperature_eV,
    )
    observable_matrices = tuple(
        states_minus.conjugate().T @ vertex @ states_plus for vertex in observable_vertices
    )
    source_matrices = tuple(
        states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices
    )
    response = np.zeros((len(observable_vertices), len(source_vertices)), dtype=complex)
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            occupation_diff = float(occupations_minus[m] - occupations_plus[n])
            if occupation_diff == 0.0:
                continue
            denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
            factor = occupation_diff / denominator
            for mu, observable_matrix in enumerate(observable_matrices):
                for nu, source_matrix in enumerate(source_matrices):
                    response[mu, nu] += (
                        factor
                        * observable_matrix[m, n]
                        * np.conjugate(source_matrix[m, n])
                    )
    return response


def _hopping_terms_from_spec(spec, hopping_terms):
    return spec.hopping_terms() if hopping_terms is None else hopping_terms


def _normal_hamiltonian(spec, kx: float, ky: float, hopping_terms):
    if hopping_terms is None:
        return spec.normal_hamiltonian(kx, ky)
    return spec.normal_hamiltonian_from_hoppings(kx, ky, hopping_terms)


def _normal_expectation_from_bands(energies: np.ndarray, states: np.ndarray, vertex: np.ndarray, config: KuboConfig) -> complex:
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    vertex_in_band = states.conjugate().T @ vertex @ states
    return complex(np.sum(occupations * np.diag(vertex_in_band)))


def _compatible_workspace_config(workspace_config: KuboConfig, eval_config: KuboConfig) -> None:
    if float(workspace_config.temperature_eV) != float(eval_config.temperature_eV):
        raise ValueError("workspace config temperature_eV changed; rebuild the workspace")
    if float(workspace_config.fermi_level_eV) != float(eval_config.fermi_level_eV):
        raise ValueError("workspace config fermi_level_eV changed; rebuild the workspace")
    if bool(workspace_config.output_si) != bool(eval_config.output_si):
        raise ValueError("workspace config output_si changed; rebuild the workspace")


def _precompute_normal_density_current_workspace(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None,
    *,
    vertex_scheme: str,
    contact_scheme: str,
    contact_sign_convention: str,
    hopping_terms,
    response_convention: str,
) -> NormalDensityCurrentWorkspace:
    points, weights, q_vector = _validate_inputs(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    shared_eigenbasis_q0 = bool(qx == 0.0 and qy == 0.0)
    if vertex_scheme not in {"midpoint", "peierls"}:
        raise ValueError("vertex_scheme must be 'midpoint' or 'peierls'")
    if contact_scheme not in {"none", "q0_mass_diagnostic", "finite_q_peierls"}:
        raise ValueError("contact_scheme must be 'none', 'q0_mass_diagnostic', or 'finite_q_peierls'")
    if contact_sign_convention not in {"plus", "minus"}:
        raise ValueError("contact_sign_convention must be 'plus' or 'minus'")
    if response_convention not in {"diagnostic", "physical"}:
        raise ValueError("response_convention must be 'diagnostic' or 'physical'")
    peierls_terms = (
        _hopping_terms_from_spec(spec, hopping_terms)
        if vertex_scheme == "peierls" or contact_scheme == "finite_q_peierls" or hopping_terms is not None
        else None
    )
    sample_kx = float(points[0, 0])
    sample_ky = float(points[0, 1])
    orbital_dim = np.asarray(_normal_hamiltonian(spec, sample_kx, sample_ky, peierls_terms)).shape[0]
    rho = np.eye(orbital_dim, dtype=complex)
    directions = ("x", "y")
    entries: list[NormalDensityCurrentWorkspaceEntry] = []
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        if shared_eigenbasis_q0:
            bands_minus = bands_plus = diagonalize_hermitian(_normal_hamiltonian(spec, kx, ky, peierls_terms))
            bands_midpoint = bands_minus
        else:
            bands_minus = diagonalize_hermitian(
                _normal_hamiltonian(spec, kx - 0.5 * qx, ky - 0.5 * qy, peierls_terms)
            )
            bands_plus = diagonalize_hermitian(
                _normal_hamiltonian(spec, kx + 0.5 * qx, ky + 0.5 * qy, peierls_terms)
            )
            bands_midpoint = diagonalize_hermitian(_normal_hamiltonian(spec, kx, ky, peierls_terms))
        occupations_minus = fermi_function(bands_minus.energies, config.fermi_level_eV, config.temperature_eV)
        occupations_plus = fermi_function(bands_plus.energies, config.fermi_level_eV, config.temperature_eV)
        if vertex_scheme == "midpoint":
            vector_x = spec.velocity_operator(kx, ky, "x")
            vector_y = spec.velocity_operator(kx, ky, "y")
        else:
            vector_x = spec.peierls_hamiltonian_vector_vertex(
                kx,
                ky,
                qx,
                qy,
                "x",
                hopping_terms=peierls_terms,
            )
            vector_y = spec.peierls_hamiltonian_vector_vertex(
                kx,
                ky,
                qx,
                qy,
                "y",
                hopping_terms=peierls_terms,
            )
        if response_convention == "physical":
            observable_vertices = (rho, -vector_x, -vector_y)
            source_vertices = (rho, vector_x, vector_y)
        else:
            observable_vertices = (rho, vector_x, vector_y)
            source_vertices = observable_vertices
        observable_vertices_band = tuple(
            bands_minus.states.conjugate().T @ vertex @ bands_plus.states for vertex in observable_vertices
        )
        source_vertices_band = tuple(
            bands_minus.states.conjugate().T @ vertex @ bands_plus.states for vertex in source_vertices
        )
        direct_contact_contribution = np.zeros((3, 3), dtype=complex)
        if contact_scheme in {"q0_mass_diagnostic", "finite_q_peierls"}:
            sign = 1.0 if contact_sign_convention == "plus" else -1.0
            if response_convention == "physical":
                sign = -1.0
            for i, direction_i in enumerate(directions):
                for j, direction_j in enumerate(directions):
                    if contact_scheme == "q0_mass_diagnostic":
                        contact_matrix = spec.mass_operator(kx, ky, direction_i, direction_j)
                    else:
                        contact_matrix = spec.peierls_hamiltonian_contact_vertex(
                            kx,
                            ky,
                            qx,
                            qy,
                            direction_i,
                            direction_j,
                            hopping_terms=peierls_terms,
                        )
                    direct_contact_contribution[1 + i, 1 + j] += (
                        sign
                        * float(weight)
                        * _normal_expectation_from_bands(
                            bands_midpoint.energies,
                            bands_midpoint.states,
                            contact_matrix,
                            config,
                        )
                    )
        entries.append(
            NormalDensityCurrentWorkspaceEntry(
                weight=float(weight),
                kx=kx,
                ky=ky,
                qx=qx,
                qy=qy,
                shared_eigenbasis_q0=shared_eigenbasis_q0,
                energies_minus=bands_minus.energies,
                energies_plus=bands_plus.energies,
                states_minus=bands_minus.states,
                states_plus=bands_plus.states,
                occupations_minus=occupations_minus,
                occupations_plus=occupations_plus,
                observable_vertices_band=observable_vertices_band,
                source_vertices_band=source_vertices_band,
                direct_contact_contribution=direct_contact_contribution,
            )
        )
    return NormalDensityCurrentWorkspace(
        spec=spec,
        k_points=points,
        k_weights=weights,
        q=q_vector,
        config=config,
        vertex_scheme=vertex_scheme,
        contact_scheme=contact_scheme,
        contact_sign_convention=contact_sign_convention,
        hopping_terms=peierls_terms,
        shared_eigenbasis_q0=shared_eigenbasis_q0,
        response_convention=response_convention,
        entries=tuple(entries),
    )


def _normal_density_current_response_from_workspace_entries(
    workspace: NormalDensityCurrentWorkspace,
    config: KuboConfig,
) -> dict[str, np.ndarray]:
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    for entry in workspace.entries:
        add_band_bubble(
            bubble,
            entry.observable_vertices_band,
            entry.source_vertices_band,
            entry.energies_minus,
            entry.occupations_minus,
            entry.energies_plus,
            entry.occupations_plus,
            config.omega_eV,
            entry.weight,
            config=None,
            static_limit=False,
            prefactor=1.0,
        )
        direct += entry.direct_contact_contribution
    return {"bubble": bubble, "direct": direct, "total": bubble + direct}


def normal_density_current_response_imag_axis_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    *,
    vertex_scheme: str = "midpoint",
    contact_scheme: str = "none",
    contact_sign_convention: str = "plus",
    hopping_terms=None,
) -> np.ndarray:
    points, weights, q_vector = _validate_inputs(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    shared_eigenbasis_q0 = bool(qx == 0.0 and qy == 0.0)
    if vertex_scheme not in {"midpoint", "peierls"}:
        raise ValueError("vertex_scheme must be 'midpoint' or 'peierls'")
    if contact_scheme not in {"none", "q0_mass_diagnostic", "finite_q_peierls"}:
        raise ValueError("contact_scheme must be 'none', 'q0_mass_diagnostic', or 'finite_q_peierls'")
    if contact_sign_convention not in {"plus", "minus"}:
        raise ValueError("contact_sign_convention must be 'plus' or 'minus'")
    peierls_terms = (
        _hopping_terms_from_spec(spec, hopping_terms)
        if vertex_scheme == "peierls" or contact_scheme == "finite_q_peierls" or hopping_terms is not None
        else None
    )
    sample_kx = float(points[0, 0])
    sample_ky = float(points[0, 1])
    orbital_dim = np.asarray(_normal_hamiltonian(spec, sample_kx, sample_ky, peierls_terms)).shape[0]
    density_vertex = np.eye(orbital_dim, dtype=complex)
    response = np.zeros((3, 3), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        if shared_eigenbasis_q0:
            h_midpoint = _normal_hamiltonian(spec, kx, ky, peierls_terms)
            bands_minus = bands_plus = diagonalize_hermitian(h_midpoint)
        else:
            h_minus = _normal_hamiltonian(spec, kx - 0.5 * qx, ky - 0.5 * qy, peierls_terms)
            h_plus = _normal_hamiltonian(spec, kx + 0.5 * qx, ky + 0.5 * qy, peierls_terms)
            bands_minus = diagonalize_hermitian(h_minus)
            bands_plus = diagonalize_hermitian(h_plus)
        occupations_minus = fermi_function(
            bands_minus.energies,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        occupations_plus = fermi_function(
            bands_plus.energies,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        if vertex_scheme == "midpoint":
            vector_x = spec.velocity_operator(kx, ky, "x")
            vector_y = spec.velocity_operator(kx, ky, "y")
        else:
            vector_x = spec.peierls_hamiltonian_vector_vertex(
                kx,
                ky,
                qx,
                qy,
                "x",
                hopping_terms=peierls_terms,
            )
            vector_y = spec.peierls_hamiltonian_vector_vertex(
                kx,
                ky,
                qx,
                qy,
                "y",
                hopping_terms=peierls_terms,
            )
        vertices = (
            bands_minus.states.conjugate().T @ density_vertex @ bands_plus.states,
            bands_minus.states.conjugate().T @ vector_x @ bands_plus.states,
            bands_minus.states.conjugate().T @ vector_y @ bands_plus.states,
        )
        for m, energy_minus in enumerate(bands_minus.energies):
            for n, energy_plus in enumerate(bands_plus.energies):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = occupation_diff / denominator
                for mu in range(3):
                    for nu in range(3):
                        response[mu, nu] += (
                            weight
                            * factor
                            * vertices[mu][m, n]
                            * np.conjugate(vertices[nu][m, n])
                        )
        if contact_scheme in {"q0_mass_diagnostic", "finite_q_peierls"}:
            if shared_eigenbasis_q0:
                bands_midpoint = bands_minus
            else:
                h_midpoint = _normal_hamiltonian(spec, kx, ky, peierls_terms)
                bands_midpoint = diagonalize_hermitian(h_midpoint)
            occupations_midpoint = fermi_function(
                bands_midpoint.energies,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            sign = 1.0 if contact_sign_convention == "plus" else -1.0
            directions = ("x", "y")
            for i, direction_i in enumerate(directions):
                for j, direction_j in enumerate(directions):
                    if contact_scheme == "q0_mass_diagnostic":
                        contact_matrix = spec.mass_operator(kx, ky, direction_i, direction_j)
                    else:
                        contact_matrix = spec.peierls_hamiltonian_contact_vertex(
                            kx,
                            ky,
                            qx,
                            qy,
                            direction_i,
                            direction_j,
                            hopping_terms=peierls_terms,
                        )
                    band_contact = bands_midpoint.states.conjugate().T @ contact_matrix @ bands_midpoint.states
                    contact_value = np.sum(occupations_midpoint * np.diag(band_contact))
                    response[1 + i, 1 + j] += sign * weight * contact_value
    return response


def precompute_normal_density_current_workspace_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    *,
    vertex_scheme: str = "midpoint",
    contact_scheme: str = "none",
    contact_sign_convention: str = "plus",
    hopping_terms=None,
) -> NormalDensityCurrentWorkspace:
    return _precompute_normal_density_current_workspace(
        spec,
        k_points,
        config,
        q,
        k_weights,
        vertex_scheme=vertex_scheme,
        contact_scheme=contact_scheme,
        contact_sign_convention=contact_sign_convention,
        hopping_terms=hopping_terms,
        response_convention="diagnostic",
    )


def normal_density_current_response_imag_axis_from_workspace(
    workspace: NormalDensityCurrentWorkspace,
    config: KuboConfig | None = None,
) -> np.ndarray:
    eval_config = config or workspace.config
    _compatible_workspace_config(workspace.config, eval_config)
    return _normal_density_current_response_from_workspace_entries(workspace, eval_config)["total"]


def _normal_density_current_response_imag_axis_legacy_compatible(
    k_points,
    config,
    q,
    k_weights,
    hamiltonian,
    velocity,
    mass_operator,
    vertex_scheme,
    contact_scheme,
    contact_sign_convention,
) -> np.ndarray:
    points, weights, q_vector = _validate_inputs(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    if vertex_scheme not in {"midpoint", "peierls"}:
        raise ValueError("vertex_scheme must be 'midpoint' or 'peierls'")
    if contact_scheme not in {"none", "q0_mass_diagnostic", "finite_q_peierls"}:
        raise ValueError("contact_scheme must be 'none', 'q0_mass_diagnostic', or 'finite_q_peierls'")
    if contact_sign_convention not in {"plus", "minus"}:
        raise ValueError("contact_sign_convention must be 'plus' or 'minus'")
    density_vertex = np.eye(np.asarray(hamiltonian(float(points[0, 0]), float(points[0, 1]))).shape[0], dtype=complex)
    response = np.zeros((3, 3), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        bands_minus = diagonalize_hermitian(h_minus)
        bands_plus = diagonalize_hermitian(h_plus)
        occupations_minus = fermi_function(bands_minus.energies, config.fermi_level_eV, config.temperature_eV)
        occupations_plus = fermi_function(bands_plus.energies, config.fermi_level_eV, config.temperature_eV)
        vector_x = velocity(kx, ky, "x")
        vector_y = velocity(kx, ky, "y")
        vertices = (
            bands_minus.states.conjugate().T @ density_vertex @ bands_plus.states,
            bands_minus.states.conjugate().T @ vector_x @ bands_plus.states,
            bands_minus.states.conjugate().T @ vector_y @ bands_plus.states,
        )
        for m, energy_minus in enumerate(bands_minus.energies):
            for n, energy_plus in enumerate(bands_plus.energies):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = occupation_diff / denominator
                for mu in range(3):
                    for nu in range(3):
                        response[mu, nu] += (
                            weight
                            * factor
                            * vertices[mu][m, n]
                            * np.conjugate(vertices[nu][m, n])
                        )
        if contact_scheme == "q0_mass_diagnostic":
            bands_midpoint = diagonalize_hermitian(hamiltonian(kx, ky))
            occupations_midpoint = fermi_function(
                bands_midpoint.energies,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            sign = 1.0 if contact_sign_convention == "plus" else -1.0
            directions = ("x", "y")
            for i, direction_i in enumerate(directions):
                for j, direction_j in enumerate(directions):
                    contact_matrix = mass_operator(kx, ky, direction_i, direction_j)
                    band_contact = bands_midpoint.states.conjugate().T @ contact_matrix @ bands_midpoint.states
                    contact_value = np.sum(occupations_midpoint * np.diag(band_contact))
                    response[1 + i, 1 + j] += sign * weight * contact_value
    return response


def normal_density_current_response_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder | None = None,
    velocity: VelocityBuilder | None = None,
    mass_operator: MassBuilder | None = None,
    vertex_scheme: str = "midpoint",
    hopping_terms=None,
    contact_scheme: str = "none",
    contact_sign_convention: str = "plus",
) -> np.ndarray:
    """Diagnostic-only convention scanner.

    This diagnostic scanner may use Hamiltonian vector vertices directly.
    Do not use it as the main response path. For the physical-current
    candidate use ``normal_physical_density_current_response_imag_axis``.
    """

    if hamiltonian is not None or velocity is not None or mass_operator is not None:
        if hamiltonian is None or velocity is None or mass_operator is None:
            raise ValueError("hamiltonian, velocity, and mass_operator must be provided together")
        return _normal_density_current_response_imag_axis_legacy_compatible(
            k_points,
            config,
            q,
            k_weights,
            hamiltonian,
            velocity,
            mass_operator,
            vertex_scheme,
            contact_scheme,
            contact_sign_convention,
        )
    return normal_density_current_response_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        k_points,
        config,
        q,
        k_weights,
        vertex_scheme=vertex_scheme,
        contact_scheme=contact_scheme,
        contact_sign_convention=contact_sign_convention,
        hopping_terms=hopping_terms,
    )


def normal_physical_density_current_response_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder | None = None,
    hopping_terms=None,
) -> np.ndarray:
    return normal_physical_density_current_response_components_imag_axis(
        k_points,
        config,
        q,
        k_weights,
        hamiltonian,
        hopping_terms,
    )["total"]


def normal_physical_density_current_response_components_imag_axis_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    *,
    hopping_terms=None,
) -> dict[str, np.ndarray]:
    points, weights, q_vector = _validate_inputs(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    peierls_terms = _hopping_terms_from_spec(spec, hopping_terms)
    sample_kx = float(points[0, 0])
    sample_ky = float(points[0, 1])
    orbital_dim = np.asarray(_normal_hamiltonian(spec, sample_kx, sample_ky, peierls_terms)).shape[0]
    rho = np.eye(orbital_dim, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = _normal_hamiltonian(spec, kx - 0.5 * qx, ky - 0.5 * qy, peierls_terms)
        h_plus = _normal_hamiltonian(spec, kx + 0.5 * qx, ky + 0.5 * qy, peierls_terms)
        bands_minus = diagonalize_hermitian(h_minus)
        bands_plus = diagonalize_hermitian(h_plus)

        vector_x = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x", hopping_terms=peierls_terms)
        vector_y = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y", hopping_terms=peierls_terms)
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        bubble += weight * _finite_q_band_bubble_imag_axis(
            bands_minus.energies,
            bands_minus.states,
            bands_plus.energies,
            bands_plus.states,
            observable_vertices,
            source_vertices,
            config,
        )

        bands_midpoint = diagonalize_hermitian(_normal_hamiltonian(spec, kx, ky, peierls_terms))
        occupations_midpoint = fermi_function(
            bands_midpoint.energies,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        directions = ("x", "y")
        for i, direction_i in enumerate(directions):
            for j, direction_j in enumerate(directions):
                contact_matrix = spec.peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    hopping_terms=peierls_terms,
                )
                band_contact = bands_midpoint.states.conjugate().T @ contact_matrix @ bands_midpoint.states
                expect_mij = np.sum(occupations_midpoint * np.diag(band_contact))
                physical_direct_contact = -expect_mij
                direct[1 + i, 1 + j] += weight * physical_direct_contact
    return {"bubble": bubble, "direct": direct, "total": bubble + direct}


def normal_physical_bubble_ward_contribution_records_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    *,
    target_component: str = "left_current_x",
    hopping_terms=None,
) -> dict[str, object]:
    """Diagnostic-only per-k/band-pair contributions to a bubble Ward component."""
    points, weights, q_vector = _validate_inputs(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    peierls_terms = _hopping_terms_from_spec(spec, hopping_terms)
    sample_kx = float(points[0, 0])
    sample_ky = float(points[0, 1])
    orbital_dim = np.asarray(_normal_hamiltonian(spec, sample_kx, sample_ky, peierls_terms)).shape[0]
    rho = np.eye(orbital_dim, dtype=complex)
    coefficients = _ward_component_coefficients(target_component, float(config.omega_eV), qx, qy)
    grid_shape = _infer_rectangular_grid_shape(points)
    records: list[dict[str, object]] = []
    bubble = np.zeros((3, 3), dtype=complex)

    for flat_index, (weight, (kx_value, ky_value)) in enumerate(zip(weights, points, strict=True)):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = _normal_hamiltonian(spec, kx - 0.5 * qx, ky - 0.5 * qy, peierls_terms)
        h_plus = _normal_hamiltonian(spec, kx + 0.5 * qx, ky + 0.5 * qy, peierls_terms)
        bands_minus = diagonalize_hermitian(h_minus)
        bands_plus = diagonalize_hermitian(h_plus)
        occupations_minus = fermi_function(bands_minus.energies, config.fermi_level_eV, config.temperature_eV)
        occupations_plus = fermi_function(bands_plus.energies, config.fermi_level_eV, config.temperature_eV)
        vector_x = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x", hopping_terms=peierls_terms)
        vector_y = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y", hopping_terms=peierls_terms)
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        observable_matrices = tuple(
            bands_minus.states.conjugate().T @ vertex @ bands_plus.states for vertex in observable_vertices
        )
        source_matrices = tuple(
            bands_minus.states.conjugate().T @ vertex @ bands_plus.states for vertex in source_vertices
        )
        k_index = _grid_index_from_flat(flat_index, grid_shape)
        for m, energy_minus in enumerate(bands_minus.energies):
            for n, energy_plus in enumerate(bands_plus.energies):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = float(weight) * occupation_diff / denominator
                contribution_matrix = np.zeros((3, 3), dtype=complex)
                for mu, observable_matrix in enumerate(observable_matrices):
                    for nu, source_matrix in enumerate(source_matrices):
                        contribution_matrix[mu, nu] = (
                            factor
                            * observable_matrix[m, n]
                            * np.conjugate(source_matrix[m, n])
                        )
                bubble += contribution_matrix
                target_contribution = sum(
                    coefficient * contribution_matrix[mu, nu]
                    for mu, nu, coefficient in coefficients
                )
                vertex_products = [
                    observable_matrices[mu][m, n] * np.conjugate(source_matrices[nu][m, n])
                    for mu, nu, _coefficient in coefficients
                ]
                records.append(
                    {
                        "flat_k_index": int(flat_index),
                        "k_index": k_index,
                        "k": [kx, ky],
                        "k_plus_q": [kx + qx, ky + qy],
                        "band_pair": [int(m), int(n)],
                        "contribution": _complex_record(target_contribution),
                        "energy": {
                            "epsilon_m_k": float(energy_minus),
                            "epsilon_n_k_plus_q": float(energy_plus),
                            "energy_difference": float(energy_minus - energy_plus),
                            "abs_energy_difference": float(abs(energy_minus - energy_plus)),
                        },
                        "occupation": {
                            "f_m_k": float(occupations_minus[m]),
                            "f_n_k_plus_q": float(occupations_plus[n]),
                            "occupation_difference": occupation_diff,
                            "abs_occupation_difference": float(abs(occupation_diff)),
                        },
                        "denominator": _complex_record(denominator),
                        "vertices": {
                            "density_vertex_abs": float(
                                max(abs(observable_matrices[0][m, n]), abs(source_matrices[0][m, n]))
                            ),
                            "current_x_vertex_abs": float(
                                max(abs(observable_matrices[1][m, n]), abs(source_matrices[1][m, n]))
                            ),
                            "current_y_vertex_abs": float(
                                max(abs(observable_matrices[2][m, n]), abs(source_matrices[2][m, n]))
                            ),
                            "vertex_product_abs": float(max((abs(value) for value in vertex_products), default=0.0)),
                        },
                    }
                )

    left, right = _physical_ward_residuals_local(bubble, float(config.omega_eV), q_vector)
    target_total = _target_residual_component(left, right, target_component)
    return {
        "target_component": target_component,
        "q_model": [qx, qy],
        "omega_eV": float(config.omega_eV),
        "mesh_size": int(points.shape[0]),
        "total_residual_component": _complex_record(target_total),
        "total_residual_max_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
        "dominant_component": _dominant_ward_component_label(left, right),
        "records": records,
    }


def _ward_component_coefficients(
    target_component: str,
    omega_eV: float,
    qx: float,
    qy: float,
) -> tuple[tuple[int, int, complex], ...]:
    labels = ("density", "current_x", "current_y")
    if "_" not in target_component:
        raise ValueError("target_component must start with left_ or right_")
    side, component = target_component.split("_", 1)
    if component not in labels:
        raise ValueError("target_component component must be density, current_x, or current_y")
    index = labels.index(component)
    if side == "left":
        return ((0, index, 1j * omega_eV), (1, index, qx), (2, index, qy))
    if side == "right":
        return ((index, 0, 1j * omega_eV), (index, 1, -qx), (index, 2, -qy))
    raise ValueError("target_component must start with left_ or right_")


def _target_residual_component(left: np.ndarray, right: np.ndarray, target_component: str) -> complex:
    labels = ("density", "current_x", "current_y")
    side, component = target_component.split("_", 1)
    index = labels.index(component)
    return complex(left[index] if side == "left" else right[index])


def _physical_ward_residuals_local(response: np.ndarray, omega_eV: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qx, qy = (float(q[0]), float(q[1]))
    left = 1j * omega_eV * response[0, :] + qx * response[1, :] + qy * response[2, :]
    right = 1j * omega_eV * response[:, 0] - response[:, 1] * qx - response[:, 2] * qy
    return left, right


def _dominant_ward_component_label(left: np.ndarray, right: np.ndarray) -> str:
    labels = ("density", "current_x", "current_y")
    vector = np.concatenate([np.asarray(left), np.asarray(right)])
    index = int(np.argmax(np.abs(vector)))
    side = "left" if index < 3 else "right"
    return f"{side}_{labels[index % 3]}"


def _complex_record(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {
        "real": float(np.real(scalar)),
        "imag": float(np.imag(scalar)),
        "abs": float(abs(scalar)),
    }


def _infer_rectangular_grid_shape(points: np.ndarray) -> tuple[int, int] | None:
    unique_x = np.unique(np.round(points[:, 0], decimals=14))
    unique_y = np.unique(np.round(points[:, 1], decimals=14))
    if unique_x.size * unique_y.size != points.shape[0]:
        return None
    return int(unique_x.size), int(unique_y.size)


def _grid_index_from_flat(flat_index: int, grid_shape: tuple[int, int] | None) -> list[int] | None:
    if grid_shape is None:
        return None
    _nx, ny = grid_shape
    return [int(flat_index // ny), int(flat_index % ny)]


def precompute_normal_physical_density_current_workspace_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    *,
    hopping_terms=None,
) -> NormalDensityCurrentWorkspace:
    return _precompute_normal_density_current_workspace(
        spec,
        k_points,
        config,
        q,
        k_weights,
        vertex_scheme="peierls",
        contact_scheme="finite_q_peierls",
        contact_sign_convention="plus",
        hopping_terms=hopping_terms,
        response_convention="physical",
    )


def normal_physical_density_current_response_components_imag_axis_from_workspace(
    workspace: NormalDensityCurrentWorkspace,
    config: KuboConfig | None = None,
) -> dict[str, np.ndarray]:
    if workspace.response_convention != "physical":
        raise ValueError("workspace was not precomputed for the physical density-current convention")
    eval_config = config or workspace.config
    _compatible_workspace_config(workspace.config, eval_config)
    return _normal_density_current_response_from_workspace_entries(workspace, eval_config)


def normal_physical_density_current_response_imag_axis_from_workspace(
    workspace: NormalDensityCurrentWorkspace,
    config: KuboConfig | None = None,
) -> np.ndarray:
    return normal_physical_density_current_response_components_imag_axis_from_workspace(workspace, config)["total"]


def _normal_physical_components_legacy_compatible(k_points, config, q, k_weights, hamiltonian) -> dict[str, np.ndarray]:
    raise ValueError(
        "explicit hamiltonian fallback is not supported for the physical density-current response; "
        "use the default spec-driven path or a model spec with Peierls vertices"
    )


def normal_physical_density_current_response_components_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder | None = None,
    hopping_terms=None,
) -> dict[str, np.ndarray]:
    if hamiltonian is not None:
        return _normal_physical_components_legacy_compatible(k_points, config, q, k_weights, hamiltonian)
    return normal_physical_density_current_response_components_imag_axis_from_model(
        LNO327FourOrbitalSpec(),
        k_points,
        config,
        q,
        k_weights,
        hopping_terms=hopping_terms,
    )
