"""Normal-state density/current response prototypes for Ward diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lno327.bdg.spectrum import diagonalize_hermitian
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.occupations import fermi_function

HamiltonianBuilder = Callable[[float, float], np.ndarray]
VelocityBuilder = Callable[[float, float, str], np.ndarray]
MassBuilder = Callable[[float, float, str, str], np.ndarray]


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
