"""非局域赝势 projector 的构造与作用。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.special import sph_harm as _scipy_spherical_harmonic

    def _复球谐函数(magnetic_index: int, angular_momentum: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
        return _scipy_spherical_harmonic(magnetic_index, angular_momentum, azimuth, polar)

except ImportError:
    from scipy.special import sph_harm_y as _scipy_spherical_harmonic_y

    def _复球谐函数(magnetic_index: int, angular_momentum: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
        return _scipy_spherical_harmonic_y(angular_momentum, magnetic_index, polar, azimuth)

from ..core.models import AtomSite, ComplexArray, PseudopotentialInfo, RealSpaceGrid


@dataclass(slots=True)
class AtomicNonlocalProjectorBlock:
    """保存单个原子的非局域 projector 展开。"""

    element: str
    projector_matrix: np.ndarray
    coupling_matrix_hartree: np.ndarray
    labels: tuple[str, ...]

    @property
    def projector_count(self) -> int:
        return int(self.projector_matrix.shape[1])


class NonlocalPseudopotentialOperator:
    """矩阵自由的非局域赝势算符。"""

    def __init__(self, real_space_grid: RealSpaceGrid, blocks: list[AtomicNonlocalProjectorBlock]):
        self.grid = real_space_grid
        self.blocks = tuple(blocks)
        self._diagonal_approximation = self._compute_diagonal_approximation()

    @property
    def projector_count(self) -> int:
        return sum(block.projector_count for block in self.blocks)

    def _compute_diagonal_approximation(self) -> np.ndarray:
        diagonal = np.zeros(self.grid.point_count, dtype=np.float64)
        for block in self.blocks:
            diagonal += np.einsum(
                "pi,ij,pj->p",
                block.projector_matrix.conj(),
                block.coupling_matrix_hartree,
                block.projector_matrix,
                optimize=True,
            ).real
        return diagonal

    def approximate_diagonal(self) -> np.ndarray:
        return self._diagonal_approximation.copy()

    def apply_to_grid(self, wavefunction_grid: ComplexArray) -> ComplexArray:
        psi = np.asarray(wavefunction_grid, dtype=np.complex128)
        if psi.ndim == 3:
            psi_matrix = self.grid.flatten_values(psi).reshape(self.grid.point_count, 1)
            is_single_state = True
        else:
            psi_matrix = self.grid.flatten_values(psi)
            is_single_state = False

        result_matrix = np.zeros_like(psi_matrix, dtype=np.complex128)
        for block in self.blocks:
            coefficients = self.grid.volume_element_bohr3 * (block.projector_matrix.conj().T @ psi_matrix)
            result_matrix += block.projector_matrix @ (block.coupling_matrix_hartree @ coefficients)

        if is_single_state:
            return self.grid.reshape_vector(result_matrix[:, 0])
        return self.grid.reshape_wavefunctions(result_matrix)

    def expectation_per_band(self, wavefunction_grids: ComplexArray) -> np.ndarray:
        psi = np.asarray(wavefunction_grids, dtype=np.complex128)
        if psi.ndim != 4:
            raise ValueError("非局域赝势能量期望需要一组波函数块。")
        applied = self.apply_to_grid(psi)
        psi_matrix = self.grid.flatten_values(psi)
        applied_matrix = self.grid.flatten_values(applied)
        return np.real(
            self.grid.volume_element_bohr3 * np.sum(psi_matrix.conj() * applied_matrix, axis=0)
        )


def _插值径向projector(distances: np.ndarray, pseudopotential: PseudopotentialInfo, projector_index: int) -> np.ndarray:
    projector = pseudopotential.nonlocal_projectors[projector_index]
    radial_grid = np.asarray(pseudopotential.radial_grid_bohr, dtype=np.float64)
    radial_values = np.asarray(projector.radial_values, dtype=np.float64)
    interpolated = np.interp(
        distances.reshape(-1),
        radial_grid,
        radial_values,
        left=float(radial_values[0]),
        right=0.0,
    ).reshape(distances.shape)
    return np.where(distances <= projector.cutoff_radius_bohr, interpolated, 0.0)


def _build_expanded_projector_block(
    real_space_grid: RealSpaceGrid,
    atom_site: AtomSite,
    pseudopotential: PseudopotentialInfo,
) -> AtomicNonlocalProjectorBlock | None:
    if not pseudopotential.has_nonlocal_projectors:
        return None

    dx, dy, dz = real_space_grid.minimum_image_displacements_bohr(atom_site.fractional_position)
    distances = np.sqrt(dx * dx + dy * dy + dz * dz)
    safe_distances = np.where(distances > 1.0e-14, distances, 1.0)
    polar_angle = np.arccos(np.clip(dz / safe_distances, -1.0, 1.0))
    azimuthal_angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)

    expanded_projectors: list[np.ndarray] = []
    expanded_labels: list[str] = []
    expanded_metadata: list[tuple[int, int, int]] = []
    for projector_index, projector in enumerate(pseudopotential.nonlocal_projectors):
        radial_values = _插值径向projector(distances, pseudopotential, projector_index)
        for magnetic_index in range(-projector.angular_momentum, projector.angular_momentum + 1):
            angular_part = _复球谐函数(
                magnetic_index,
                projector.angular_momentum,
                azimuthal_angle,
                polar_angle,
            )
            angular_part = np.where(distances > 1.0e-14, angular_part, 0.0)
            if projector.angular_momentum == 0:
                angular_part = np.full_like(angular_part, 1.0 / np.sqrt(4.0 * np.pi), dtype=np.complex128)
            projector_field = radial_values.astype(np.complex128) * angular_part.astype(np.complex128)
            expanded_projectors.append(projector_field.reshape(-1, order="C"))
            expanded_labels.append(
                f"{atom_site.element}_beta{projector.index}_l{projector.angular_momentum}_m{magnetic_index}"
            )
            expanded_metadata.append((projector_index, projector.angular_momentum, magnetic_index))

    projector_matrix = np.column_stack(expanded_projectors).astype(np.complex128)
    coupling_dimension = projector_matrix.shape[1]
    expanded_coupling = np.zeros((coupling_dimension, coupling_dimension), dtype=np.complex128)
    radial_coupling = np.asarray(pseudopotential.dij_matrix_hartree, dtype=np.float64)

    for row_index, (projector_i, angular_i, magnetic_i) in enumerate(expanded_metadata):
        for column_index, (projector_j, angular_j, magnetic_j) in enumerate(expanded_metadata):
            if angular_i == angular_j and magnetic_i == magnetic_j:
                expanded_coupling[row_index, column_index] = radial_coupling[projector_i, projector_j]

    return AtomicNonlocalProjectorBlock(
        element=atom_site.element,
        projector_matrix=projector_matrix,
        coupling_matrix_hartree=expanded_coupling,
        labels=tuple(expanded_labels),
    )


def 构造非局域赝势算符(
    real_space_grid: RealSpaceGrid,
    atom_sites: tuple[AtomSite, ...],
    pseudopotentials: dict[str, PseudopotentialInfo],
) -> NonlocalPseudopotentialOperator | None:
    """根据原子与赝势信息构造矩阵自由的非局域赝势算符。"""

    blocks: list[AtomicNonlocalProjectorBlock] = []
    for atom_site in atom_sites:
        block = _build_expanded_projector_block(
            real_space_grid,
            atom_site,
            pseudopotentials[atom_site.element],
        )
        if block is not None:
            blocks.append(block)

    if not blocks:
        return None
    return NonlocalPseudopotentialOperator(real_space_grid, blocks)
