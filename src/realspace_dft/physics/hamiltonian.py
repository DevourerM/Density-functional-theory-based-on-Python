"""三维实空间 Kohn-Sham 哈密顿量的最小分项构造与矩阵自由表示。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..core.models import (
    ComplexArray,
    DensityGrid,
    InputConfig,
    KPoint,
    PotentialField,
    PseudopotentialInfo,
    RealSpaceGrid,
)
from .nonlocal_operator import NonlocalPseudopotentialOperator, 构造非局域赝势算符
from .xc import ExchangeCorrelationData, 计算交换关联数据


@dataclass(slots=True)
class KineticOperator:
    """固定 7 点有限差分的动能算符。"""

    grid: RealSpaceGrid
    kpoint: KPoint | None = None

    @property
    def stencil_point_count(self) -> int:
        """返回差分模板的点数。"""

        return 7

    def apply_to_grid(self, wavefunction_grid: ComplexArray) -> ComplexArray:
        """把动能项作用到单个或一组三维波函数上。"""

        psi = np.asarray(wavefunction_grid, dtype=np.complex128)
        step_u, step_v, step_w = self.grid.finite_difference_steps

        if self.kpoint is None:
            phase_u_forward = 1.0 + 0.0j
            phase_v_forward = 1.0 + 0.0j
            phase_w_forward = 1.0 + 0.0j
        else:
            reduced_kx, reduced_ky, reduced_kz = self.kpoint.fractional_coordinates
            phase_u_forward = np.exp(2.0j * np.pi * reduced_kx)
            phase_v_forward = np.exp(2.0j * np.pi * reduced_ky)
            phase_w_forward = np.exp(2.0j * np.pi * reduced_kz)

        def roll_with_bloch_phase(array: np.ndarray, shift: int, axis: int, forward_phase: complex) -> np.ndarray:
            rolled = np.roll(array, shift, axis=axis).astype(np.complex128, copy=False)
            if shift == -1:
                boundary_slice = [slice(None)] * rolled.ndim
                boundary_slice[axis] = -1
                rolled[tuple(boundary_slice)] *= forward_phase
            elif shift == 1:
                boundary_slice = [slice(None)] * rolled.ndim
                boundary_slice[axis] = 0
                rolled[tuple(boundary_slice)] *= np.conjugate(forward_phase)
            return rolled

        d2_u = (
            roll_with_bloch_phase(psi, -1, axis=0, forward_phase=phase_u_forward)
            - 2.0 * psi
            + roll_with_bloch_phase(psi, 1, axis=0, forward_phase=phase_u_forward)
        ) / (step_u * step_u)
        d2_v = (
            roll_with_bloch_phase(psi, -1, axis=1, forward_phase=phase_v_forward)
            - 2.0 * psi
            + roll_with_bloch_phase(psi, 1, axis=1, forward_phase=phase_v_forward)
        ) / (step_v * step_v)
        d2_w = (
            roll_with_bloch_phase(psi, -1, axis=2, forward_phase=phase_w_forward)
            - 2.0 * psi
            + roll_with_bloch_phase(psi, 1, axis=2, forward_phase=phase_w_forward)
        ) / (step_w * step_w)

        return -0.5 * (d2_u + d2_v + d2_w)


@dataclass(slots=True)
class HamiltonianComponents:
    """显式保存哈密顿量的各个组成项。"""

    kinetic: KineticOperator
    ionic_local_potential: PotentialField
    hartree_potential: PotentialField
    exchange_correlation_potential: PotentialField
    exchange_correlation_total_energy_hartree: float
    exchange_correlation_model: str
    effective_local_potential: PotentialField
    nonlocal_pseudopotential: NonlocalPseudopotentialOperator | None
    nonlocal_pseudopotential_note: str
    kpoint: KPoint


class HamiltonianOperator(LinearOperator):
    """总哈密顿算符的矩阵自由表示。"""

    def __init__(self, grid: RealSpaceGrid, components: HamiltonianComponents):
        self.grid = grid
        self.components = components
        super().__init__(dtype=np.complex128, shape=(grid.point_count, grid.point_count))

    def _matvec(self, wavefunction_vector: np.ndarray) -> np.ndarray:
        """实现 LinearOperator 所需的矩阵向量乘法。"""

        flattened_vector = np.asarray(wavefunction_vector, dtype=np.complex128).reshape(-1)
        psi_grid = self.grid.reshape_vector(flattened_vector)
        return self.grid.flatten_values(self.apply_to_grid(psi_grid))

    def _matmat(self, wavefunction_matrix: np.ndarray) -> np.ndarray:
        """实现分块矩阵向量乘法，供 LOBPCG 这类块迭代算法使用。"""

        psi_block = self.grid.reshape_wavefunctions(np.asarray(wavefunction_matrix, dtype=np.complex128))
        return self.grid.flatten_values(self.apply_to_grid(psi_block))

    def apply_to_grid(self, wavefunction_grid: ComplexArray) -> ComplexArray:
        """把总哈密顿量作用到单个或一组三维波函数上。"""

        psi = np.asarray(wavefunction_grid, dtype=np.complex128)
        kinetic_term = self.components.kinetic.apply_to_grid(psi)
        local_term = self.components.effective_local_potential.apply_to_grid(psi)
        nonlocal_term = 0.0
        if self.components.nonlocal_pseudopotential is not None:
            nonlocal_term = self.components.nonlocal_pseudopotential.apply_to_grid(psi)
        return kinetic_term + local_term + nonlocal_term

    def apply_components_to_grid(self, wavefunction_grid: ComplexArray) -> dict[str, ComplexArray]:
        """分别返回每个已实现哈密顿项对波函数的作用结果。"""

        psi = np.asarray(wavefunction_grid, dtype=np.complex128)
        components = {
            "动能项": self.components.kinetic.apply_to_grid(psi),
            "离子局域势项": self.components.ionic_local_potential.apply_to_grid(psi),
            "Hartree势项": self.components.hartree_potential.apply_to_grid(psi),
            "交换关联势项": self.components.exchange_correlation_potential.apply_to_grid(psi),
        }
        if self.components.nonlocal_pseudopotential is not None:
            components["非局域赝势项"] = self.components.nonlocal_pseudopotential.apply_to_grid(psi)
        return components



def _evaluate_radial_local_potential(
    distance_grid_bohr: np.ndarray,
    pseudopotential: PseudopotentialInfo,
) -> np.ndarray:
    """把径向局域赝势插值到三维网格上。"""

    if not pseudopotential.has_local_potential:
        raise ValueError(f"元素 {pseudopotential.element} 的局域赝势数据尚未载入。")

    radial_grid_bohr = np.asarray(pseudopotential.radial_grid_bohr, dtype=np.float64)
    local_potential_hartree = np.asarray(
        pseudopotential.local_potential_hartree,
        dtype=np.float64,
    )

    flat_distance = distance_grid_bohr.reshape(-1)
    flat_values = np.interp(
        flat_distance,
        radial_grid_bohr,
        local_potential_hartree,
        left=float(local_potential_hartree[0]),
        right=float(local_potential_hartree[-1]),
    )

    outside_mask = flat_distance > radial_grid_bohr[-1]
    if np.any(outside_mask):
        safe_distance = np.clip(flat_distance[outside_mask], 1.0e-12, None)
        flat_values[outside_mask] = -pseudopotential.z_valence / safe_distance

    return flat_values.reshape(distance_grid_bohr.shape)



def 构造离子局域势(
    real_space_grid: RealSpaceGrid,
    atom_sites,
    pseudopotentials: dict[str, PseudopotentialInfo],
) -> PotentialField:
    """由原子位置和局域赝势构造离子局域势。"""

    potential_values = np.zeros(real_space_grid.shape, dtype=np.float64)
    for atom_site in atom_sites:
        distance_grid_bohr = real_space_grid.minimum_image_distance_bohr(
            atom_site.fractional_position
        )
        potential_values += _evaluate_radial_local_potential(
            distance_grid_bohr,
            pseudopotentials[atom_site.element],
        )

    return PotentialField(
        name="离子局域势",
        values=potential_values,
        note="当前采用最邻近周期映像叠加 PP_LOCAL 局域赝势。",
        metadata={"原子数": len(atom_sites)},
    )



def 构造Hartree势(real_space_grid: RealSpaceGrid, density_grid: DensityGrid) -> PotentialField:
    """采用 FFT 泊松求解器构造周期性 Hartree 势。"""

    nx, ny, nz = real_space_grid.shape
    reciprocal = real_space_grid.reciprocal_lattice_matrix_bohr_inv

    k_u = np.fft.fftfreq(nx, d=1.0 / nx)
    k_v = np.fft.fftfreq(ny, d=1.0 / ny)
    k_w = np.fft.fftfreq(nz, d=1.0 / nz)
    ku_grid, kv_grid, kw_grid = np.meshgrid(k_u, k_v, k_w, indexing="ij")

    gx = (
        ku_grid * reciprocal[0, 0]
        + kv_grid * reciprocal[1, 0]
        + kw_grid * reciprocal[2, 0]
    )
    gy = (
        ku_grid * reciprocal[0, 1]
        + kv_grid * reciprocal[1, 1]
        + kw_grid * reciprocal[2, 1]
    )
    gz = (
        ku_grid * reciprocal[0, 2]
        + kv_grid * reciprocal[1, 2]
        + kw_grid * reciprocal[2, 2]
    )
    g2 = gx * gx + gy * gy + gz * gz

    density_fft = np.fft.fftn(density_grid.values)
    hartree_fft = np.zeros_like(density_fft, dtype=np.complex128)
    nonzero_mask = g2 > 1.0e-14
    hartree_fft[nonzero_mask] = 4.0 * np.pi * density_fft[nonzero_mask] / g2[nonzero_mask]

    hartree_potential = np.fft.ifftn(hartree_fft).real.astype(np.float64)
    return PotentialField(
        name="Hartree势",
        values=hartree_potential,
        note="采用周期性 FFT 泊松求解，G=0 分量置零。",
    )



def 构造交换关联势(
    real_space_grid: RealSpaceGrid,
    density_grid: DensityGrid,
    xc_functional: str,
) -> ExchangeCorrelationData:
    """根据当前电荷密度构造交换关联势与 XC 总能。"""

    return 计算交换关联数据(real_space_grid, density_grid, xc_functional)



def 构造哈密顿算符(
    config: InputConfig,
    real_space_grid: RealSpaceGrid,
    density_grid: DensityGrid,
    pseudopotentials: dict[str, PseudopotentialInfo],
    kpoint: KPoint | None = None,
    *,
    ionic_local_potential: PotentialField | None = None,
    nonlocal_pseudopotential: NonlocalPseudopotentialOperator | None = None,
) -> tuple[HamiltonianComponents, HamiltonianOperator]:
    """构造分项哈密顿量及其矩阵自由总算符。"""

    resolved_kpoint = kpoint or config.numerical.kpoints[0]
    kinetic = KineticOperator(real_space_grid, resolved_kpoint)
    resolved_ionic_local_potential = ionic_local_potential or 构造离子局域势(
        real_space_grid=real_space_grid,
        atom_sites=config.crystal.atom_sites,
        pseudopotentials=pseudopotentials,
    )
    hartree_potential = 构造Hartree势(real_space_grid, density_grid)
    exchange_correlation = 构造交换关联势(
        real_space_grid,
        density_grid,
        config.numerical.xc_functional,
    )
    resolved_nonlocal_pseudopotential = nonlocal_pseudopotential
    if resolved_nonlocal_pseudopotential is None:
        resolved_nonlocal_pseudopotential = 构造非局域赝势算符(
            real_space_grid=real_space_grid,
            atom_sites=config.crystal.atom_sites,
            pseudopotentials=pseudopotentials,
            kpoint=resolved_kpoint,
        )
    effective_local_potential = PotentialField(
        name="总局域有效势",
        values=(
            resolved_ionic_local_potential.values
            + hartree_potential.values
            + exchange_correlation.potential.values
        ),
        note="由离子局域势、Hartree 势和交换关联势相加得到。",
    )
    nonlocal_note = "当前体系未从赝势中解析出非局域 projector。"
    if resolved_nonlocal_pseudopotential is not None:
        nonlocal_note = (
            "已装配非局域 Kleinman-Bylander projector，总展开数 {}，k=({}, {}, {})。".format(
                resolved_nonlocal_pseudopotential.projector_count,
                *resolved_kpoint.fractional_coordinates,
            )
        )
    components = HamiltonianComponents(
        kinetic=kinetic,
        ionic_local_potential=resolved_ionic_local_potential,
        hartree_potential=hartree_potential,
        exchange_correlation_potential=exchange_correlation.potential,
        exchange_correlation_total_energy_hartree=exchange_correlation.total_energy_hartree,
        exchange_correlation_model=exchange_correlation.model_name,
        effective_local_potential=effective_local_potential,
        nonlocal_pseudopotential=resolved_nonlocal_pseudopotential,
        nonlocal_pseudopotential_note=nonlocal_note,
        kpoint=resolved_kpoint,
    )
    return components, HamiltonianOperator(real_space_grid, components)
