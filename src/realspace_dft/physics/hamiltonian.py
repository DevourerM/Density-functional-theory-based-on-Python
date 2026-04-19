"""三维实空间 Kohn-Sham 哈密顿量的分项构造与矩阵自由表示。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ..core.models import (
    ComplexArray,
    DensityGrid,
    InputConfig,
    PotentialField,
    PseudopotentialInfo,
    RealSpaceGrid,
)
from .nonlocal_operator import NonlocalPseudopotentialOperator, 构造非局域赝势算符
from .xc import ExchangeCorrelationData, 计算交换关联数据


def _periodic_first_derivative_matrix(point_count: int, step: float) -> sparse.csr_matrix:
    """构造周期性边界条件下的一阶中心差分矩阵。"""

    matrix = sparse.lil_matrix((point_count, point_count), dtype=np.float64)
    coefficient = 0.5 / step
    for index in range(point_count):
        matrix[index, (index + 1) % point_count] = coefficient
        matrix[index, (index - 1) % point_count] = -coefficient
    return matrix.tocsr()


def _periodic_second_derivative_matrix(point_count: int, step: float) -> sparse.csr_matrix:
    """构造周期性边界条件下的二阶中心差分矩阵。"""

    matrix = sparse.lil_matrix((point_count, point_count), dtype=np.float64)
    coefficient = 1.0 / (step * step)
    for index in range(point_count):
        matrix[index, index] = -2.0 * coefficient
        matrix[index, (index + 1) % point_count] = coefficient
        matrix[index, (index - 1) % point_count] = coefficient
    return matrix.tocsr()


def _kron3(
    matrix_x: sparse.spmatrix,
    matrix_y: sparse.spmatrix,
    matrix_z: sparse.spmatrix,
) -> sparse.csr_matrix:
    """构造三维 Kronecker 乘积，默认使用 C-order 展平顺序。"""

    return sparse.kron(
        sparse.kron(matrix_x, matrix_y, format="csr"),
        matrix_z,
        format="csr",
    )


@dataclass(slots=True)
class KineticOperator:
    """矩阵自由的动能算符，同时支持按需导出稀疏矩阵。"""

    grid: RealSpaceGrid

    @property
    def stencil_point_count(self) -> int:
        """返回差分模板的点数。"""

        metric = self.grid.metric_contravariant
        off_diagonal = metric - np.diag(np.diag(metric))
        return 19 if np.any(np.abs(off_diagonal) > 1.0e-14) else 7

    @property
    def diagonal_value(self) -> float:
        """返回动能算符在标准基上的近似对角元。"""

        step_u, step_v, step_w = self.grid.finite_difference_steps
        metric = self.grid.metric_contravariant
        return float(
            metric[0, 0] / (step_u * step_u)
            + metric[1, 1] / (step_v * step_v)
            + metric[2, 2] / (step_w * step_w)
        )

    def approximate_diagonal(self) -> np.ndarray:
        """返回供预条件器使用的动能对角近似。"""

        return np.full(self.grid.point_count, self.diagonal_value, dtype=np.float64)

    def apply_to_grid(self, wavefunction_grid: ComplexArray) -> ComplexArray:
        """把动能项作用到单个或一组三维波函数上。"""

        psi = np.asarray(wavefunction_grid, dtype=np.complex128)
        metric = self.grid.metric_contravariant
        step_u, step_v, step_w = self.grid.finite_difference_steps

        d2_u = (
            np.roll(psi, -1, axis=0) - 2.0 * psi + np.roll(psi, 1, axis=0)
        ) / (step_u * step_u)
        d2_v = (
            np.roll(psi, -1, axis=1) - 2.0 * psi + np.roll(psi, 1, axis=1)
        ) / (step_v * step_v)
        d2_w = (
            np.roll(psi, -1, axis=2) - 2.0 * psi + np.roll(psi, 1, axis=2)
        ) / (step_w * step_w)

        d2_uv = (
            np.roll(np.roll(psi, -1, axis=0), -1, axis=1)
            - np.roll(np.roll(psi, -1, axis=0), 1, axis=1)
            - np.roll(np.roll(psi, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(psi, 1, axis=0), 1, axis=1)
        ) / (4.0 * step_u * step_v)
        d2_uw = (
            np.roll(np.roll(psi, -1, axis=0), -1, axis=2)
            - np.roll(np.roll(psi, -1, axis=0), 1, axis=2)
            - np.roll(np.roll(psi, 1, axis=0), -1, axis=2)
            + np.roll(np.roll(psi, 1, axis=0), 1, axis=2)
        ) / (4.0 * step_u * step_w)
        d2_vw = (
            np.roll(np.roll(psi, -1, axis=1), -1, axis=2)
            - np.roll(np.roll(psi, -1, axis=1), 1, axis=2)
            - np.roll(np.roll(psi, 1, axis=1), -1, axis=2)
            + np.roll(np.roll(psi, 1, axis=1), 1, axis=2)
        ) / (4.0 * step_v * step_w)

        laplacian = (
            metric[0, 0] * d2_u
            + metric[1, 1] * d2_v
            + metric[2, 2] * d2_w
            + 2.0 * metric[0, 1] * d2_uv
            + 2.0 * metric[0, 2] * d2_uw
            + 2.0 * metric[1, 2] * d2_vw
        )
        return -0.5 * laplacian

    def apply_to_vector(self, wavefunction_vector: np.ndarray) -> np.ndarray:
        """把动能项作用到展平的一维波函数向量上。"""

        psi_grid = self.grid.reshape_vector(np.asarray(wavefunction_vector, dtype=np.complex128))
        return self.grid.flatten_values(self.apply_to_grid(psi_grid))

    def apply_to_matrix(self, wavefunction_matrix: np.ndarray) -> np.ndarray:
        """把动能项作用到列向量堆叠的波函数块上。"""

        psi_block = self.grid.reshape_wavefunctions(np.asarray(wavefunction_matrix, dtype=np.complex128))
        return self.grid.flatten_values(self.apply_to_grid(psi_block))

    def to_sparse_matrix(self) -> sparse.csr_matrix:
        """导出动能项的显式稀疏矩阵。"""

        nx, ny, nz = self.grid.shape
        step_u, step_v, step_w = self.grid.finite_difference_steps
        metric = self.grid.metric_contravariant

        identity_x = sparse.identity(nx, format="csr")
        identity_y = sparse.identity(ny, format="csr")
        identity_z = sparse.identity(nz, format="csr")

        d1_u = _periodic_first_derivative_matrix(nx, step_u)
        d1_v = _periodic_first_derivative_matrix(ny, step_v)
        d1_w = _periodic_first_derivative_matrix(nz, step_w)
        d2_u = _periodic_second_derivative_matrix(nx, step_u)
        d2_v = _periodic_second_derivative_matrix(ny, step_v)
        d2_w = _periodic_second_derivative_matrix(nz, step_w)

        laplacian = (
            metric[0, 0] * _kron3(d2_u, identity_y, identity_z)
            + metric[1, 1] * _kron3(identity_x, d2_v, identity_z)
            + metric[2, 2] * _kron3(identity_x, identity_y, d2_w)
            + 2.0 * metric[0, 1] * _kron3(d1_u, d1_v, identity_z)
            + 2.0 * metric[0, 2] * _kron3(d1_u, identity_y, d1_w)
            + 2.0 * metric[1, 2] * _kron3(identity_x, d1_v, d1_w)
        )
        return (-0.5 * laplacian).tocsr()


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

    def approximate_diagonal(self) -> np.ndarray:
        """返回供 Jacobi 预条件器使用的总哈密顿量对角近似。"""

        diagonal = (
            self.components.kinetic.approximate_diagonal()
            + self.components.effective_local_potential.values.reshape(-1, order="C")
        )
        if self.components.nonlocal_pseudopotential is not None:
            diagonal += self.components.nonlocal_pseudopotential.approximate_diagonal()
        return diagonal

    def to_sparse_matrix(self) -> sparse.csr_matrix:
        """按需导出总哈密顿量的显式稀疏矩阵。"""

        if self.components.nonlocal_pseudopotential is not None:
            raise NotImplementedError("当前非局域赝势采用矩阵自由低秩形式，暂不导出显式稀疏矩阵。")
        return (
            self.components.kinetic.to_sparse_matrix()
            + self.components.effective_local_potential.sparse_diagonal()
        ).tocsr()


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
) -> tuple[HamiltonianComponents, HamiltonianOperator]:
    """构造分项哈密顿量及其矩阵自由总算符。"""

    kinetic = KineticOperator(real_space_grid)
    ionic_local_potential = 构造离子局域势(
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
    nonlocal_pseudopotential = 构造非局域赝势算符(
        real_space_grid=real_space_grid,
        atom_sites=config.crystal.atom_sites,
        pseudopotentials=pseudopotentials,
    )
    effective_local_potential = PotentialField(
        name="总局域有效势",
        values=(
            ionic_local_potential.values
            + hartree_potential.values
            + exchange_correlation.potential.values
        ),
        note="由离子局域势、Hartree 势和交换关联势相加得到。",
    )
    nonlocal_note = "当前体系未从赝势中解析出非局域 projector。"
    if nonlocal_pseudopotential is not None:
        nonlocal_note = (
            f"已装配非局域 Kleinman-Bylander projector，总展开数 {nonlocal_pseudopotential.projector_count}。"
        )
    components = HamiltonianComponents(
        kinetic=kinetic,
        ionic_local_potential=ionic_local_potential,
        hartree_potential=hartree_potential,
        exchange_correlation_potential=exchange_correlation.potential,
        exchange_correlation_total_energy_hartree=exchange_correlation.total_energy_hartree,
        exchange_correlation_model=exchange_correlation.model_name,
        effective_local_potential=effective_local_potential,
        nonlocal_pseudopotential=nonlocal_pseudopotential,
        nonlocal_pseudopotential_note=nonlocal_note,
    )
    return components, HamiltonianOperator(real_space_grid, components)
