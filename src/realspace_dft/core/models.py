"""核心数据结构定义。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..physics.hamiltonian import HamiltonianComponents, HamiltonianOperator

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(slots=True, frozen=True)
class AtomSite:
    """保存单个原子的元素与分数坐标。"""

    element: str
    fractional_position: tuple[float, float, float]


@dataclass(slots=True, frozen=True)
class CrystalStructure:
    """保存晶体结构输入。"""

    lattice_constant_bohr: float
    lattice_constant_unit: str
    normalized_lattice_vectors: FloatArray
    atom_sites: tuple[AtomSite, ...]

    @property
    def lattice_vectors_bohr(self) -> FloatArray:
        """把归一化基矢缩放为真实晶格基矢。"""

        return self.lattice_constant_bohr * self.normalized_lattice_vectors

    @property
    def cell_volume_bohr3(self) -> float:
        """返回晶胞体积，单位为 bohr^3。"""

        return float(abs(np.linalg.det(self.lattice_vectors_bohr)))


@dataclass(slots=True, frozen=True)
class RealSpaceGrid:
    """保存三维实空间网格几何信息。"""

    shape: tuple[int, int, int]
    cell_matrix_bohr: FloatArray
    inverse_cell_matrix_bohr_inv: FloatArray
    reciprocal_lattice_matrix_bohr_inv: FloatArray
    metric_contravariant: FloatArray
    cell_volume_bohr3: float
    volume_element_bohr3: float
    fractional_coordinate_arrays: tuple[FloatArray, FloatArray, FloatArray]
    cartesian_coordinate_arrays_bohr: tuple[FloatArray, FloatArray, FloatArray]
    finite_difference_steps: tuple[float, float, float]

    @property
    def point_count(self) -> int:
        """返回总网格点数。"""

        return int(np.prod(self.shape))

    def reshape_vector(self, values: NDArray[np.generic]) -> NDArray[np.generic]:
        """把扁平向量恢复为三维网格形状。"""

        array = np.asarray(values)
        if array.ndim != 1 or array.size != self.point_count:
            raise ValueError(
                f"输入向量长度 {array.size} 与网格点数 {self.point_count} 不一致。"
            )
        return array.reshape(self.shape, order="C")

    def reshape_wavefunctions(self, values: NDArray[np.generic]) -> NDArray[np.generic]:
        """把列向量堆叠恢复为四维波函数块。"""

        array = np.asarray(values)
        if array.ndim != 2 or array.shape[0] != self.point_count:
            raise ValueError(
                "输入波函数块必须是形状为 (网格点数, 波函数数) 的二维数组。"
            )
        return array.reshape(self.shape + (array.shape[1],), order="C")

    def flatten_values(self, values: NDArray[np.generic]) -> NDArray[np.generic]:
        """把三维或四维网格值展平为向量或列向量堆叠。"""

        array = np.asarray(values)
        if array.shape[:3] != self.shape:
            raise ValueError(f"输入数组的前三个维度必须与网格形状 {self.shape} 一致。")

        if array.ndim == 3:
            return array.reshape(-1, order="C")
        return array.reshape(self.point_count, -1, order="C")

    def minimum_image_displacements_bohr(
        self,
        fractional_position: tuple[float, float, float],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """返回原子到所有网格点的最邻近周期映像位移向量。"""

        delta_u = self.fractional_coordinate_arrays[0] - fractional_position[0]
        delta_v = self.fractional_coordinate_arrays[1] - fractional_position[1]
        delta_w = self.fractional_coordinate_arrays[2] - fractional_position[2]

        delta_u = delta_u - np.rint(delta_u)
        delta_v = delta_v - np.rint(delta_v)
        delta_w = delta_w - np.rint(delta_w)

        dx = (
            delta_u * self.cell_matrix_bohr[0, 0]
            + delta_v * self.cell_matrix_bohr[1, 0]
            + delta_w * self.cell_matrix_bohr[2, 0]
        )
        dy = (
            delta_u * self.cell_matrix_bohr[0, 1]
            + delta_v * self.cell_matrix_bohr[1, 1]
            + delta_w * self.cell_matrix_bohr[2, 1]
        )
        dz = (
            delta_u * self.cell_matrix_bohr[0, 2]
            + delta_v * self.cell_matrix_bohr[1, 2]
            + delta_w * self.cell_matrix_bohr[2, 2]
        )
        return dx, dy, dz

    def minimum_image_distance_bohr(
        self,
        fractional_position: tuple[float, float, float],
    ) -> FloatArray:
        """计算一个原子到所有网格点的最邻近周期映像距离。"""

        dx, dy, dz = self.minimum_image_displacements_bohr(fractional_position)
        return np.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(slots=True, frozen=True)
class GridSpec:
    """保存实空间网格划分信息。"""

    shape: tuple[int, int, int]

    @property
    def point_count(self) -> int:
        """返回总网格点数。"""

        return int(np.prod(self.shape))


@dataclass(slots=True, frozen=True)
class MixingConfig:
    """保存密度混合配置。"""

    method: str
    linear_coefficient: float


@dataclass(slots=True, frozen=True)
class KPoint:
    """保存一个采用倒格矢分数坐标表示的 k 点及其权重。"""

    fractional_coordinates: tuple[float, float, float]
    weight: float

    @property
    def reduced_coordinates(self) -> tuple[float, float, float]:
        """返回 reduced 坐标。"""

        return self.fractional_coordinates


@dataclass(slots=True, frozen=True)
class OccupationSettings:
    """保存周期固体占据设置。"""

    method: str
    smearing_width_hartree: float
    spin_degeneracy: float = 2.0


@dataclass(slots=True, frozen=True)
class SCFSettings:
    """保存 SCF 迭代控制参数。"""

    max_iterations: int
    scf_tolerance: float
    mixing: MixingConfig


@dataclass(slots=True, frozen=True)
class NumericalSettings:
    """保存数值离散与本征求解配置。"""

    grid: GridSpec
    xc_functional: str
    nbands: int
    kpoints: tuple[KPoint, ...]
    occupations: OccupationSettings


@dataclass(slots=True, frozen=True)
class InputConfig:
    """保存经过校验后的输入参数。"""

    input_path: Path
    pseudopotential_dir_hint: str
    crystal: CrystalStructure
    scf: SCFSettings
    numerical: NumericalSettings

    @property
    def species(self) -> tuple[str, ...]:
        """返回体系中出现的元素种类。"""

        return tuple(sorted({site.element for site in self.crystal.atom_sites}))

    @property
    def species_counts(self) -> dict[str, int]:
        """返回每种元素对应的原子数。"""

        return dict(Counter(site.element for site in self.crystal.atom_sites))

    @property
    def kpoint_count(self) -> int:
        """返回 k 点数。"""

        return len(self.numerical.kpoints)


@dataclass(slots=True, frozen=True)
class NonlocalProjectorRadial:
    """保存单个非局域径向 projector。"""

    index: int
    angular_momentum: int
    radial_values: FloatArray
    cutoff_radius_bohr: float


@dataclass(slots=True, frozen=True)
class PseudopotentialInfo:
    """保存从 UPF 文件头部提取的赝势元数据。"""

    file_path: Path
    element: str
    functional: str
    z_valence: float
    pseudo_type: str
    radial_grid_bohr: FloatArray | None = None
    local_potential_hartree: FloatArray | None = None
    nonlocal_projectors: tuple[NonlocalProjectorRadial, ...] = ()
    dij_matrix_hartree: FloatArray | None = None

    @property
    def has_local_potential(self) -> bool:
        """返回是否已经载入局域赝势径向数据。"""

        return self.radial_grid_bohr is not None and self.local_potential_hartree is not None

    @property
    def has_nonlocal_projectors(self) -> bool:
        """返回是否已经载入非局域赝势 projector。"""

        return bool(self.nonlocal_projectors) and self.dij_matrix_hartree is not None


@dataclass(slots=True, frozen=True)
class PotentialField:
    """保存定义在实空间网格上的局域势场。"""

    name: str
    values: FloatArray
    unit: str = "hartree"
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def min_value(self) -> float:
        """返回势场的最小值。"""

        return float(np.min(self.values))

    @property
    def max_value(self) -> float:
        """返回势场的最大值。"""

        return float(np.max(self.values))

    def apply_to_grid(self, wavefunction_grid: ComplexArray) -> ComplexArray:
        """把对角局域势作用到单个或一组波函数上。"""

        psi = np.asarray(wavefunction_grid, dtype=np.complex128)
        if psi.shape[: self.values.ndim] != self.values.shape:
            raise ValueError("波函数网格形状与势场网格形状不一致。")
        if psi.ndim == self.values.ndim:
            return self.values * psi
        return self.values[..., np.newaxis] * psi

    def sparse_diagonal(self):
        """把局域势导出为稀疏对角矩阵。"""

        from scipy import sparse

        return sparse.diags(self.values.reshape(-1, order="C"), format="csr")


@dataclass(slots=True)
class DensityGrid:
    """保存电荷密度网格及其积分信息。"""

    values: FloatArray
    grid_shape: tuple[int, int, int]
    cell_volume_bohr3: float
    dvol_bohr3: float
    total_electrons: float

    @property
    def integrated_electrons(self) -> float:
        """对当前电荷密度进行体积分。"""

        return float(np.sum(self.values) * self.dvol_bohr3)

    @property
    def uniform_density(self) -> float:
        """返回均匀初始密度的值。"""

        return float(self.values.flat[0])


@dataclass(slots=True)
class RuntimeContext:
    """保存初始化阶段得到的完整运行上下文。"""

    config: InputConfig
    pseudopotential_dir: Path
    pseudopotentials: dict[str, PseudopotentialInfo]
    real_space_grid: RealSpaceGrid
    total_valence_electrons: float
    minimum_occupied_bands: int
    density_grid: DensityGrid
    hamiltonian_components: HamiltonianComponents
    hamiltonian: HamiltonianOperator

    def summary(self) -> str:
        """为命令行入口生成简洁的初始化摘要。"""

        pseudopotential_summary = ", ".join(
            f"{element}:{info.file_path.name}"
            for element, info in sorted(self.pseudopotentials.items())
        )

        return "\n".join(
            [
                "初始化完成。",
                f"输入文件: {self.config.input_path}",
                f"赝势目录: {self.pseudopotential_dir}",
                f"赝势映射: {pseudopotential_summary}",
                f"总价电子数: {self.total_valence_electrons:.6f}",
                f"最小占据轨道数: {self.minimum_occupied_bands}",
                f"输入轨道数 nbands: {self.config.numerical.nbands}",
                f"k 点数: {self.config.kpoint_count}",
                "k 点权重和: {:.6f}".format(
                    sum(kpoint.weight for kpoint in self.config.numerical.kpoints)
                ),
                (
                    "占据设置: {} smearing={:.6e} Ha spin_deg={:.1f}".format(
                        self.config.numerical.occupations.method,
                        self.config.numerical.occupations.smearing_width_hartree,
                        self.config.numerical.occupations.spin_degeneracy,
                    )
                ),
                f"电荷密度网格: {self.density_grid.grid_shape}",
                f"均匀初始密度: {self.density_grid.uniform_density:.6e}",
                f"电荷积分检查: {self.density_grid.integrated_electrons:.6f}",
                f"动能差分模板: {self.hamiltonian_components.kinetic.stencil_point_count} 点",
                "离子局域势范围: [{:.6e}, {:.6e}]".format(
                    self.hamiltonian_components.ionic_local_potential.min_value,
                    self.hamiltonian_components.ionic_local_potential.max_value,
                ),
                "Hartree 势范围: [{:.6e}, {:.6e}]".format(
                    self.hamiltonian_components.hartree_potential.min_value,
                    self.hamiltonian_components.hartree_potential.max_value,
                ),
                "XC 势范围: [{:.6e}, {:.6e}]".format(
                    self.hamiltonian_components.exchange_correlation_potential.min_value,
                    self.hamiltonian_components.exchange_correlation_potential.max_value,
                ),
                f"XC 实现说明: {self.hamiltonian_components.exchange_correlation_model}",
                f"非局域赝势说明: {self.hamiltonian_components.nonlocal_pseudopotential_note}",
            ]
        )
