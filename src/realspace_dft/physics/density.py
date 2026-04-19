"""由本征波函数重构电子密度和占据数。"""

from __future__ import annotations

import numpy as np

from ..config.exceptions import DFTInputError
from ..core.models import DensityGrid, KPoint, RealSpaceGrid
from ..mesh.density import 构造电荷密度网格
from ..solvers.eigen import IterativeEigenSolverResult


def 计算轨道占据(total_electrons: float, nbands: int) -> np.ndarray:
    """兼容旧接口，返回单个 Gamma 点下的双占据近似。"""

    if total_electrons <= 0.0:
        raise DFTInputError("总电子数必须大于 0。")
    if nbands <= 0:
        raise DFTInputError("nbands 必须大于 0。")

    occupations = np.zeros(nbands, dtype=np.float64)
    remaining_electrons = float(total_electrons)
    for band_index in range(nbands):
        if remaining_electrons <= 1.0e-12:
            break
        occupations[band_index] = min(2.0, remaining_electrons)
        remaining_electrons -= occupations[band_index]

    if remaining_electrons > 1.0e-8:
        raise DFTInputError(
            "nbands 不足以容纳全部电子，请检查输入的轨道数设置。"
        )

    return occupations


def _费米狄拉克分布(
    eigenvalues_hartree: np.ndarray,
    chemical_potential_hartree: float,
    smearing_width_hartree: float,
) -> np.ndarray:
    scaled = (np.asarray(eigenvalues_hartree, dtype=np.float64) - chemical_potential_hartree) / smearing_width_hartree
    scaled = np.clip(scaled, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(scaled))


def 计算费米能与占据(
    eigenvalues_by_kpoint_hartree: np.ndarray,
    kpoints: tuple[KPoint, ...],
    total_electrons: float,
    smearing_width_hartree: float,
    spin_degeneracy: float = 2.0,
) -> tuple[np.ndarray, float]:
    """对给定的 k 点本征值求费米能和分数占据。"""

    eigenvalues = np.asarray(eigenvalues_by_kpoint_hartree, dtype=np.float64)
    if eigenvalues.ndim != 2:
        raise ValueError("k 点本征值数组必须是二维 (nk, nbands) 结构。")
    if len(kpoints) != eigenvalues.shape[0]:
        raise ValueError("k 点数与本征值数组第一维不一致。")
    if total_electrons <= 0.0:
        raise DFTInputError("总电子数必须大于 0。")
    if smearing_width_hartree <= 0.0:
        raise DFTInputError("费米展宽必须大于 0。")
    if spin_degeneracy <= 0.0:
        raise DFTInputError("自旋简并度必须大于 0。")

    weights = np.asarray([kpoint.weight for kpoint in kpoints], dtype=np.float64)
    if not np.isclose(np.sum(weights), 1.0, atol=1.0e-10):
        raise ValueError("k 点权重和必须为 1。")

    lower_bound = float(np.min(eigenvalues) - 20.0 * smearing_width_hartree - abs(total_electrons))
    upper_bound = float(np.max(eigenvalues) + 20.0 * smearing_width_hartree + abs(total_electrons))

    def electron_count(chemical_potential: float) -> float:
        occupations = _费米狄拉克分布(eigenvalues, chemical_potential, smearing_width_hartree)
        return float(spin_degeneracy * np.sum(weights[:, np.newaxis] * occupations))

    lower_count = electron_count(lower_bound)
    upper_count = electron_count(upper_bound)
    if lower_count > total_electrons or upper_count < total_electrons:
        raise DFTInputError("费米能搜索区间未能包络总电子数，请增大展宽或轨道数。")

    for _ in range(200):
        midpoint = 0.5 * (lower_bound + upper_bound)
        midpoint_count = electron_count(midpoint)
        if abs(midpoint_count - total_electrons) < 1.0e-12:
            occupations = spin_degeneracy * _费米狄拉克分布(eigenvalues, midpoint, smearing_width_hartree)
            return occupations, midpoint
        if midpoint_count < total_electrons:
            lower_bound = midpoint
        else:
            upper_bound = midpoint

    chemical_potential = 0.5 * (lower_bound + upper_bound)
    occupations = spin_degeneracy * _费米狄拉克分布(eigenvalues, chemical_potential, smearing_width_hartree)
    return occupations, chemical_potential


def 由波函数计算电荷密度(
    eigen_result: IterativeEigenSolverResult,
    real_space_grid: RealSpaceGrid,
    occupations: np.ndarray,
) -> DensityGrid:
    """根据波函数与占据数重构新的电子密度。"""

    occupation_numbers = np.asarray(occupations, dtype=np.float64)
    if occupation_numbers.shape != (eigen_result.nbands,):
        raise ValueError("占据数数组长度必须与求得的波函数数量一致。")

    wavefunctions = eigen_result.wavefunctions_as_grids()
    orbital_densities = np.abs(wavefunctions) ** 2
    density_values = np.tensordot(orbital_densities, occupation_numbers, axes=([3], [0]))

    return 构造电荷密度网格(
        real_space_grid=real_space_grid,
        values=density_values,
        total_electrons=float(np.sum(occupation_numbers)),
        clip_minimum=0.0,
    )


def 由k点波函数计算电荷密度(
    eigen_results: tuple[IterativeEigenSolverResult, ...],
    real_space_grid: RealSpaceGrid,
    kpoints: tuple[KPoint, ...],
    occupations_by_kpoint: np.ndarray,
) -> DensityGrid:
    """根据全部 k 点波函数与占据数重构总电子密度。"""

    if len(eigen_results) != len(kpoints):
        raise ValueError("k 点求解结果数与 k 点数不一致。")

    occupations = np.asarray(occupations_by_kpoint, dtype=np.float64)
    if occupations.shape[0] != len(kpoints):
        raise ValueError("占据数数组第一维必须等于 k 点数。")

    density_values = np.zeros(real_space_grid.shape, dtype=np.float64)
    for kpoint_index, eigen_result in enumerate(eigen_results):
        if occupations.shape[1] != eigen_result.nbands:
            raise ValueError("占据数数组第二维必须与 band 数一致。")
        wavefunctions = eigen_result.wavefunctions_as_grids()
        orbital_densities = np.abs(wavefunctions) ** 2
        density_values += kpoints[kpoint_index].weight * np.tensordot(
            orbital_densities,
            occupations[kpoint_index],
            axes=([3], [0]),
        )

    total_electrons = float(np.sum(density_values) * real_space_grid.volume_element_bohr3)
    return 构造电荷密度网格(
        real_space_grid=real_space_grid,
        values=density_values,
        total_electrons=total_electrons,
        clip_minimum=0.0,
    )
