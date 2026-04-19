"""由本征波函数重构电子密度和占据数。"""

from __future__ import annotations

import numpy as np

from ..config.exceptions import DFTInputError
from ..core.models import DensityGrid, RealSpaceGrid
from ..mesh.density import 构造电荷密度网格
from ..solvers.eigen import IterativeEigenSolverResult


def 计算轨道占据(total_electrons: float, nbands: int) -> np.ndarray:
    """在非自旋极化近似下构造轨道占据数。"""

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
    density_values = np.tensordot(occupation_numbers, orbital_densities, axes=(0, 0))

    return 构造电荷密度网格(
        real_space_grid=real_space_grid,
        values=density_values,
        total_electrons=float(np.sum(occupation_numbers)),
        clip_minimum=0.0,
    )
