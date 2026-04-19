"""电荷密度网格的初始化逻辑。"""

from __future__ import annotations

import numpy as np

from ..core.models import DensityGrid, RealSpaceGrid


def 构造电荷密度网格(
    real_space_grid: RealSpaceGrid,
    values: np.ndarray,
    total_electrons: float,
    *,
    clip_minimum: float | None = None,
) -> DensityGrid:
    """根据给定网格值构造电荷密度对象，并保证电子数归一化。"""

    density_values = np.asarray(values, dtype=np.float64)
    if density_values.shape != real_space_grid.shape:
        raise ValueError(
            f"输入电荷密度形状 {density_values.shape} 与网格形状 {real_space_grid.shape} 不一致。"
        )

    density_values = density_values.copy()
    if clip_minimum is not None:
        density_values = np.clip(density_values, clip_minimum, None)

    integrated_electrons = float(np.sum(density_values) * real_space_grid.volume_element_bohr3)
    if integrated_electrons <= 0.0:
        raise ValueError("输入电荷密度的积分必须大于 0。")

    normalization_factor = total_electrons / integrated_electrons
    density_values *= normalization_factor

    return DensityGrid(
        values=density_values,
        grid_shape=real_space_grid.shape,
        cell_volume_bohr3=real_space_grid.cell_volume_bohr3,
        dvol_bohr3=real_space_grid.volume_element_bohr3,
        total_electrons=total_electrons,
    )


def 初始化电荷密度网格(
    real_space_grid: RealSpaceGrid,
    total_valence_electrons: float,
) -> DensityGrid:
    """先用均匀密度完成初始化，后续再替换为更物理的初猜。"""

    cell_volume_bohr3 = real_space_grid.cell_volume_bohr3
    uniform_density = total_valence_electrons / cell_volume_bohr3

    density_grid = 构造电荷密度网格(
        real_space_grid=real_space_grid,
        values=np.full(real_space_grid.shape, uniform_density, dtype=np.float64),
        total_electrons=total_valence_electrons,
    )

    if not np.isclose(
        density_grid.integrated_electrons,
        total_valence_electrons,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("初始化电荷密度的积分与总价电子数不一致。")

    return density_grid
