"""周期性实空间网格上的微分算子。"""

from __future__ import annotations

import numpy as np

from ..core.models import RealSpaceGrid


def 周期性一阶导数(values: np.ndarray, axis: int, step: float) -> np.ndarray:
    """在指定轴上计算周期性中心差分一阶导数。"""

    array = np.asarray(values, dtype=np.float64)
    return (np.roll(array, -1, axis=axis) - np.roll(array, 1, axis=axis)) / (2.0 * step)


def 计算笛卡尔梯度(values: np.ndarray, real_space_grid: RealSpaceGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算标量场在笛卡尔坐标下的梯度。"""

    array = np.asarray(values, dtype=np.float64)
    step_u, step_v, step_w = real_space_grid.finite_difference_steps
    du = 周期性一阶导数(array, axis=0, step=step_u)
    dv = 周期性一阶导数(array, axis=1, step=step_v)
    dw = 周期性一阶导数(array, axis=2, step=step_w)

    inverse = real_space_grid.inverse_cell_matrix_bohr_inv
    grad_x = inverse[0, 0] * du + inverse[0, 1] * dv + inverse[0, 2] * dw
    grad_y = inverse[1, 0] * du + inverse[1, 1] * dv + inverse[1, 2] * dw
    grad_z = inverse[2, 0] * du + inverse[2, 1] * dv + inverse[2, 2] * dw
    return grad_x, grad_y, grad_z


def 计算梯度模平方(values: np.ndarray, real_space_grid: RealSpaceGrid) -> np.ndarray:
    """计算笛卡尔梯度的模平方。"""

    grad_x, grad_y, grad_z = 计算笛卡尔梯度(values, real_space_grid)
    return grad_x * grad_x + grad_y * grad_y + grad_z * grad_z


def 计算笛卡尔散度(
    vector_components: tuple[np.ndarray, np.ndarray, np.ndarray],
    real_space_grid: RealSpaceGrid,
) -> np.ndarray:
    """计算笛卡尔向量场的散度。"""

    fx, fy, fz = (np.asarray(component, dtype=np.float64) for component in vector_components)
    step_u, step_v, step_w = real_space_grid.finite_difference_steps

    du_fx = 周期性一阶导数(fx, axis=0, step=step_u)
    dv_fx = 周期性一阶导数(fx, axis=1, step=step_v)
    dw_fx = 周期性一阶导数(fx, axis=2, step=step_w)
    du_fy = 周期性一阶导数(fy, axis=0, step=step_u)
    dv_fy = 周期性一阶导数(fy, axis=1, step=step_v)
    dw_fy = 周期性一阶导数(fy, axis=2, step=step_w)
    du_fz = 周期性一阶导数(fz, axis=0, step=step_u)
    dv_fz = 周期性一阶导数(fz, axis=1, step=step_v)
    dw_fz = 周期性一阶导数(fz, axis=2, step=step_w)

    inverse = real_space_grid.inverse_cell_matrix_bohr_inv
    return (
        inverse[0, 0] * du_fx
        + inverse[0, 1] * dv_fx
        + inverse[0, 2] * dw_fx
        + inverse[1, 0] * du_fy
        + inverse[1, 1] * dv_fy
        + inverse[1, 2] * dw_fy
        + inverse[2, 0] * du_fz
        + inverse[2, 1] * dv_fz
        + inverse[2, 2] * dw_fz
    )