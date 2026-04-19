"""三维实空间网格几何构造。"""

from __future__ import annotations

import numpy as np

from ..core.models import CrystalStructure, GridSpec, RealSpaceGrid


def 构造实空间网格(crystal: CrystalStructure, grid_spec: GridSpec) -> RealSpaceGrid:
    """根据晶体结构与网格划分生成三维周期性实空间网格。"""

    shape = grid_spec.shape
    cell_matrix_bohr = crystal.lattice_vectors_bohr
    inverse_cell_matrix = np.linalg.inv(cell_matrix_bohr)
    reciprocal_lattice_matrix_bohr_inv = 2.0 * np.pi * inverse_cell_matrix
    metric_contravariant = inverse_cell_matrix @ inverse_cell_matrix.T
    cell_volume_bohr3 = crystal.cell_volume_bohr3
    volume_element_bohr3 = cell_volume_bohr3 / grid_spec.point_count
    finite_difference_steps = tuple(1.0 / point_count for point_count in shape)

    fractional_axes = tuple(
        np.arange(point_count, dtype=np.float64) / point_count for point_count in shape
    )
    frac_u, frac_v, frac_w = np.meshgrid(*fractional_axes, indexing="ij")

    cart_x = (
        frac_u * cell_matrix_bohr[0, 0]
        + frac_v * cell_matrix_bohr[1, 0]
        + frac_w * cell_matrix_bohr[2, 0]
    )
    cart_y = (
        frac_u * cell_matrix_bohr[0, 1]
        + frac_v * cell_matrix_bohr[1, 1]
        + frac_w * cell_matrix_bohr[2, 1]
    )
    cart_z = (
        frac_u * cell_matrix_bohr[0, 2]
        + frac_v * cell_matrix_bohr[1, 2]
        + frac_w * cell_matrix_bohr[2, 2]
    )

    return RealSpaceGrid(
        shape=shape,
        cell_matrix_bohr=cell_matrix_bohr,
        inverse_cell_matrix_bohr_inv=inverse_cell_matrix,
        reciprocal_lattice_matrix_bohr_inv=reciprocal_lattice_matrix_bohr_inv,
        metric_contravariant=metric_contravariant,
        cell_volume_bohr3=cell_volume_bohr3,
        volume_element_bohr3=volume_element_bohr3,
        fractional_coordinate_arrays=(frac_u, frac_v, frac_w),
        cartesian_coordinate_arrays_bohr=(cart_x, cart_y, cart_z),
        finite_difference_steps=finite_difference_steps,
    )
