"""周期性晶体中的 Ewald 离子-离子求和。"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc

from ..core.models import CrystalStructure, PseudopotentialInfo


def _validate_relative_tolerance(relative_tolerance: float) -> float:
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
        raise ValueError("Ewald 求和的 relative_tolerance 必须位于 (0, 1) 区间内。")
    return tolerance


def _默认屏蔽参数_bohr_inv(cell_volume_bohr3: float, charge_count: int) -> float:
    """根据晶胞体积和离子数选择经验屏蔽参数。"""

    return float(np.sqrt(np.pi) * max(charge_count, 1) ** (1.0 / 6.0) / cell_volume_bohr3 ** (1.0 / 3.0))


def _枚举截断内晶格向量(basis_matrix: np.ndarray, norm_cutoff: float) -> np.ndarray:
    """枚举范数截断球内的整数晶格向量组合。"""

    if norm_cutoff <= 0.0:
        return np.zeros((1, 3), dtype=np.float64)

    basis = np.asarray(basis_matrix, dtype=np.float64)
    singular_values = np.linalg.svd(basis, compute_uv=False)
    min_singular_value = float(np.min(singular_values))
    if min_singular_value <= 1.0e-14:
        raise ValueError("晶格基矢矩阵接近奇异，无法执行 Ewald 求和。")

    max_index = int(np.ceil(norm_cutoff / min_singular_value))
    integer_range = np.arange(-max_index, max_index + 1, dtype=np.int32)
    triplets = np.stack(np.meshgrid(integer_range, integer_range, integer_range, indexing="ij"), axis=-1).reshape(-1, 3)
    vectors = triplets @ basis
    norms = np.linalg.norm(vectors, axis=1)
    return vectors[norms <= norm_cutoff + 1.0e-14]


def 计算周期点电荷Ewald能(
    cell_matrix_bohr: np.ndarray,
    fractional_positions: np.ndarray,
    charges: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
    screening_parameter_bohr_inv: float | None = None,
) -> float:
    """对一个周期性点电荷体系计算 Ewald 库仑能。

    当前实现包含标准自能修正和均匀中性背景修正，
    与周期性体系中去除 G=0 库仑分量的约定保持一致。
    """

    tolerance = _validate_relative_tolerance(relative_tolerance)
    cell_matrix = np.asarray(cell_matrix_bohr, dtype=np.float64)
    fractional = np.asarray(fractional_positions, dtype=np.float64)
    ionic_charges = np.asarray(charges, dtype=np.float64)

    if cell_matrix.shape != (3, 3):
        raise ValueError("cell_matrix_bohr 必须是 3x3 实矩阵。")
    if fractional.ndim != 2 or fractional.shape[1] != 3:
        raise ValueError("fractional_positions 必须是形状为 (离子数, 3) 的数组。")
    if ionic_charges.shape != (fractional.shape[0],):
        raise ValueError("charges 的长度必须与离子位置数量一致。")

    cell_volume_bohr3 = float(abs(np.linalg.det(cell_matrix)))
    if cell_volume_bohr3 <= 1.0e-14:
        raise ValueError("晶胞体积过小，无法进行 Ewald 求和。")

    eta = float(screening_parameter_bohr_inv or _默认屏蔽参数_bohr_inv(cell_volume_bohr3, ionic_charges.size))
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("Ewald 屏蔽参数必须是正的有限实数。")

    decay_factor = math.sqrt(-math.log(tolerance))
    real_cutoff_bohr = decay_factor / eta
    reciprocal_cutoff_bohr_inv = 2.0 * eta * decay_factor

    cartesian_positions_bohr = fractional @ cell_matrix
    real_lattice_vectors_bohr = _枚举截断内晶格向量(cell_matrix, real_cutoff_bohr)
    reciprocal_matrix_bohr_inv = 2.0 * np.pi * np.linalg.inv(cell_matrix)
    reciprocal_lattice_vectors_bohr_inv = _枚举截断内晶格向量(
        reciprocal_matrix_bohr_inv,
        reciprocal_cutoff_bohr_inv,
    )
    reciprocal_norms = np.linalg.norm(reciprocal_lattice_vectors_bohr_inv, axis=1)
    nonzero_reciprocal_mask = reciprocal_norms > 1.0e-14
    reciprocal_lattice_vectors_bohr_inv = reciprocal_lattice_vectors_bohr_inv[nonzero_reciprocal_mask]
    reciprocal_norms = reciprocal_norms[nonzero_reciprocal_mask]

    charge_outer = ionic_charges[:, np.newaxis] * ionic_charges[np.newaxis, :]
    pair_displacements = cartesian_positions_bohr[np.newaxis, :, :] - cartesian_positions_bohr[:, np.newaxis, :]

    real_space_energy = 0.0
    for translation_vector in real_lattice_vectors_bohr:
        translated_displacements = pair_displacements + translation_vector[np.newaxis, np.newaxis, :]
        distances = np.linalg.norm(translated_displacements, axis=-1)
        valid_mask = distances > 1.0e-14
        contribution = np.zeros_like(distances)
        contribution[valid_mask] = erfc(eta * distances[valid_mask]) / distances[valid_mask]
        real_space_energy += 0.5 * float(np.sum(charge_outer * contribution))

    if reciprocal_lattice_vectors_bohr_inv.size == 0:
        reciprocal_space_energy = 0.0
    else:
        phases = reciprocal_lattice_vectors_bohr_inv @ cartesian_positions_bohr.T
        structure_factor = np.exp(1j * phases) @ ionic_charges
        damping = np.exp(-(reciprocal_norms * reciprocal_norms) / (4.0 * eta * eta)) / (reciprocal_norms * reciprocal_norms)
        reciprocal_space_energy = float((2.0 * np.pi / cell_volume_bohr3) * np.sum(damping * np.abs(structure_factor) ** 2))

    self_energy = float(-(eta / np.sqrt(np.pi)) * np.sum(ionic_charges * ionic_charges))
    background_energy = float(-np.pi * np.sum(ionic_charges) ** 2 / (2.0 * eta * eta * cell_volume_bohr3))
    return real_space_energy + reciprocal_space_energy + self_energy + background_energy


def 计算离子离子Ewald能(
    crystal: CrystalStructure,
    pseudopotentials: dict[str, PseudopotentialInfo],
    *,
    relative_tolerance: float = 1.0e-12,
) -> float:
    """根据晶体结构和赝势的价电子数计算离子-离子 Ewald 常数项。"""

    fractional_positions = np.asarray(
        [atom_site.fractional_position for atom_site in crystal.atom_sites],
        dtype=np.float64,
    )
    charges = np.asarray(
        [pseudopotentials[atom_site.element].z_valence for atom_site in crystal.atom_sites],
        dtype=np.float64,
    )
    return 计算周期点电荷Ewald能(
        crystal.lattice_vectors_bohr,
        fractional_positions,
        charges,
        relative_tolerance=relative_tolerance,
    )
