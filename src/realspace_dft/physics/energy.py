"""总能与能量分项计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.models import DensityGrid, KPoint, RealSpaceGrid
from ..solvers.eigen import IterativeEigenSolverResult


@dataclass(slots=True, frozen=True)
class TotalEnergyComponents:
    """保存当前 SCF 迭代的能量分项。"""

    kinetic_hartree: float
    ionic_local_hartree: float
    hartree_hartree: float
    exchange_correlation_hartree: float
    band_energy_sum_hartree: float
    total_hartree: float
    nonlocal_hartree: float = 0.0
    notes: tuple[str, ...] = ()



def _算符期望每个band(real_space_grid: RealSpaceGrid, wavefunctions: np.ndarray, applied: np.ndarray) -> np.ndarray:
    psi_matrix = real_space_grid.flatten_values(wavefunctions)
    applied_matrix = real_space_grid.flatten_values(applied)
    return np.real(
        real_space_grid.volume_element_bohr3 * np.sum(psi_matrix.conj() * applied_matrix, axis=0)
    )



def 计算总能分项(
    real_space_grid: RealSpaceGrid,
    density_grid: DensityGrid,
    hamiltonian_components,
    eigensolutions: tuple[IterativeEigenSolverResult, ...],
    occupations_by_kpoint: np.ndarray,
    kpoints: tuple[KPoint, ...],
) -> TotalEnergyComponents:
    """计算当前密度和波函数对应的电子总能。"""

    occupations = np.asarray(occupations_by_kpoint, dtype=np.float64)
    if occupations.shape[0] != len(eigensolutions) or len(eigensolutions) != len(kpoints):
        raise ValueError("总能计算时 k 点、波函数和占据数组长度必须一致。")

    kinetic_energy = 0.0
    nonlocal_energy = 0.0
    band_energy_sum = 0.0
    for kpoint_index, eigensolution in enumerate(eigensolutions):
        wavefunctions = eigensolution.wavefunctions_as_grids()
        kinetic_applied = hamiltonian_components[kpoint_index].kinetic.apply_to_grid(wavefunctions)
        kinetic_expectation = _算符期望每个band(real_space_grid, wavefunctions, kinetic_applied)
        occupation_numbers = occupations[kpoint_index]
        weight = kpoints[kpoint_index].weight
        kinetic_energy += weight * float(np.dot(occupation_numbers, kinetic_expectation))
        if hamiltonian_components[kpoint_index].nonlocal_pseudopotential is not None:
            nonlocal_expectation = hamiltonian_components[kpoint_index].nonlocal_pseudopotential.expectation_per_band(
                wavefunctions
            )
            nonlocal_energy += weight * float(np.dot(occupation_numbers, nonlocal_expectation))
        band_energy_sum += weight * float(np.dot(occupation_numbers, eigensolution.eigenvalues_hartree))

    ionic_local_energy = float(
        np.sum(density_grid.values * hamiltonian_components[0].ionic_local_potential.values)
        * real_space_grid.volume_element_bohr3
    )
    hartree_energy = float(
        0.5 * np.sum(density_grid.values * hamiltonian_components[0].hartree_potential.values)
        * real_space_grid.volume_element_bohr3
    )
    exchange_correlation_energy = float(hamiltonian_components[0].exchange_correlation_total_energy_hartree)
    total_energy = (
        kinetic_energy
        + ionic_local_energy
        + nonlocal_energy
        + hartree_energy
        + exchange_correlation_energy
    )
    return TotalEnergyComponents(
        kinetic_hartree=kinetic_energy,
        ionic_local_hartree=ionic_local_energy,
        hartree_hartree=hartree_energy,
        exchange_correlation_hartree=exchange_correlation_energy,
        band_energy_sum_hartree=band_energy_sum,
        total_hartree=total_energy,
        nonlocal_hartree=nonlocal_energy,
        notes=("当前总能包含动能、局域/非局域赝势、Hartree 与交换关联分项。",),
    )
