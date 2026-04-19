"""总能与能量分项计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.models import DensityGrid, RealSpaceGrid
from ..solvers.eigen import IterativeEigenSolverResult


@dataclass(slots=True, frozen=True)
class TotalEnergyComponents:
    """保存当前 SCF 迭代的能量分项。"""

    kinetic_hartree: float
    ionic_local_hartree: float
    nonlocal_hartree: float
    hartree_hartree: float
    exchange_correlation_hartree: float
    ion_ion_ewald_hartree: float
    band_energy_sum_hartree: float
    total_electronic_hartree: float
    total_crystal_hartree: float
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
    eigensolution: IterativeEigenSolverResult,
    occupations: np.ndarray,
    *,
    ion_ion_ewald_hartree: float = 0.0,
) -> TotalEnergyComponents:
    """计算当前密度和波函数对应的电子总能。"""

    occupation_numbers = np.asarray(occupations, dtype=np.float64)
    wavefunctions = eigensolution.wavefunctions_as_grids()

    kinetic_applied = hamiltonian_components.kinetic.apply_to_grid(wavefunctions)
    kinetic_expectation = _算符期望每个band(real_space_grid, wavefunctions, kinetic_applied)
    kinetic_energy = float(np.dot(occupation_numbers, kinetic_expectation))

    ionic_local_energy = float(
        np.sum(density_grid.values * hamiltonian_components.ionic_local_potential.values)
        * real_space_grid.volume_element_bohr3
    )
    hartree_energy = float(
        0.5 * np.sum(density_grid.values * hamiltonian_components.hartree_potential.values)
        * real_space_grid.volume_element_bohr3
    )
    exchange_correlation_energy = float(hamiltonian_components.exchange_correlation_total_energy_hartree)

    nonlocal_energy = 0.0
    if hamiltonian_components.nonlocal_pseudopotential is not None:
        nonlocal_expectation = hamiltonian_components.nonlocal_pseudopotential.expectation_per_band(
            wavefunctions
        )
        nonlocal_energy = float(np.dot(occupation_numbers, nonlocal_expectation))

    band_energy_sum = float(np.dot(occupation_numbers, eigensolution.eigenvalues_hartree))
    total_electronic_energy = (
        kinetic_energy
        + ionic_local_energy
        + nonlocal_energy
        + hartree_energy
        + exchange_correlation_energy
    )
    total_crystal_energy = total_electronic_energy + float(ion_ion_ewald_hartree)
    return TotalEnergyComponents(
        kinetic_hartree=kinetic_energy,
        ionic_local_hartree=ionic_local_energy,
        nonlocal_hartree=nonlocal_energy,
        hartree_hartree=hartree_energy,
        exchange_correlation_hartree=exchange_correlation_energy,
        ion_ion_ewald_hartree=float(ion_ion_ewald_hartree),
        band_energy_sum_hartree=band_energy_sum,
        total_electronic_hartree=total_electronic_energy,
        total_crystal_hartree=total_crystal_energy,
        notes=("离子-离子静电常数项采用周期性点电荷 Ewald 求和，并包含与 G=0 去除一致的中性背景修正。",),
    )
