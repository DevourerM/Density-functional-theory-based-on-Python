"""SCF 主循环工作流。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.models import DensityGrid, InputConfig, RuntimeContext
from ..mesh.density import 构造电荷密度网格
from ..physics.density import 由k点波函数计算电荷密度, 计算费米能与占据
from ..physics.energy import TotalEnergyComponents, 计算总能分项
from ..physics.hamiltonian import HamiltonianComponents, HamiltonianOperator, 构造哈密顿算符
from ..solvers.eigen import IterativeEigenSolverConfig, IterativeEigenSolverResult, 求解本征波函数
from ..solvers.mixing import BaseDensityMixer, create_density_mixer
from .bootstrap import 初始化计算上下文



def _计算密度RMS差值(first_density: np.ndarray, second_density: np.ndarray) -> float:
    """计算两个密度场之间的 RMS 差值。"""

    difference = np.asarray(first_density, dtype=np.float64) - np.asarray(second_density, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference)))



def _计算能级最大变化(
    current_eigenvalues_hartree: np.ndarray,
    previous_eigenvalues_hartree: np.ndarray | None,
) -> float | None:
    """返回相邻两轮最低若干能级的最大变化。"""

    if previous_eigenvalues_hartree is None:
        return None

    current_values = np.asarray(current_eigenvalues_hartree, dtype=np.float64)
    previous_values = np.asarray(previous_eigenvalues_hartree, dtype=np.float64)
    if current_values.shape != previous_values.shape:
        raise ValueError("前后两轮本征值数组形状不一致。")
    return float(np.max(np.abs(current_values - previous_values)))



def _判断SCF收敛(
    density_residual_rms: float,
    eigenvalue_change_max: float | None,
    scf_tolerance: float,
) -> tuple[bool, bool]:
    """返回密度和能级两个判据是否满足。"""

    density_converged = density_residual_rms < scf_tolerance
    eigenvalue_converged = (
        eigenvalue_change_max is not None
        and eigenvalue_change_max < scf_tolerance
    )
    return density_converged, eigenvalue_converged



def _构造默认波函数求解配置(config: InputConfig) -> IterativeEigenSolverConfig:
    """根据输入参数生成默认的本征求解配置。"""

    solver_tolerance = max(1.0e-10, min(1.0e-8, 0.1 * config.scf.scf_tolerance))
    return IterativeEigenSolverConfig(
        method="LOBPCG",
        tolerance=solver_tolerance,
        max_iterations=200,
        largest=False,
        random_seed=0,
        verbosity_level=0,
    )


@dataclass(slots=True, frozen=True)
class SCFIterationRecord:
    """保存单次 SCF 迭代的关键信息。"""

    iteration: int
    density_residual_rms: float
    mixed_density_change_rms: float
    eigenvalue_change_max: float | None
    total_energy: TotalEnergyComponents
    density_converged: bool
    eigenvalue_converged: bool
    output_density_integrated_electrons: float
    mixed_density_integrated_electrons: float


@dataclass(slots=True)
class SCFResult:
    """保存整个 SCF 循环的结果。"""

    config: InputConfig
    converged: bool
    occupations: np.ndarray
    fermi_energy_hartree: float
    iterations: list[SCFIterationRecord]
    final_density: DensityGrid
    final_kpoint_eigensolutions: tuple[IterativeEigenSolverResult, ...]
    final_hamiltonian_components: tuple[HamiltonianComponents, ...]
    final_hamiltonians: tuple[HamiltonianOperator, ...]
    final_total_energy: TotalEnergyComponents
    linear_mixing_coefficient_used: float
    scf_tolerance_used: float
    max_scf_iterations_used: int
    eigensolver_method_used: str
    eigensolver_tolerance_used: float

    @property
    def iteration_count(self) -> int:
        """返回已经完成的 SCF 迭代步数。"""

        return len(self.iterations)

    @property
    def final_density_residual_rms(self) -> float:
        """返回最后一次迭代的密度残差 RMS。"""

        if not self.iterations:
            return 0.0
        return self.iterations[-1].density_residual_rms

    @property
    def final_eigenvalue_change_max(self) -> float | None:
        """返回最后一步最大能级变化。"""

        if not self.iterations:
            return None
        return self.iterations[-1].eigenvalue_change_max

    @property
    def final_total_energy_hartree(self) -> float:
        """返回最终总能。"""

        return self.final_total_energy.total_hartree

    @property
    def final_eigensolution(self) -> IterativeEigenSolverResult:
        """兼容旧接口，返回第一个 k 点的本征解。"""

        return self.final_kpoint_eigensolutions[0]

    @property
    def final_hamiltonian(self) -> HamiltonianOperator:
        """兼容旧接口，返回第一个 k 点哈密顿量。"""

        return self.final_hamiltonians[0]

    def summary(self) -> str:
        """生成简洁的 SCF 运行摘要。"""

        final_eigenvalue_change_text = "None"
        if self.final_eigenvalue_change_max is not None:
            final_eigenvalue_change_text = f"{self.final_eigenvalue_change_max:.6e}"
        return "\n".join(
            [
                "SCF 完成。",
                f"是否收敛: {self.converged}",
                f"完成迭代步数: {self.iteration_count}",
                f"SCF 收敛阈值: {self.scf_tolerance_used:.3e}",
                f"线性混合系数: {self.linear_mixing_coefficient_used:.3f}",
                f"本征求解方法: {self.eigensolver_method_used}",
                f"本征求解阈值: {self.eigensolver_tolerance_used:.3e}",
                f"k 点数: {self.config.kpoint_count}",
                f"费米能 (Ha): {self.fermi_energy_hartree:.12f}",
                f"最后密度残差 RMS: {self.final_density_residual_rms:.6e}",
                f"最后最大能级变化: {final_eigenvalue_change_text}",
                f"最终总能 (Ha): {self.final_total_energy_hartree:.12f}",
                f"最终电荷积分: {self.final_density.integrated_electrons:.6f}",
            ]
        )



def 运行SCF循环(
    input_path: str | Path = "INPUT.json",
    *,
    initial_context: RuntimeContext | None = None,
    solver_config: IterativeEigenSolverConfig | None = None,
) -> SCFResult:
    """执行基于密度重构和线性密度混合的最小 SCF 主循环。"""

    context = initial_context or 初始化计算上下文(input_path)
    config = context.config
    resolved_solver_config = solver_config or _构造默认波函数求解配置(config)
    density_mixer: BaseDensityMixer = create_density_mixer(config.scf.mixing)
    kpoints = config.numerical.kpoints

    current_density = context.density_grid
    fixed_ionic_local_potential = context.hamiltonian_components.ionic_local_potential
    static_nonlocal_by_kpoint: list | None = None
    previous_eigenvalues: np.ndarray | None = None
    iteration_records: list[SCFIterationRecord] = []
    latest_eigensolutions: tuple[IterativeEigenSolverResult, ...] | None = None
    latest_hamiltonian_components: tuple[HamiltonianComponents, ...] = (context.hamiltonian_components,)
    latest_hamiltonians: tuple[HamiltonianOperator, ...] = (context.hamiltonian,)
    latest_occupations = np.zeros((len(kpoints), config.numerical.nbands), dtype=np.float64)
    latest_fermi_energy = 0.0
    scf_converged = False

    for iteration in range(1, config.scf.max_iterations + 1):
        component_list: list[HamiltonianComponents] = []
        hamiltonian_list: list[HamiltonianOperator] = []
        eigen_list: list[IterativeEigenSolverResult] = []
        for kpoint_index, kpoint in enumerate(kpoints):
            fixed_nonlocal = None
            if static_nonlocal_by_kpoint is not None:
                fixed_nonlocal = static_nonlocal_by_kpoint[kpoint_index]
            components, hamiltonian = 构造哈密顿算符(
                config=config,
                real_space_grid=context.real_space_grid,
                density_grid=current_density,
                pseudopotentials=context.pseudopotentials,
                kpoint=kpoint,
                ionic_local_potential=fixed_ionic_local_potential,
                nonlocal_pseudopotential=fixed_nonlocal,
            )
            component_list.append(components)
            hamiltonian_list.append(hamiltonian)
            eigen_list.append(
                求解本征波函数(
                    hamiltonian,
                    config.numerical.nbands,
                    solver_config=resolved_solver_config,
                )
            )

        latest_hamiltonian_components = tuple(component_list)
        latest_hamiltonians = tuple(hamiltonian_list)
        latest_eigensolutions = tuple(eigen_list)
        if static_nonlocal_by_kpoint is None:
            static_nonlocal_by_kpoint = [components.nonlocal_pseudopotential for components in latest_hamiltonian_components]

        eigenvalues_by_kpoint = np.vstack(
            [eigensolution.eigenvalues_hartree for eigensolution in latest_eigensolutions]
        )
        latest_occupations, latest_fermi_energy = 计算费米能与占据(
            eigenvalues_by_kpoint,
            kpoints,
            context.total_valence_electrons,
            config.numerical.occupations.smearing_width_hartree,
            spin_degeneracy=config.numerical.occupations.spin_degeneracy,
        )
        reconstructed_density = 由k点波函数计算电荷密度(
            latest_eigensolutions,
            context.real_space_grid,
            kpoints,
            latest_occupations,
        )
        total_energy = 计算总能分项(
            real_space_grid=context.real_space_grid,
            density_grid=reconstructed_density,
            hamiltonian_components=latest_hamiltonian_components,
            eigensolutions=latest_eigensolutions,
            occupations_by_kpoint=latest_occupations,
            kpoints=kpoints,
        )

        density_residual_rms = _计算密度RMS差值(
            reconstructed_density.values,
            current_density.values,
        )
        mixed_density_change_rms = density_residual_rms
        eigenvalue_change_max = _计算能级最大变化(
            eigenvalues_by_kpoint.reshape(-1),
            previous_eigenvalues,
        )
        density_converged, eigenvalue_converged = _判断SCF收敛(
            density_residual_rms=density_residual_rms,
            eigenvalue_change_max=eigenvalue_change_max,
            scf_tolerance=config.scf.scf_tolerance,
        )

        if density_converged and eigenvalue_converged:
            current_density = reconstructed_density
            iteration_records.append(
                SCFIterationRecord(
                    iteration=iteration,
                    density_residual_rms=density_residual_rms,
                    mixed_density_change_rms=mixed_density_change_rms,
                    eigenvalue_change_max=eigenvalue_change_max,
                    total_energy=total_energy,
                    density_converged=density_converged,
                    eigenvalue_converged=eigenvalue_converged,
                    output_density_integrated_electrons=reconstructed_density.integrated_electrons,
                    mixed_density_integrated_electrons=current_density.integrated_electrons,
                )
            )
            previous_eigenvalues = eigenvalues_by_kpoint.reshape(-1).copy()
            scf_converged = True
            break

        mixed_density_values = density_mixer.mix(
            current_density.values,
            reconstructed_density.values,
        )
        mixed_density = 构造电荷密度网格(
            real_space_grid=context.real_space_grid,
            values=mixed_density_values,
            total_electrons=context.total_valence_electrons,
            clip_minimum=1.0e-12,
        )
        mixed_density_change_rms = _计算密度RMS差值(
            mixed_density.values,
            current_density.values,
        )

        iteration_records.append(
            SCFIterationRecord(
                iteration=iteration,
                density_residual_rms=density_residual_rms,
                mixed_density_change_rms=mixed_density_change_rms,
                eigenvalue_change_max=eigenvalue_change_max,
                total_energy=total_energy,
                density_converged=density_converged,
                eigenvalue_converged=eigenvalue_converged,
                output_density_integrated_electrons=reconstructed_density.integrated_electrons,
                mixed_density_integrated_electrons=mixed_density.integrated_electrons,
            )
        )

        current_density = mixed_density
        previous_eigenvalues = eigenvalues_by_kpoint.reshape(-1).copy()

    if latest_eigensolutions is None:
        raise RuntimeError("SCF 循环未能执行任何一次本征求解。")

    component_list = []
    hamiltonian_list = []
    eigen_list = []
    for kpoint_index, kpoint in enumerate(kpoints):
        components, hamiltonian = 构造哈密顿算符(
            config=config,
            real_space_grid=context.real_space_grid,
            density_grid=current_density,
            pseudopotentials=context.pseudopotentials,
            kpoint=kpoint,
            ionic_local_potential=fixed_ionic_local_potential,
            nonlocal_pseudopotential=None if static_nonlocal_by_kpoint is None else static_nonlocal_by_kpoint[kpoint_index],
        )
        component_list.append(components)
        hamiltonian_list.append(hamiltonian)
        eigen_list.append(
            求解本征波函数(
                hamiltonian,
                config.numerical.nbands,
                solver_config=resolved_solver_config,
            )
        )

    latest_hamiltonian_components = tuple(component_list)
    latest_hamiltonians = tuple(hamiltonian_list)
    latest_eigensolutions = tuple(eigen_list)
    eigenvalues_by_kpoint = np.vstack([eigensolution.eigenvalues_hartree for eigensolution in latest_eigensolutions])
    latest_occupations, latest_fermi_energy = 计算费米能与占据(
        eigenvalues_by_kpoint,
        kpoints,
        context.total_valence_electrons,
        config.numerical.occupations.smearing_width_hartree,
        spin_degeneracy=config.numerical.occupations.spin_degeneracy,
    )
    final_total_energy = 计算总能分项(
        real_space_grid=context.real_space_grid,
        density_grid=current_density,
        hamiltonian_components=latest_hamiltonian_components,
        eigensolutions=latest_eigensolutions,
        occupations_by_kpoint=latest_occupations,
        kpoints=kpoints,
    )

    return SCFResult(
        config=config,
        converged=scf_converged,
        occupations=latest_occupations,
        fermi_energy_hartree=latest_fermi_energy,
        iterations=iteration_records,
        final_density=current_density,
        final_kpoint_eigensolutions=latest_eigensolutions,
        final_hamiltonian_components=latest_hamiltonian_components,
        final_hamiltonians=latest_hamiltonians,
        final_total_energy=final_total_energy,
        linear_mixing_coefficient_used=config.scf.mixing.linear_coefficient,
        scf_tolerance_used=config.scf.scf_tolerance,
        max_scf_iterations_used=config.scf.max_iterations,
        eigensolver_method_used=resolved_solver_config.normalized_method,
        eigensolver_tolerance_used=resolved_solver_config.tolerance,
    )


执行SCF循环 = 运行SCF循环
