"""SCF 主循环工作流。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.models import DensityGrid, InputConfig, RuntimeContext
from ..mesh.density import 构造电荷密度网格
from ..physics.density import 由波函数计算电荷密度, 计算轨道占据
from ..physics.energy import TotalEnergyComponents, 计算总能分项
from ..physics.hamiltonian import HamiltonianComponents, HamiltonianOperator, 构造哈密顿算符
from ..solvers.eigen import (
    IterativeEigenSolverConfig,
    IterativeEigenSolverResult,
    求解本征波函数,
)
from ..solvers.mixing import BaseDensityMixer, create_density_mixer
from .bootstrap import 初始化计算上下文


def _计算密度RMS差值(first_density: np.ndarray, second_density: np.ndarray) -> float:
    """计算两个密度场之间的 RMS 差值。"""

    difference = np.asarray(first_density, dtype=np.float64) - np.asarray(second_density, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


def _计算能量变化(
    current_energy_hartree: float,
    previous_energy_hartree: float | None,
) -> tuple[float | None, float | None]:
    """返回绝对与相对总能变化。"""

    if previous_energy_hartree is None:
        return None, None
    absolute_change = abs(current_energy_hartree - previous_energy_hartree)
    relative_change = absolute_change / max(1.0, abs(current_energy_hartree), abs(previous_energy_hartree))
    return float(absolute_change), float(relative_change)


def _判断SCF收敛(
    density_residual_rms: float,
    total_energy_change_relative: float | None,
    wavefunction_residual_max: float | None,
    scf_tolerance: float,
    wavefunction_tolerance: float,
) -> tuple[bool, bool, bool]:
    """返回密度、能量、波函数三个判据是否满足。"""

    density_converged = density_residual_rms < scf_tolerance
    energy_converged = (
        total_energy_change_relative is not None
        and total_energy_change_relative < scf_tolerance
    )
    wavefunction_converged = (
        wavefunction_residual_max is None
        or wavefunction_residual_max < wavefunction_tolerance
    )
    return density_converged, energy_converged, wavefunction_converged


def _构造默认波函数求解配置(config: InputConfig) -> IterativeEigenSolverConfig:
    """根据输入参数生成默认的本征求解配置。"""

    return IterativeEigenSolverConfig(
        method="LOBPCG",
        tolerance=config.scf.wavefunction_tolerance,
        max_iterations=200,
        largest=False,
        random_seed=0,
        use_preconditioner=True,
        verbosity_level=0,
    )


@dataclass(slots=True, frozen=True)
class SCFIterationRecord:
    """保存单次 SCF 迭代的关键信息。"""

    iteration: int
    density_residual_rms: float
    mixed_density_change_rms: float
    wavefunction_residual_max: float | None
    total_energy: TotalEnergyComponents
    total_energy_change_abs: float | None
    total_energy_change_relative: float | None
    density_converged: bool
    energy_converged: bool
    wavefunction_converged: bool
    output_density_integrated_electrons: float
    mixed_density_integrated_electrons: float


@dataclass(slots=True)
class SCFResult:
    """保存整个 SCF 循环的结果。"""

    config: InputConfig
    converged: bool
    occupations: np.ndarray
    iterations: list[SCFIterationRecord]
    final_density: DensityGrid
    final_eigensolution: IterativeEigenSolverResult
    final_hamiltonian_components: HamiltonianComponents
    final_hamiltonian: HamiltonianOperator
    final_total_energy: TotalEnergyComponents
    density_mixer_method: str
    linear_mixing_coefficient_used: float
    diis_history_steps_used: int
    scf_tolerance_used: float
    wavefunction_tolerance_used: float
    max_scf_iterations_used: int
    eigensolver_method_used: str

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
    def final_total_energy_hartree(self) -> float:
        """返回最终晶体总能。"""

        return self.final_total_energy.total_crystal_hartree

    @property
    def final_total_electronic_energy_hartree(self) -> float:
        """返回最终电子总能。"""

        return self.final_total_energy.total_electronic_hartree

    @property
    def final_energy_change_relative(self) -> float | None:
        """返回最后一步相对能量变化。"""

        if not self.iterations:
            return None
        return self.iterations[-1].total_energy_change_relative

    def summary(self) -> str:
        """生成简洁的 SCF 运行摘要。"""

        final_energy_change_text = "None"
        if self.final_energy_change_relative is not None:
            final_energy_change_text = f"{self.final_energy_change_relative:.6e}"
        return "\n".join(
            [
                "SCF 完成。",
                f"是否收敛: {self.converged}",
                f"完成迭代步数: {self.iteration_count}",
                f"SCF 收敛阈值: {self.scf_tolerance_used:.3e}",
                f"波函数收敛阈值: {self.wavefunction_tolerance_used:.3e}",
                f"密度混合方法: {self.density_mixer_method}",
                f"线性混合系数: {self.linear_mixing_coefficient_used:.3f}",
                f"DIIS 历史步数: {self.diis_history_steps_used}",
                f"本征求解方法: {self.eigensolver_method_used}",
                f"最后密度残差 RMS: {self.final_density_residual_rms:.6e}",
                f"最终电子总能 (Ha): {self.final_total_electronic_energy_hartree:.12f}",
                f"离子-离子 Ewald 能 (Ha): {self.final_total_energy.ion_ion_ewald_hartree:.12f}",
                f"最终晶体总能 (Ha): {self.final_total_energy_hartree:.12f}",
                f"最后相对能量变化: {final_energy_change_text}",
                f"最终电荷积分: {self.final_density.integrated_electrons:.6f}",
            ]
        )


def 运行SCF循环(
    input_path: str | Path = "INPUT.json",
    *,
    initial_context: RuntimeContext | None = None,
    solver_config: IterativeEigenSolverConfig | None = None,
) -> SCFResult:
    """执行基于密度重构和密度混合的 SCF 主循环。"""

    context = initial_context or 初始化计算上下文(input_path)
    config = context.config
    resolved_solver_config = solver_config or _构造默认波函数求解配置(config)
    density_mixer: BaseDensityMixer = create_density_mixer(config.scf.mixing)
    occupations = 计算轨道占据(context.total_valence_electrons, config.numerical.nbands)

    current_density = context.density_grid
    previous_subspace: np.ndarray | None = None
    iteration_records: list[SCFIterationRecord] = []
    latest_eigensolution: IterativeEigenSolverResult | None = None
    latest_hamiltonian_components = context.hamiltonian_components
    latest_hamiltonian = context.hamiltonian
    previous_total_energy_hartree: float | None = None
    scf_converged = False

    for iteration in range(1, config.scf.max_iterations + 1):
        latest_hamiltonian_components, latest_hamiltonian = 构造哈密顿算符(
            config=config,
            real_space_grid=context.real_space_grid,
            density_grid=current_density,
            pseudopotentials=context.pseudopotentials,
        )

        latest_eigensolution = 求解本征波函数(
            latest_hamiltonian,
            config.numerical.nbands,
            solver_config=resolved_solver_config,
            initial_subspace=previous_subspace,
        )
        reconstructed_density = 由波函数计算电荷密度(
            latest_eigensolution,
            context.real_space_grid,
            occupations,
        )
        energy_components, _ = 构造哈密顿算符(
            config=config,
            real_space_grid=context.real_space_grid,
            density_grid=reconstructed_density,
            pseudopotentials=context.pseudopotentials,
        )
        total_energy = 计算总能分项(
            real_space_grid=context.real_space_grid,
            density_grid=reconstructed_density,
            hamiltonian_components=energy_components,
            eigensolution=latest_eigensolution,
            occupations=occupations,
            ion_ion_ewald_hartree=context.ion_ion_ewald_hartree,
        )

        density_residual_rms = _计算密度RMS差值(
            reconstructed_density.values,
            current_density.values,
        )
        mixed_density_change_rms = density_residual_rms
        wavefunction_residual_max = None
        if latest_eigensolution.final_residual_norms is not None:
            wavefunction_residual_max = float(np.max(latest_eigensolution.final_residual_norms))

        total_energy_change_abs, total_energy_change_relative = _计算能量变化(
            total_energy.total_crystal_hartree,
            previous_total_energy_hartree,
        )
        density_converged, energy_converged, wavefunction_converged = _判断SCF收敛(
            density_residual_rms=density_residual_rms,
            total_energy_change_relative=total_energy_change_relative,
            wavefunction_residual_max=wavefunction_residual_max,
            scf_tolerance=config.scf.scf_tolerance,
            wavefunction_tolerance=resolved_solver_config.tolerance,
        )

        if density_converged and energy_converged and wavefunction_converged:
            current_density = reconstructed_density
            iteration_records.append(
                SCFIterationRecord(
                    iteration=iteration,
                    density_residual_rms=density_residual_rms,
                    mixed_density_change_rms=mixed_density_change_rms,
                    wavefunction_residual_max=wavefunction_residual_max,
                    total_energy=total_energy,
                    total_energy_change_abs=total_energy_change_abs,
                    total_energy_change_relative=total_energy_change_relative,
                    density_converged=density_converged,
                    energy_converged=energy_converged,
                    wavefunction_converged=wavefunction_converged,
                    output_density_integrated_electrons=reconstructed_density.integrated_electrons,
                    mixed_density_integrated_electrons=current_density.integrated_electrons,
                )
            )
            previous_subspace = latest_eigensolution.eigenvectors
            previous_total_energy_hartree = total_energy.total_crystal_hartree
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
                wavefunction_residual_max=wavefunction_residual_max,
                total_energy=total_energy,
                total_energy_change_abs=total_energy_change_abs,
                total_energy_change_relative=total_energy_change_relative,
                density_converged=density_converged,
                energy_converged=energy_converged,
                wavefunction_converged=wavefunction_converged,
                output_density_integrated_electrons=reconstructed_density.integrated_electrons,
                mixed_density_integrated_electrons=mixed_density.integrated_electrons,
            )
        )

        current_density = mixed_density
        previous_subspace = latest_eigensolution.eigenvectors
        previous_total_energy_hartree = total_energy.total_crystal_hartree

    if latest_eigensolution is None:
        raise RuntimeError("SCF 循环未能执行任何一次本征求解。")

    latest_hamiltonian_components, latest_hamiltonian = 构造哈密顿算符(
        config=config,
        real_space_grid=context.real_space_grid,
        density_grid=current_density,
        pseudopotentials=context.pseudopotentials,
    )
    latest_eigensolution = 求解本征波函数(
        latest_hamiltonian,
        config.numerical.nbands,
        solver_config=resolved_solver_config,
        initial_subspace=previous_subspace,
    )
    final_total_energy = 计算总能分项(
        real_space_grid=context.real_space_grid,
        density_grid=current_density,
        hamiltonian_components=latest_hamiltonian_components,
        eigensolution=latest_eigensolution,
        occupations=occupations,
        ion_ion_ewald_hartree=context.ion_ion_ewald_hartree,
    )

    return SCFResult(
        config=config,
        converged=scf_converged,
        occupations=occupations,
        iterations=iteration_records,
        final_density=current_density,
        final_eigensolution=latest_eigensolution,
        final_hamiltonian_components=latest_hamiltonian_components,
        final_hamiltonian=latest_hamiltonian,
        final_total_energy=final_total_energy,
        density_mixer_method=config.scf.mixing.method,
        linear_mixing_coefficient_used=config.scf.mixing.linear_coefficient,
        diis_history_steps_used=config.scf.mixing.diis_history_steps,
        scf_tolerance_used=config.scf.scf_tolerance,
        wavefunction_tolerance_used=resolved_solver_config.tolerance,
        max_scf_iterations_used=config.scf.max_iterations,
        eigensolver_method_used=resolved_solver_config.normalized_method,
    )


执行SCF循环 = 运行SCF循环
