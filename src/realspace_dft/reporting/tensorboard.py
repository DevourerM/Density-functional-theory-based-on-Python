"""TensorBoard 日志输出。"""

from __future__ import annotations

import shutil
from pathlib import Path

from tensorboardX import SummaryWriter

from ..workflows.scf import SCFResult


def 清空日志目录(log_dir: str | Path) -> Path:
    """清空并重建日志目录。"""

    resolved_log_dir = Path(log_dir).resolve()
    if resolved_log_dir.exists():
        shutil.rmtree(resolved_log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    return resolved_log_dir


def 写入SCF过程到TensorBoard(log_dir: str | Path, scf_result: SCFResult) -> Path:
    """把 SCF 迭代过程写入 TensorBoard event 文件。"""

    resolved_log_dir = Path(log_dir).resolve()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(logdir=str(resolved_log_dir)) as writer:
        writer.add_text("run/input_path", str(scf_result.config.input_path), 0)
        writer.add_text("run/xc_functional", scf_result.config.numerical.xc_functional, 0)
        writer.add_text("run/mixer", scf_result.density_mixer_method, 0)

        for record in scf_result.iterations:
            step = record.iteration
            writer.add_scalar("scf/density_residual_rms", record.density_residual_rms, step)
            writer.add_scalar("scf/mixed_density_change_rms", record.mixed_density_change_rms, step)
            writer.add_scalar("scf/total_energy_hartree", record.total_energy.total_crystal_hartree, step)
            writer.add_scalar("scf/total_electronic_energy_hartree", record.total_energy.total_electronic_hartree, step)
            writer.add_scalar("energy/kinetic_hartree", record.total_energy.kinetic_hartree, step)
            writer.add_scalar("energy/ionic_local_hartree", record.total_energy.ionic_local_hartree, step)
            writer.add_scalar("energy/nonlocal_hartree", record.total_energy.nonlocal_hartree, step)
            writer.add_scalar("energy/hartree_hartree", record.total_energy.hartree_hartree, step)
            writer.add_scalar("energy/xc_hartree", record.total_energy.exchange_correlation_hartree, step)
            writer.add_scalar("energy/ion_ion_ewald_hartree", record.total_energy.ion_ion_ewald_hartree, step)
            writer.add_scalar("energy/band_sum_hartree", record.total_energy.band_energy_sum_hartree, step)
            writer.add_scalar("charge/output_integrated_electrons", record.output_density_integrated_electrons, step)
            writer.add_scalar("charge/mixed_integrated_electrons", record.mixed_density_integrated_electrons, step)
            if record.wavefunction_residual_max is not None:
                writer.add_scalar("scf/wavefunction_residual_max", record.wavefunction_residual_max, step)
            if record.total_energy_change_abs is not None:
                writer.add_scalar("scf/total_energy_change_abs", record.total_energy_change_abs, step)
            if record.total_energy_change_relative is not None:
                writer.add_scalar("scf/total_energy_change_relative", record.total_energy_change_relative, step)
            writer.add_scalar("convergence/density", float(record.density_converged), step)
            writer.add_scalar("convergence/energy", float(record.energy_converged), step)
            writer.add_scalar("convergence/wavefunction", float(record.wavefunction_converged), step)

        writer.add_scalar("final/total_energy_hartree", scf_result.final_total_energy_hartree, scf_result.iteration_count)
        writer.add_scalar("final/total_electronic_energy_hartree", scf_result.final_total_electronic_energy_hartree, scf_result.iteration_count)
        writer.add_scalar("final/integrated_electrons", scf_result.final_density.integrated_electrons, scf_result.iteration_count)

    return resolved_log_dir
