"""TensorBoard 日志输出。"""

from __future__ import annotations

import shutil
from pathlib import Path

from tensorboardX import SummaryWriter

from ..workflows.scf import SCFResult


def 清空日志目录(log_dir: str | Path) -> Path:
    """删除旧日志并重新创建目录。"""

    resolved_log_dir = Path(log_dir).expanduser().resolve()
    if resolved_log_dir.exists():
        shutil.rmtree(resolved_log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    return resolved_log_dir


def 写入SCF过程到TensorBoard(log_dir: str | Path, scf_result: SCFResult) -> Path:
    """把 SCF 迭代过程写入 TensorBoard event 文件。"""

    resolved_log_dir = Path(log_dir).expanduser().resolve()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(logdir=str(resolved_log_dir)) as writer:
        for record in scf_result.iterations:
            step = record.iteration
            writer.add_scalar("scf/density_residual_rms", record.density_residual_rms, step)
            writer.add_scalar("scf/mixed_density_change_rms", record.mixed_density_change_rms, step)
            writer.add_scalar("scf/density_converged", float(record.density_converged), step)
            writer.add_scalar("scf/eigenvalue_converged", float(record.eigenvalue_converged), step)
            writer.add_scalar("scf/output_density_integrated_electrons", record.output_density_integrated_electrons, step)
            writer.add_scalar("scf/mixed_density_integrated_electrons", record.mixed_density_integrated_electrons, step)
            if record.eigenvalue_change_max is not None:
                writer.add_scalar("scf/eigenvalue_change_max", record.eigenvalue_change_max, step)

            total_energy = record.total_energy
            writer.add_scalar("energy/total_hartree", total_energy.total_hartree, step)
            writer.add_scalar("energy/kinetic_hartree", total_energy.kinetic_hartree, step)
            writer.add_scalar("energy/ionic_local_hartree", total_energy.ionic_local_hartree, step)
            writer.add_scalar("energy/nonlocal_hartree", total_energy.nonlocal_hartree, step)
            writer.add_scalar("energy/hartree_hartree", total_energy.hartree_hartree, step)
            writer.add_scalar(
                "energy/exchange_correlation_hartree",
                total_energy.exchange_correlation_hartree,
                step,
            )
            writer.add_scalar("energy/band_energy_sum_hartree", total_energy.band_energy_sum_hartree, step)

        writer.add_scalar("final/total_energy_hartree", scf_result.final_total_energy.total_hartree, scf_result.iteration_count)
        writer.add_scalar("final/final_density_integrated_electrons", scf_result.final_density.integrated_electrons, scf_result.iteration_count)

    return resolved_log_dir