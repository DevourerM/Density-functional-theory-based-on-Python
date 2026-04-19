"""SCF 结果持久化输出。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..workflows.scf import SCFResult


def _准备输出目录(output_dir: str | Path) -> Path:
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_output_dir


def 保存最终电荷密度(output_dir: str | Path, scf_result: SCFResult) -> Path:
    """保存最终电荷密度与网格信息。"""

    resolved_output_dir = _准备输出目录(output_dir)
    density_path = resolved_output_dir / "final_density.npz"
    np.savez_compressed(
        density_path,
        density=scf_result.final_density.values,
        grid_shape=np.asarray(scf_result.final_density.grid_shape, dtype=np.int64),
        total_electrons=np.float64(scf_result.final_density.total_electrons),
        integrated_electrons=np.float64(scf_result.final_density.integrated_electrons),
        dvol_bohr3=np.float64(scf_result.final_density.dvol_bohr3),
        cell_volume_bohr3=np.float64(scf_result.final_density.cell_volume_bohr3),
        total_energy_hartree=np.float64(scf_result.final_total_energy_hartree),
        total_electronic_energy_hartree=np.float64(scf_result.final_total_electronic_energy_hartree),
        ion_ion_ewald_hartree=np.float64(scf_result.final_total_energy.ion_ion_ewald_hartree),
    )
    return density_path


def 保存SCF摘要(output_dir: str | Path, scf_result: SCFResult) -> Path:
    """保存简洁 JSON 摘要。"""

    resolved_output_dir = _准备输出目录(output_dir)
    summary_path = resolved_output_dir / "scf_summary.json"
    summary_payload = {
        "input_path": str(scf_result.config.input_path),
        "converged": scf_result.converged,
        "iteration_count": scf_result.iteration_count,
        "xc_functional": scf_result.config.numerical.xc_functional,
        "density_mixer_method": scf_result.density_mixer_method,
        "eigensolver_method": scf_result.eigensolver_method_used,
        "final_density_residual_rms": scf_result.final_density_residual_rms,
        "final_energy_change_relative": scf_result.final_energy_change_relative,
        "final_total_energy_hartree": scf_result.final_total_energy_hartree,
        "final_total_electronic_energy_hartree": scf_result.final_total_electronic_energy_hartree,
        "final_integrated_electrons": scf_result.final_density.integrated_electrons,
        "energy_components": {
            "kinetic_hartree": scf_result.final_total_energy.kinetic_hartree,
            "ionic_local_hartree": scf_result.final_total_energy.ionic_local_hartree,
            "nonlocal_hartree": scf_result.final_total_energy.nonlocal_hartree,
            "hartree_hartree": scf_result.final_total_energy.hartree_hartree,
            "exchange_correlation_hartree": scf_result.final_total_energy.exchange_correlation_hartree,
            "ion_ion_ewald_hartree": scf_result.final_total_energy.ion_ion_ewald_hartree,
            "band_energy_sum_hartree": scf_result.final_total_energy.band_energy_sum_hartree,
            "total_electronic_hartree": scf_result.final_total_energy.total_electronic_hartree,
            "total_crystal_hartree": scf_result.final_total_energy.total_crystal_hartree,
        },
        "notes": list(scf_result.final_total_energy.notes),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
