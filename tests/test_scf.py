"""SCF 密度重构与密度迭代的回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from realspace_dft import 执行完整计算流程
from realspace_dft import 运行SCF循环
from realspace_dft.core.models import MixingConfig
from realspace_dft.physics.density import 计算轨道占据
from realspace_dft.solvers.mixing import create_density_mixer


class DensityMixingTest(unittest.TestCase):
    """验证线性和 DIIS 密度混合器的基本行为。"""

    def test_linear_density_mixer_uses_input_coefficient(self) -> None:
        mixer = create_density_mixer(
            MixingConfig(method="linear", linear_coefficient=0.25, diis_history_steps=4)
        )
        density_in = np.array([1.0, 3.0], dtype=np.float64)
        density_out = np.array([3.0, 7.0], dtype=np.float64)

        np.testing.assert_allclose(
            mixer.mix(density_in, density_out),
            np.array([1.5, 4.0], dtype=np.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_occupation_numbers_fill_two_electrons_per_band(self) -> None:
        np.testing.assert_allclose(
            计算轨道占据(7.0, 4),
            np.array([2.0, 2.0, 2.0, 1.0], dtype=np.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


class SCFWorkflowTest(unittest.TestCase):
    """验证 SCF 主循环能读取 input 参数并执行密度迭代。"""

    def test_scf_workflow_reads_input_parameters_and_preserves_charge(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "INPUT.json").open("r", encoding="utf-8") as file:
            input_data = json.load(file)

        input_data["路径设置"]["赝势文件夹路径"] = str(
            project_root / "SG15-Version1p0_Pseudopotential" / "SG15_ONCV_v1.0_upf"
        ).replace("\\", "/")
        input_data["SCF控制"]["最大迭代次数"] = 2
        input_data["SCF控制"]["SCF收敛阈值"] = 1.0e-12
        input_data["SCF控制"]["波函数收敛阈值"] = 1.0e-7
        input_data["SCF控制"]["密度混合"]["方法"] = "linear"
        input_data["SCF控制"]["密度混合"]["线性混合系数"] = 0.4
        input_data["数值设置"]["空间网格划分"] = {
            "x方向网格数": 4,
            "y方向网格数": 4,
            "z方向网格数": 4,
        }
        input_data["数值设置"]["轨道数"] = 4

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "small_input.json"
            with input_path.open("w", encoding="utf-8") as file:
                json.dump(input_data, file, ensure_ascii=False, indent=2)

            scf_result = 运行SCF循环(input_path)

        self.assertEqual(scf_result.max_scf_iterations_used, 2)
        self.assertAlmostEqual(scf_result.scf_tolerance_used, 1.0e-12)
        self.assertAlmostEqual(scf_result.wavefunction_tolerance_used, 1.0e-7)
        self.assertEqual(scf_result.density_mixer_method, "linear")
        self.assertAlmostEqual(scf_result.linear_mixing_coefficient_used, 0.4)
        self.assertEqual(scf_result.diis_history_steps_used, 6)
        self.assertEqual(scf_result.final_eigensolution.nbands, 4)
        self.assertLessEqual(scf_result.iteration_count, 2)
        self.assertAlmostEqual(scf_result.final_density.integrated_electrons, 8.0, places=8)
        self.assertEqual(scf_result.occupations.shape, (4,))
        self.assertEqual(scf_result.eigensolver_method_used, "LOBPCG")
        self.assertIsNotNone(scf_result.final_total_energy)
        self.assertEqual(len(scf_result.iterations), scf_result.iteration_count)
        self.assertTrue(all(record.total_energy.total_electronic_hartree == record.total_energy.total_electronic_hartree for record in scf_result.iterations))
        self.assertTrue(np.isfinite(scf_result.final_total_energy.ion_ion_ewald_hartree))
        self.assertNotEqual(scf_result.final_total_energy.ion_ion_ewald_hartree, 0.0)
        self.assertAlmostEqual(
            scf_result.final_total_energy.total_crystal_hartree,
            scf_result.final_total_energy.total_electronic_hartree
            + scf_result.final_total_energy.ion_ion_ewald_hartree,
            places=10,
        )

    def test_full_pipeline_clears_logs_and_saves_density(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "INPUT.json").open("r", encoding="utf-8") as file:
            input_data = json.load(file)

        input_data["路径设置"]["赝势文件夹路径"] = str(
            project_root / "SG15-Version1p0_Pseudopotential" / "SG15_ONCV_v1.0_upf"
        ).replace("\\", "/")
        input_data["SCF控制"]["最大迭代次数"] = 2
        input_data["SCF控制"]["SCF收敛阈值"] = 1.0e-12
        input_data["SCF控制"]["波函数收敛阈值"] = 1.0e-7
        input_data["数值设置"]["空间网格划分"] = {
            "x方向网格数": 4,
            "y方向网格数": 4,
            "z方向网格数": 4,
        }
        input_data["数值设置"]["轨道数"] = 4

        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "small_input.json"
            log_dir = temporary_path / "logs"
            output_dir = temporary_path / "outputs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "stale.txt").write_text("stale", encoding="utf-8")

            with input_path.open("w", encoding="utf-8") as file:
                json.dump(input_data, file, ensure_ascii=False, indent=2)

            pipeline_result = 执行完整计算流程(
                input_path=input_path,
                log_dir=log_dir,
                output_dir=output_dir,
                clear_logs=True,
            )

            self.assertFalse((log_dir / "stale.txt").exists())
            self.assertTrue(pipeline_result.density_file_path.exists())
            self.assertTrue(pipeline_result.summary_file_path.exists())
            self.assertGreater(len(list(log_dir.iterdir())), 0)

            with np.load(pipeline_result.density_file_path) as density_data:
                self.assertAlmostEqual(float(density_data["integrated_electrons"]), 8.0, places=8)
                self.assertAlmostEqual(
                    float(density_data["total_energy_hartree"]),
                    pipeline_result.scf_result.final_total_energy_hartree,
                    places=10,
                )
                self.assertAlmostEqual(
                    float(density_data["total_electronic_energy_hartree"]),
                    pipeline_result.scf_result.final_total_electronic_energy_hartree,
                    places=10,
                )
