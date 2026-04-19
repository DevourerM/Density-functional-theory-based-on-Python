"""最小 SCF 工作流的回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from realspace_dft import 执行完整计算流程
from realspace_dft import 运行SCF循环
from realspace_dft.core.models import KPoint, MixingConfig
from realspace_dft.physics.density import 计算费米能与占据, 计算轨道占据
from realspace_dft.solvers.mixing import create_density_mixer


class DensityMixingTest(unittest.TestCase):
    """验证最小线性混合器的基本行为。"""

    def test_linear_density_mixer_uses_input_coefficient(self) -> None:
        mixer = create_density_mixer(
            MixingConfig(method="linear", linear_coefficient=0.25)
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

    def test_fermi_dirac_occupations_match_total_electrons(self) -> None:
        occupations, fermi_energy = 计算费米能与占据(
            np.array([[0.0, 0.5, 1.0], [0.1, 0.6, 1.1]], dtype=np.float64),
            (
                KPoint(fractional_coordinates=(0.0, 0.0, 0.0), weight=0.5),
                KPoint(fractional_coordinates=(0.5, 0.0, 0.0), weight=0.5),
            ),
            total_electrons=2.0,
            smearing_width_hartree=0.05,
            spin_degeneracy=2.0,
        )
        self.assertEqual(occupations.shape, (2, 3))
        self.assertTrue(np.isfinite(fermi_energy))
        self.assertAlmostEqual(0.5 * np.sum(occupations[0]) + 0.5 * np.sum(occupations[1]), 2.0, places=6)


class SCFWorkflowTest(unittest.TestCase):
    """验证最小 SCF 主循环能读取输入并保持电荷守恒。"""

    def test_scf_workflow_reads_input_parameters_and_preserves_charge(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "INPUT.json").open("r", encoding="utf-8") as file:
            input_data = json.load(file)

        input_data["路径设置"]["赝势文件夹路径"] = str(
            project_root / "SG15-Version1p0_Pseudopotential" / "SG15_ONCV_v1.0_upf"
        ).replace("\\", "/")
        input_data["SCF控制"]["最大迭代次数"] = 2
        input_data["SCF控制"]["SCF收敛阈值"] = 1.0e-6
        input_data["SCF控制"]["密度混合"]["方法"] = "linear"
        input_data["SCF控制"]["密度混合"]["线性混合系数"] = 0.4
        input_data["数值设置"]["空间网格划分"] = {
            "x方向网格数": 4,
            "y方向网格数": 4,
            "z方向网格数": 4,
        }
        input_data["数值设置"]["轨道数"] = 4
        input_data["数值设置"]["k点"] = [
            {"坐标": [0.0, 0.0, 0.0], "权重": 1.0}
        ]
        input_data["数值设置"]["占据设置"] = {
            "方法": "FERMI_DIRAC",
            "展宽哈特里": 0.02,
            "自旋简并度": 2.0,
        }

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "small_input.json"
            with input_path.open("w", encoding="utf-8") as file:
                json.dump(input_data, file, ensure_ascii=False, indent=2)

            scf_result = 运行SCF循环(input_path)

        self.assertEqual(scf_result.max_scf_iterations_used, 2)
        self.assertAlmostEqual(scf_result.scf_tolerance_used, 1.0e-6)
        self.assertAlmostEqual(scf_result.linear_mixing_coefficient_used, 0.4)
        self.assertEqual(scf_result.final_eigensolution.nbands, 4)
        self.assertLessEqual(scf_result.iteration_count, 2)
        self.assertAlmostEqual(scf_result.final_density.integrated_electrons, 8.0, places=8)
        self.assertEqual(scf_result.occupations.shape, (1, 4))
        self.assertTrue(np.isfinite(scf_result.fermi_energy_hartree))
        self.assertEqual(scf_result.eigensolver_method_used, "LOBPCG")
        self.assertIsNotNone(scf_result.final_total_energy)
        self.assertEqual(len(scf_result.iterations), scf_result.iteration_count)
        self.assertTrue(all(np.isfinite(record.total_energy.total_hartree) for record in scf_result.iterations))
        self.assertTrue(
            all(
                record.eigenvalue_change_max is None or np.isfinite(record.eigenvalue_change_max)
                for record in scf_result.iterations
            )
        )

    def test_minimal_pipeline_returns_context_and_scf_result(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "INPUT.json").open("r", encoding="utf-8") as file:
            input_data = json.load(file)

        input_data["路径设置"]["赝势文件夹路径"] = str(
            project_root / "SG15-Version1p0_Pseudopotential" / "SG15_ONCV_v1.0_upf"
        ).replace("\\", "/")
        input_data["SCF控制"]["最大迭代次数"] = 2
        input_data["SCF控制"]["SCF收敛阈值"] = 1.0e-6
        input_data["数值设置"]["空间网格划分"] = {
            "x方向网格数": 4,
            "y方向网格数": 4,
            "z方向网格数": 4,
        }
        input_data["数值设置"]["轨道数"] = 4
        input_data["数值设置"]["k点"] = [
            {"坐标": [0.0, 0.0, 0.0], "权重": 1.0}
        ]
        input_data["数值设置"]["占据设置"] = {
            "方法": "FERMI_DIRAC",
            "展宽哈特里": 0.02,
            "自旋简并度": 2.0,
        }

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "small_input.json"
            log_dir = Path(temporary_directory) / "logs"
            output_dir = Path(temporary_directory) / "outputs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "stale.txt").write_text("stale", encoding="utf-8")
            with input_path.open("w", encoding="utf-8") as file:
                json.dump(input_data, file, ensure_ascii=False, indent=2)

            pipeline_result = 执行完整计算流程(
                input_path,
                log_dir=log_dir,
                output_dir=output_dir,
                clear_logs=True,
            )
            self.assertEqual(pipeline_result.context.real_space_grid.shape, (4, 4, 4))
            self.assertEqual(pipeline_result.scf_result.final_density.grid_shape, (4, 4, 4))
            self.assertFalse((log_dir / "stale.txt").exists())
            self.assertTrue(pipeline_result.log_dir.exists())
            self.assertTrue(any(pipeline_result.log_dir.iterdir()))
            self.assertTrue(pipeline_result.density_txt_path.exists())
            density_text = pipeline_result.density_txt_path.read_text(encoding="utf-8")
            self.assertIn("density_bohr^-3", density_text)
            self.assertAlmostEqual(
                pipeline_result.scf_result.final_total_energy.total_hartree,
                pipeline_result.scf_result.final_total_energy_hartree,
                places=10,
            )
            self.assertEqual(pipeline_result.scf_result.config.kpoint_count, 1)
