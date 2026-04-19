"""初始化阶段的最小回归测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from realspace_dft import 初始化计算上下文


class BootstrapInitializationTest(unittest.TestCase):
    """确保输入读取、赝势定位和初始电荷密度初始化能串起来。"""

    def test_project_input_can_be_initialized(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        context = 初始化计算上下文(project_root / "INPUT.json")

        self.assertEqual(context.config.numerical.nbands, 8)
        self.assertEqual(context.minimum_occupied_bands, 4)
        self.assertEqual(context.pseudopotentials["Si"].functional, "PBE")
        self.assertEqual(context.density_grid.grid_shape, (64, 64, 64))
        self.assertEqual(context.real_space_grid.shape, (64, 64, 64))
        self.assertAlmostEqual(context.total_valence_electrons, 8.0)
        self.assertAlmostEqual(context.density_grid.integrated_electrons, 8.0)
        self.assertEqual(context.hamiltonian.shape, (64 * 64 * 64, 64 * 64 * 64))
        self.assertTrue(np.isfinite(context.ion_ion_ewald_hartree))
        self.assertNotEqual(context.ion_ion_ewald_hartree, 0.0)
        self.assertEqual(
            context.hamiltonian_components.effective_local_potential.values.shape,
            (64, 64, 64),
        )
        self.assertTrue(context.pseudopotentials["Si"].has_nonlocal_projectors)
        self.assertIsNotNone(context.hamiltonian_components.nonlocal_pseudopotential)


if __name__ == "__main__":
    unittest.main()
