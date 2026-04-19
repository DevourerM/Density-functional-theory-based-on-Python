"""哈密顿量构造阶段的回归测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from realspace_dft import 初始化计算上下文


class HamiltonianConstructionTest(unittest.TestCase):
    """验证三维实空间哈密顿量的基础结构。"""

    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cls.context = 初始化计算上下文(project_root / "INPUT.json")

    def test_kinetic_term_annihilates_constant_wavefunction(self) -> None:
        constant_wavefunction = np.ones(
            self.context.real_space_grid.shape,
            dtype=np.complex128,
        )
        kinetic_result = self.context.hamiltonian_components.kinetic.apply_to_grid(
            constant_wavefunction
        )
        self.assertLess(np.max(np.abs(kinetic_result)), 1.0e-12)

    def test_effective_local_potential_is_explicit_sum(self) -> None:
        components = self.context.hamiltonian_components
        expected = (
            components.ionic_local_potential.values
            + components.hartree_potential.values
            + components.exchange_correlation_potential.values
        )
        np.testing.assert_allclose(
            components.effective_local_potential.values,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_total_operator_matches_componentwise_application(self) -> None:
        flat_wavefunction = np.linspace(
            1.0,
            2.0,
            self.context.real_space_grid.point_count,
            dtype=np.float64,
        ).astype(np.complex128)
        grid_wavefunction = flat_wavefunction.reshape(self.context.real_space_grid.shape)

        components = self.context.hamiltonian.apply_components_to_grid(grid_wavefunction)
        expected_total = (
            components["动能项"]
            + components["离子局域势项"]
            + components["Hartree势项"]
            + components["交换关联势项"]
        )
        if "非局域赝势项" in components:
            expected_total = expected_total + components["非局域赝势项"]
        np.testing.assert_allclose(
            self.context.hamiltonian.matvec(flat_wavefunction),
            expected_total.reshape(-1, order="C"),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()