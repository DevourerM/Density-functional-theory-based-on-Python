"""总能与 Ewald 求和的回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from realspace_dft.physics.ewald import 计算周期点电荷Ewald能


class EwaldEnergyTest(unittest.TestCase):
    """验证 Ewald 求和的基准数值。"""

    def test_cscl_madelung_reference_energy(self) -> None:
        cell_matrix_bohr = np.eye(3, dtype=np.float64)
        fractional_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0], dtype=np.float64)

        ewald_energy = 计算周期点电荷Ewald能(
            cell_matrix_bohr,
            fractional_positions,
            charges,
            relative_tolerance=1.0e-12,
        )

        nearest_neighbor_distance = np.sqrt(3.0) / 2.0
        madelung_constant_cscl = 1.762675
        reference_energy = -madelung_constant_cscl / nearest_neighbor_distance
        self.assertAlmostEqual(ewald_energy, reference_energy, places=6)


if __name__ == "__main__":
    unittest.main()
