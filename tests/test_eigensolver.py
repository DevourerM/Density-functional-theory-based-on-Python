"""波函数迭代求解器的回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from realspace_dft import 本征求解配置, 求解本征波函数


class ToyDiagonalHamiltonian(LinearOperator):
    """用于测试求解器接口的简单双对角哈密顿量。"""

    def __init__(self, diagonal: np.ndarray, grid_shape: tuple[int, int, int]):
        self._diagonal = np.asarray(diagonal, dtype=np.float64)
        self.grid = SimpleNamespace(
            shape=grid_shape,
            volume_element_bohr3=1.0 / self._diagonal.size,
        )
        super().__init__(dtype=np.float64, shape=(self._diagonal.size, self._diagonal.size))

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        return self._diagonal * vector

    def _matmat(self, matrix: np.ndarray) -> np.ndarray:
        return self._diagonal[:, np.newaxis] * matrix

    def approximate_diagonal(self) -> np.ndarray:
        return self._diagonal.copy()


class IterativeEigenSolverTest(unittest.TestCase):
    """验证 LOBPCG 波函数求解器的输入输出契约。"""

    def test_lobpcg_returns_lowest_nbands_wavefunctions(self) -> None:
        diagonal = np.arange(1.0, 65.0, dtype=np.float64)
        hamiltonian = ToyDiagonalHamiltonian(diagonal, grid_shape=(4, 4, 4))
        result = 求解本征波函数(
            hamiltonian,
            4,
            solver_config=本征求解配置(
                tolerance=1.0e-9,
                max_iterations=80,
                random_seed=7,
            ),
        )

        np.testing.assert_allclose(
            result.eigenvalues_hartree,
            np.array([1.0, 2.0, 3.0, 4.0]),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        self.assertEqual(result.wavefunctions.shape, (4, 64))
        self.assertEqual(result.wavefunctions_as_grids().shape, (4, 4, 4, 4))

        norms = np.sum(np.abs(result.wavefunctions) ** 2, axis=1) * result.volume_element_bohr3
        np.testing.assert_allclose(norms, np.ones(4), atol=1.0e-8, rtol=1.0e-8)

        overlap = result.overlap_matrix()
        np.testing.assert_allclose(overlap, np.eye(4), atol=1.0e-8, rtol=1.0e-8)
