"""哈密顿量构造阶段的回归测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from realspace_dft import 初始化计算上下文
from realspace_dft.core.models import AtomSite, CrystalStructure, GridSpec, KPoint, NonlocalProjectorRadial, PseudopotentialInfo
from realspace_dft.mesh.geometry import 构造实空间网格
from realspace_dft.physics.nonlocal_operator import 构造非局域赝势算符


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
        expected_total = np.zeros_like(grid_wavefunction)
        for component_grid in components.values():
            expected_total += component_grid
        np.testing.assert_allclose(
            self.context.hamiltonian.matvec(flat_wavefunction),
            expected_total.reshape(-1, order="C"),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


class NonlocalOperatorTest(unittest.TestCase):
    """验证非局域赝势算符的最小行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        crystal = CrystalStructure(
            lattice_constant_bohr=6.0,
            lattice_constant_unit="bohr",
            normalized_lattice_vectors=np.eye(3, dtype=np.float64),
            atom_sites=(AtomSite(element="X", fractional_position=(0.0, 0.0, 0.0)),),
        )
        cls.grid = 构造实空间网格(crystal, GridSpec(shape=(4, 4, 4)))
        cls.pseudopotential = PseudopotentialInfo(
            file_path=Path("synthetic.upf"),
            element="X",
            functional="PBE",
            z_valence=1.0,
            pseudo_type="NC",
            radial_grid_bohr=np.array([0.0, 5.0], dtype=np.float64),
            local_potential_hartree=np.array([0.0, 0.0], dtype=np.float64),
            nonlocal_projectors=(
                NonlocalProjectorRadial(
                    index=1,
                    angular_momentum=0,
                    radial_values=np.array([1.0, 1.0], dtype=np.float64),
                    cutoff_radius_bohr=5.0,
                ),
            ),
            dij_matrix_hartree=np.array([[2.0]], dtype=np.float64),
        )
        cls.operator = 构造非局域赝势算符(
            real_space_grid=cls.grid,
            atom_sites=crystal.atom_sites,
            pseudopotentials={"X": cls.pseudopotential},
            kpoint=KPoint(fractional_coordinates=(0.0, 0.0, 0.0), weight=1.0),
        )

    def test_operator_is_built_from_nonlocal_projectors(self) -> None:
        self.assertIsNotNone(self.operator)
        self.assertEqual(self.operator.projector_count, 1)

    def test_apply_to_grid_is_linear(self) -> None:
        assert self.operator is not None
        psi_a = np.ones(self.grid.shape, dtype=np.complex128)
        psi_b = np.linspace(0.1, 1.6, self.grid.point_count).reshape(self.grid.shape).astype(np.complex128)
        combined = 2.0 * psi_a - 0.5 * psi_b

        applied_combined = self.operator.apply_to_grid(combined)
        applied_separate = 2.0 * self.operator.apply_to_grid(psi_a) - 0.5 * self.operator.apply_to_grid(psi_b)
        np.testing.assert_allclose(applied_combined, applied_separate, rtol=1.0e-12, atol=1.0e-12)

    def test_expectation_per_band_is_positive_for_positive_dij(self) -> None:
        assert self.operator is not None
        wavefunctions = np.ones(self.grid.shape + (1,), dtype=np.complex128)
        expectation = self.operator.expectation_per_band(wavefunctions)
        self.assertEqual(expectation.shape, (1,))
        self.assertGreater(expectation[0], 0.0)


if __name__ == "__main__":
    unittest.main()