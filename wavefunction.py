from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from hamiltonian import (
    Hamiltonian,
    infer_electron_count,
    infer_real_space_grid,
    read_input_json,
)


class WaveFunction:
    def __init__(self, config: Dict[str, Any], total_electrons: float | None = None) -> None:
        self.total_electrons = infer_electron_count(config) if total_electrons is None else float(total_electrons)
        self.grid_shape = infer_real_space_grid(config)
        filled_bands = max(1, math.ceil(max(self.total_electrons, 0.0) / 2.0))
        requested_bands = int(config.get("nbands", filled_bands))
        self.nbands = max(filled_bands, requested_bands, 1)
        self.eigensolver_tolerance = float(config.get("eigensolver_tolerance", 1.0e-7))
        self.eigensolver_maxiter = int(config.get("eigensolver_maxiter", 1200))
        self._kinetic_operator_cache: dict[tuple[Any, ...], sparse.csr_matrix] = {}

    def _build_1d_laplacian(
        self,
        point_count: int,
        spacing: float,
        phase: complex,
    ) -> sparse.csr_matrix:
        main_diagonal = -2.0 * np.ones(point_count, dtype=float)
        off_diagonal = np.ones(point_count - 1, dtype=float)
        laplacian = sparse.diags(
            [off_diagonal, main_diagonal, off_diagonal],
            offsets=[-1, 0, 1],
            shape=(point_count, point_count),
            dtype=complex,
            format="lil",
        )
        laplacian[0, point_count - 1] = np.conjugate(phase)
        laplacian[point_count - 1, 0] = phase
        return laplacian.tocsr() / (spacing * spacing)

    def _get_kinetic_operator(self, hamiltonian_data: Dict[str, Any]) -> sparse.csr_matrix:
        grid_shape = tuple(int(value) for value in hamiltonian_data["grid_shape"])
        grid_spacing = tuple(float(value) for value in hamiltonian_data["grid_spacing"])
        k_point = tuple(float(value) for value in hamiltonian_data.get("k_point", [0.0, 0.0, 0.0]))
        cache_key = (grid_shape, grid_spacing, k_point)
        if cache_key in self._kinetic_operator_cache:
            return self._kinetic_operator_cache[cache_key]

        x_phase = complex(np.exp(2.0j * np.pi * k_point[0]))
        y_phase = complex(np.exp(2.0j * np.pi * k_point[1]))
        z_phase = complex(np.exp(2.0j * np.pi * k_point[2]))

        laplacian_x = self._build_1d_laplacian(grid_shape[0], grid_spacing[0], x_phase)
        laplacian_y = self._build_1d_laplacian(grid_shape[1], grid_spacing[1], y_phase)
        laplacian_z = self._build_1d_laplacian(grid_shape[2], grid_spacing[2], z_phase)

        identity_x = sparse.identity(grid_shape[0], dtype=complex, format="csr")
        identity_y = sparse.identity(grid_shape[1], dtype=complex, format="csr")
        identity_z = sparse.identity(grid_shape[2], dtype=complex, format="csr")

        kinetic_operator = -0.5 * (
            sparse.kron(sparse.kron(laplacian_x, identity_y), identity_z, format="csr")
            + sparse.kron(sparse.kron(identity_x, laplacian_y), identity_z, format="csr")
            + sparse.kron(sparse.kron(identity_x, identity_y), laplacian_z, format="csr")
        )
        self._kinetic_operator_cache[cache_key] = kinetic_operator
        return kinetic_operator

    def _compute_occupations(self, state_count: int) -> list[float]:
        occupations: list[float] = []
        remaining_electrons = max(self.total_electrons, 0.0)
        for _ in range(state_count):
            if remaining_electrons <= 0.0:
                occupations.append(0.0)
                continue
            occupation = min(2.0, remaining_electrons)
            occupations.append(occupation)
            remaining_electrons -= occupation
        return occupations

    def assign_kpoint_occupations(
        self,
        solver_results: Sequence[Dict[str, Any]],
        k_weights: Sequence[float],
    ) -> list[list[float]]:
        occupations = [
            [0.0 for _ in solver_result["eigenvalues"]]
            for solver_result in solver_results
        ]
        state_energies: list[tuple[float, int, int, float]] = []
        for k_index, (solver_result, weight) in enumerate(zip(solver_results, k_weights)):
            for band_index, energy in enumerate(solver_result["eigenvalues"]):
                state_energies.append((float(energy), k_index, band_index, float(weight)))

        remaining_electrons = max(self.total_electrons, 0.0)
        for _, k_index, band_index, weight in sorted(state_energies, key=lambda item: item[0]):
            if remaining_electrons <= 0.0:
                break
            state_capacity = 2.0 * weight
            occupied_electrons = min(state_capacity, remaining_electrons)
            occupations[k_index][band_index] = occupied_electrons / max(weight, 1.0e-14)
            remaining_electrons -= occupied_electrons
        return occupations

    def attach_occupations(
        self,
        solver_result: Dict[str, Any],
        occupations: Sequence[float],
    ) -> Dict[str, Any]:
        solver_result["occupations"] = [float(value) for value in occupations]
        for state, occupation in zip(solver_result["wavefunctions"], occupations):
            state["occupation"] = float(occupation)
        return solver_result

    def _build_nonlocal_operator(self, hamiltonian_data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        projector_matrix = np.asarray(
            hamiltonian_data.get("nonlocal_projector_matrix", np.zeros((0, 0), dtype=complex)),
            dtype=complex,
        )
        d_matrix = np.asarray(
            hamiltonian_data.get("nonlocal_d_matrix", np.zeros((0, 0), dtype=complex)),
            dtype=complex,
        )
        volume_element = float(hamiltonian_data["volume_element"])
        return projector_matrix, d_matrix, volume_element

    def solve(self, hamiltonian_input: Dict[str, Any] | Sequence[Sequence[complex]]) -> Dict[str, Any]:
        if isinstance(hamiltonian_input, dict):
            kinetic_operator = self._get_kinetic_operator(hamiltonian_input)
            effective_potential = np.array(hamiltonian_input["potentials"]["effective"], dtype=float).reshape(-1)
            projector_matrix, d_matrix, volume_element = self._build_nonlocal_operator(hamiltonian_input)
            matrix_size = kinetic_operator.shape[0]
            band_count = min(self.nbands, max(1, matrix_size - 2))

            def matvec(vector: np.ndarray) -> np.ndarray:
                result = kinetic_operator @ vector
                result = result + effective_potential * vector
                if projector_matrix.shape[1] > 0:
                    overlaps = volume_element * (projector_matrix.conjugate().T @ vector)
                    result = result + projector_matrix @ (d_matrix @ overlaps)
                return result

            total_operator = sparse_linalg.LinearOperator(
                shape=(matrix_size, matrix_size),
                matvec=matvec,
                dtype=complex,
            )

            if band_count >= matrix_size - 1:
                dense_operator = kinetic_operator.toarray() + np.diag(effective_potential.astype(complex))
                if projector_matrix.shape[1] > 0:
                    dense_operator = dense_operator + volume_element * (
                        projector_matrix @ d_matrix @ projector_matrix.conjugate().T
                    )
                eigenvalues, eigenvectors = np.linalg.eigh(dense_operator)
            else:
                eigenvalues, eigenvectors = sparse_linalg.eigsh(
                    total_operator,
                    k=band_count,
                    which="SA",
                    tol=self.eigensolver_tolerance,
                    maxiter=self.eigensolver_maxiter,
                )
        else:
            total_matrix = np.array(hamiltonian_input, dtype=complex)
            eigenvalues, eigenvectors = np.linalg.eigh(total_matrix)
            matrix_size = total_matrix.shape[0]
            band_count = min(self.nbands, matrix_size)
            eigenvalues = eigenvalues[:band_count]
            eigenvectors = eigenvectors[:, :band_count]

        sort_indices = np.argsort(eigenvalues.real)
        eigenvalues = np.array(eigenvalues[sort_indices], dtype=float)
        eigenvectors = np.array(eigenvectors[:, sort_indices], dtype=complex)
        occupations = self._compute_occupations(len(eigenvalues))

        states = []
        for band_index, energy in enumerate(eigenvalues.tolist()):
            band_coefficients = eigenvectors[:, band_index]
            coefficients = [complex(value) for value in band_coefficients.tolist()]
            states.append(
                {
                    "band_index": band_index,
                    "energy": float(energy),
                    "coefficients": coefficients,
                    "probability": (np.abs(band_coefficients) ** 2).astype(float).tolist(),
                    "occupation": occupations[band_index],
                }
            )

        result = {
            "solver": "scipy.sparse.linalg.eigsh",
            "basis": "real-space grid",
            "grid_shape": list(self.grid_shape),
            "k_point": list(hamiltonian_input.get("k_point", [0.0, 0.0, 0.0])) if isinstance(hamiltonian_input, dict) else [0.0, 0.0, 0.0],
            "eigenvalues": eigenvalues.tolist(),
            "occupations": occupations,
            "wavefunctions": states,
        }
        if isinstance(hamiltonian_input, dict):
            result["hamiltonian"] = hamiltonian_input
        return result

    def compute_energy_expectations(
        self,
        hamiltonian_data: Dict[str, Any],
        solver_result: Dict[str, Any],
        occupations: Sequence[float] | None = None,
        k_weight: float = 1.0,
    ) -> Dict[str, float]:
        coefficients = np.array(
            [state["coefficients"] for state in solver_result["wavefunctions"]],
            dtype=complex,
        ).T
        occupation_array = np.array(
            solver_result["occupations"] if occupations is None else occupations,
            dtype=float,
        )
        kinetic_operator = self._get_kinetic_operator(hamiltonian_data)
        projector_matrix, d_matrix, volume_element = self._build_nonlocal_operator(hamiltonian_data)

        kinetic_energy = 0.0
        nonlocal_energy = 0.0
        band_energy = 0.0
        for band_index, occupation in enumerate(occupation_array):
            if occupation <= 0.0:
                continue
            state_vector = coefficients[:, band_index]
            band_energy += float(k_weight * occupation * solver_result["eigenvalues"][band_index])
            kinetic_energy += float(
                k_weight * occupation * np.real(np.vdot(state_vector, kinetic_operator @ state_vector))
            )
            if projector_matrix.shape[1] > 0:
                overlaps = volume_element * (projector_matrix.conjugate().T @ state_vector)
                nonlocal_energy += float(
                    k_weight
                    * occupation
                    * np.real(np.vdot(overlaps, d_matrix @ overlaps) / max(volume_element, 1.0e-14))
                )

        return {
            "band": band_energy,
            "kinetic": kinetic_energy,
            "nonlocal": nonlocal_energy,
        }

    def _sample_periodic_grid(self, grid: np.ndarray, fractional_position: Sequence[float]) -> float:
        coordinates = [
            (float(fractional_position[axis]) % 1.0) * grid.shape[axis]
            for axis in range(3)
        ]
        base_indices = [
            int(np.floor(coordinate)) % grid.shape[axis]
            for axis, coordinate in enumerate(coordinates)
        ]
        weights = [coordinate - np.floor(coordinate) for coordinate in coordinates]

        interpolated_value = 0.0
        for x_offset in (0, 1):
            x_weight = (1.0 - weights[0]) if x_offset == 0 else weights[0]
            x_index = (base_indices[0] + x_offset) % grid.shape[0]
            for y_offset in (0, 1):
                y_weight = (1.0 - weights[1]) if y_offset == 0 else weights[1]
                y_index = (base_indices[1] + y_offset) % grid.shape[1]
                for z_offset in (0, 1):
                    z_weight = (1.0 - weights[2]) if z_offset == 0 else weights[2]
                    z_index = (base_indices[2] + z_offset) % grid.shape[2]
                    interpolated_value += (
                        x_weight
                        * y_weight
                        * z_weight
                        * float(grid[x_index, y_index, z_index])
                    )
        return interpolated_value

    def density_from_grid_density(
        self,
        hamiltonian_data: Dict[str, Any],
        grid_density: np.ndarray | Sequence[float],
    ) -> Dict[str, Any]:
        grid_shape = tuple(int(value) for value in hamiltonian_data["grid_shape"])
        density_array = np.array(grid_density, dtype=float).reshape(grid_shape)
        volume_element = float(hamiltonian_data["volume_element"])
        total_density = float(density_array.sum() * volume_element)
        if total_density > 0.0 and self.total_electrons > 0.0:
            density_array = density_array * (self.total_electrons / total_density)

        site_density = [
            self._sample_periodic_grid(density_array, fractional_position)
            for fractional_position in hamiltonian_data["fractional_positions"]
        ]
        return {
            "site_density": site_density,
            "grid_shape": list(grid_shape),
            "grid_density": density_array,
            "fractional_grid": np.array(hamiltonian_data["fractional_grid"], dtype=float),
            "cartesian_grid": np.array(hamiltonian_data["cartesian_grid"], dtype=float),
        }

    def solve_density(
        self,
        hamiltonian_data: Dict[str, Any],
        solver_result: Dict[str, Any],
        occupations: Sequence[float] | None = None,
        k_weight: float = 1.0,
    ) -> Dict[str, Any]:
        coefficients = np.array(
            [state["coefficients"] for state in solver_result["wavefunctions"]],
            dtype=complex,
        ).T
        occupation_array = np.array(
            solver_result["occupations"] if occupations is None else occupations,
            dtype=float,
        )
        volume_element = float(hamiltonian_data["volume_element"])
        density_flat = (
            k_weight
            * np.sum((np.abs(coefficients) ** 2) * occupation_array[np.newaxis, :], axis=1)
            / volume_element
        )
        density_grid = density_flat.reshape(tuple(int(value) for value in hamiltonian_data["grid_shape"]))
        return self.density_from_grid_density(hamiltonian_data, density_grid)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    input_path = Path(arguments[0]) if arguments else Path("INPUT.json")

    config = read_input_json(input_path)
    hamiltonian = Hamiltonian(config)
    wavefunction = WaveFunction(config, total_electrons=hamiltonian.total_electrons)
    hamiltonian_data = hamiltonian.build(hamiltonian.initial_density())
    result = wavefunction.solve(hamiltonian_data)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())