from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from scipy import special

from dft_support import (
    evaluate_exchange_correlation_potential,
    find_local_pseudopotential_file,
    generate_monkhorst_pack_grid,
    infer_k_grid,
    load_norm_conserving_pseudopotential,
    parse_atom_sites,
    spherical_harmonic,
)


def read_input_json(input_path: str | Path) -> Dict[str, Any]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_lattice_vectors(config: Dict[str, Any]) -> np.ndarray:
    if "lattice_vectors" in config:
        lattice_vectors = np.array(config["lattice_vectors"], dtype=float)
        if lattice_vectors.shape != (3, 3):
            raise ValueError("lattice_vectors 必须是 3x3 矩阵。")
        return lattice_vectors

    lattice_constant = config.get("lattice_constant", config.get("a"))
    lattice_directions = config.get("lattice_directions", config.get("vector"))
    if lattice_constant is None or lattice_directions is None:
        raise ValueError("请提供 lattice_vectors，或同时提供 lattice_constant 和 lattice_directions。")

    lattice_constant = float(lattice_constant)
    lattice_vectors = np.array(
        [
            [lattice_constant * float(component) for component in basis]
            for basis in lattice_directions
        ],
        dtype=float,
    )
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_directions 必须是 3x3 矩阵。")
    return lattice_vectors


def infer_electron_count(config: Dict[str, Any]) -> float:
    configured_electron_count = config.get("electron_count")
    if configured_electron_count is not None:
        return float(configured_electron_count)

    default_charge = float(config.get("default_charge", 0.0))
    site_charges = []
    for site in config.get("pos", []):
        if isinstance(site, dict):
            site_charges.append(float(site.get("charge", default_charge)))
        elif isinstance(site, Sequence) and not isinstance(site, (str, bytes)):
            if len(site) == 4 and not isinstance(site[0], str):
                site_charges.append(float(site[3]))
            elif len(site) == 5 and isinstance(site[0], str):
                site_charges.append(float(site[4]))
            else:
                site_charges.append(default_charge)
        else:
            site_charges.append(default_charge)

    positive_charge = sum(max(charge, 0.0) for charge in site_charges)
    if positive_charge > 0.0:
        return positive_charge

    absolute_charge = sum(abs(charge) for charge in site_charges)
    return absolute_charge


def infer_real_space_grid(config: Dict[str, Any]) -> tuple[int, int, int]:
    configured_grid = config.get("real_space_grid", config.get("density_grid", [12, 12, 12]))
    if not isinstance(configured_grid, Sequence) or isinstance(configured_grid, (str, bytes)):
        raise ValueError("real_space_grid 或 density_grid 必须是长度为 3 的序列。")
    if len(configured_grid) != 3:
        raise ValueError("real_space_grid 或 density_grid 必须包含 3 个维度。")
    return tuple(max(int(value), 2) for value in configured_grid)


class Hamiltonian:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.lattice_vectors = infer_lattice_vectors(config)
        self.atom_sites = parse_atom_sites(config)
        self.elements = [site.element for site in self.atom_sites]
        self.fractional_positions = [list(site.fractional_position) for site in self.atom_sites]
        self.site_count = len(self.atom_sites)

        self.grid_shape = infer_real_space_grid(config)
        self.k_grid = infer_k_grid(config)
        self.k_points, self.k_weights = generate_monkhorst_pack_grid(self.k_grid)
        self.k_point = self.k_points[0]
        self.xc_functional = str(config.get("xc_functional", config.get("functional", "PBE"))).upper()
        self.initial_density_values = config.get("initial_density")
        self.initial_density_seed = config.get("initial_density_seed")
        configured_noise_grid = config.get("initial_density_noise_grid", self.grid_shape)
        if not isinstance(configured_noise_grid, Sequence) or isinstance(configured_noise_grid, (str, bytes)):
            raise ValueError("initial_density_noise_grid 必须是长度为 3 的序列。")
        if len(configured_noise_grid) != 3:
            raise ValueError("initial_density_noise_grid 必须包含 3 个维度。")
        self.initial_density_noise_grid = tuple(max(int(value), 2) for value in configured_noise_grid)

        self.pseudopotential_dir = Path(
            config.get(
                "pseudopotential_dir",
                Path(__file__).resolve().parent
                / "SG15-Version1p0_Pseudopotential"
                / "SG15_ONCV_v1.0_upf",
            )
        )
        self.ionic_potential_scale = float(config.get("ionic_potential_scale", 1.0))
        self.hartree_scale = float(config.get("hartree_scale", 1.0))
        self.xc_scale = float(config.get("xc_scale", 1.0))

        self.pseudopotentials: dict[str, Any] = {}
        self.pseudopotential_warnings: list[str] = []
        for element in sorted(set(self.elements)):
            pseudo_path = find_local_pseudopotential_file(element, self.xc_functional, self.pseudopotential_dir)
            pseudopotential = load_norm_conserving_pseudopotential(str(pseudo_path))
            self.pseudopotentials[element] = pseudopotential
            if pseudopotential.functional.upper() != self.xc_functional:
                self.pseudopotential_warnings.append(
                    f"{element}: requested XC={self.xc_functional}, but pseudopotential functional={pseudopotential.functional}"
                )

        self.valence_charges = [self.pseudopotentials[element].z_valence for element in self.elements]
        configured_electron_count = config.get("electron_count")
        self.total_electrons = (
            float(configured_electron_count)
            if configured_electron_count is not None
            else float(sum(self.valence_charges))
        )
        self.ionic_charges = np.array(self.valence_charges, dtype=float)

        self.cartesian_positions = np.array(
            [self._fractional_to_cartesian(site.fractional_position) for site in self.atom_sites],
            dtype=float,
        )
        self.cell_volume = float(abs(np.linalg.det(self.lattice_vectors)))
        self.reciprocal_lattice_vectors = 2.0 * np.pi * np.linalg.inv(self.lattice_vectors).T
        self.grid_point_count = int(np.prod(self.grid_shape))
        self.volume_element = self.cell_volume / self.grid_point_count
        self.grid_spacing = np.array(
            [
                float(np.linalg.norm(self.lattice_vectors[axis]) / self.grid_shape[axis])
                for axis in range(3)
            ],
            dtype=float,
        )
        self.fractional_grid, self.cartesian_grid = self._build_real_space_grid(self.grid_shape)
        self.ionic_potential = self._build_ionic_potential()
        (
            self.nonlocal_projector_matrix,
            self.nonlocal_d_matrix,
            self.nonlocal_projector_metadata,
        ) = self._build_nonlocal_projectors()
        self._poisson_kernel = self._build_poisson_kernel()
        self.ewald_parameters = self._build_ewald_parameters()
        self.ion_ion_energy = self._compute_ion_ion_energy()
        self._initial_density_cache: np.ndarray | None = None

    def _fractional_to_cartesian(self, fractional_position: Sequence[float]) -> np.ndarray:
        return np.einsum("i,ij->j", np.array(fractional_position, dtype=float), self.lattice_vectors)

    def _build_real_space_grid(
        self,
        grid_shape: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        axes = [
            (np.arange(point_count, dtype=float) + 0.5) / point_count
            for point_count in grid_shape
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        fractional_grid = np.stack(mesh, axis=-1)
        cartesian_grid = np.einsum("...i,ij->...j", fractional_grid, self.lattice_vectors)
        return fractional_grid, cartesian_grid

    def _normalize_density(self, density: np.ndarray | Sequence[float]) -> np.ndarray:
        density_array = np.array(density, dtype=float)
        expected_size = self.grid_point_count
        if density_array.shape == tuple(self.grid_shape):
            reshaped_density = density_array.copy()
        elif density_array.size == expected_size:
            reshaped_density = density_array.reshape(self.grid_shape).copy()
        else:
            raise ValueError(
                f"电子密度应为 shape={self.grid_shape} 或长度 {expected_size}，当前为 {density_array.shape}。"
            )

        clipped_density = np.clip(reshaped_density, 0.0, None)
        if clipped_density.size == 0 or self.total_electrons <= 0.0:
            return clipped_density

        total_density = float(clipped_density.sum() * self.volume_element)
        if total_density <= 0.0:
            return np.full(self.grid_shape, self.total_electrons / self.cell_volume, dtype=float)

        return clipped_density * (self.total_electrons / total_density)

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

    def _resample_periodic_grid(self, grid: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
        target_fractional_grid, _ = self._build_real_space_grid(target_shape)
        sampled_values = np.empty(tuple(target_shape), dtype=float)
        for index in np.ndindex(*target_shape):
            sampled_values[index] = self._sample_periodic_grid(grid, target_fractional_grid[index])
        return sampled_values

    def _build_initial_noise_grid(self) -> np.ndarray:
        if self.initial_density_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(self.initial_density_seed))

        raw_noise = rng.random(self.initial_density_noise_grid)
        smoothed_noise = raw_noise.copy()
        for axis in range(3):
            smoothed_noise += np.roll(raw_noise, 1, axis=axis)
            smoothed_noise += np.roll(raw_noise, -1, axis=axis)
        smoothed_noise /= 7.0
        if smoothed_noise.shape != tuple(self.grid_shape):
            smoothed_noise = self._resample_periodic_grid(smoothed_noise, self.grid_shape)
        return smoothed_noise

    def _minimum_image_delta(self, fractional_position: Sequence[float]) -> np.ndarray:
        delta_fractional = self.fractional_grid - np.array(fractional_position, dtype=float)
        delta_fractional -= np.round(delta_fractional)
        delta_cartesian = np.einsum("...i,ij->...j", delta_fractional, self.lattice_vectors)
        return delta_cartesian

    def _build_ionic_potential(self) -> np.ndarray:
        ionic_potential = np.zeros(self.grid_shape, dtype=float)
        for site_index, fractional_position in enumerate(self.fractional_positions):
            delta_cartesian = self._minimum_image_delta(fractional_position)
            distance = np.sqrt(np.sum(delta_cartesian * delta_cartesian, axis=-1))
            local_pseudopotential = self.pseudopotentials[self.elements[site_index]]
            ionic_potential += self.ionic_potential_scale * local_pseudopotential.evaluate_local(distance)
        return ionic_potential

    def _build_nonlocal_projectors(self) -> tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]:
        projector_columns: list[np.ndarray] = []
        projector_metadata: list[Dict[str, Any]] = []
        channel_index_map: dict[tuple[int, int, int], int] = {}

        for atom_index, fractional_position in enumerate(self.fractional_positions):
            pseudopotential = self.pseudopotentials[self.elements[atom_index]]
            delta_cartesian = self._minimum_image_delta(fractional_position)
            distance = np.sqrt(np.sum(delta_cartesian * delta_cartesian, axis=-1))
            theta = np.arccos(np.clip(delta_cartesian[..., 2] / np.maximum(distance, 1.0e-12), -1.0, 1.0))
            phi = np.mod(np.arctan2(delta_cartesian[..., 1], delta_cartesian[..., 0]), 2.0 * np.pi)

            for radial_index, projector in enumerate(pseudopotential.projectors):
                radial_part = pseudopotential.evaluate_projector_radial(radial_index, distance)
                for magnetic_number in range(-projector.angular_momentum, projector.angular_momentum + 1):
                    projector_values = radial_part * spherical_harmonic(
                        projector.angular_momentum,
                        magnetic_number,
                        theta,
                        phi,
                    )
                    channel_index = len(projector_columns)
                    channel_index_map[(atom_index, radial_index, magnetic_number)] = channel_index
                    projector_columns.append(projector_values.reshape(-1).astype(complex))
                    projector_metadata.append(
                        {
                            "atom_index": atom_index,
                            "element": self.elements[atom_index],
                            "beta_index": projector.beta_index,
                            "radial_index": radial_index,
                            "angular_momentum": projector.angular_momentum,
                            "magnetic_number": magnetic_number,
                        }
                    )

        projector_count = len(projector_columns)
        if projector_count == 0:
            return (
                np.zeros((self.grid_point_count, 0), dtype=complex),
                np.zeros((0, 0), dtype=complex),
                projector_metadata,
            )

        expanded_d_matrix = np.zeros((projector_count, projector_count), dtype=complex)
        for atom_index, _ in enumerate(self.fractional_positions):
            pseudopotential = self.pseudopotentials[self.elements[atom_index]]
            radial_d_matrix = np.array(pseudopotential.dij, dtype=float)
            projector_l_values = [projector.angular_momentum for projector in pseudopotential.projectors]
            for left_radial_index, left_l in enumerate(projector_l_values):
                for right_radial_index, right_l in enumerate(projector_l_values):
                    coupling = radial_d_matrix[left_radial_index, right_radial_index]
                    if abs(coupling) <= 0.0 or left_l != right_l:
                        continue
                    for magnetic_number in range(-left_l, left_l + 1):
                        left_index = channel_index_map[(atom_index, left_radial_index, magnetic_number)]
                        right_index = channel_index_map[(atom_index, right_radial_index, magnetic_number)]
                        expanded_d_matrix[left_index, right_index] = coupling

        return np.column_stack(projector_columns), expanded_d_matrix, projector_metadata

    def _build_poisson_kernel(self) -> np.ndarray:
        reciprocal_axes = [
            2.0 * np.pi * np.fft.fftfreq(self.grid_shape[axis], d=self.grid_spacing[axis])
            for axis in range(3)
        ]
        gx, gy, gz = np.meshgrid(*reciprocal_axes, indexing="ij")
        g_squared = gx * gx + gy * gy + gz * gz
        poisson_kernel = np.zeros(self.grid_shape, dtype=float)
        mask = g_squared > 0.0
        poisson_kernel[mask] = 4.0 * np.pi / g_squared[mask]
        return poisson_kernel

    def _solve_poisson(self, density: np.ndarray) -> np.ndarray:
        density_fft = np.fft.fftn(density)
        hartree_fft = self._poisson_kernel * density_fft
        hartree_fft[(0, 0, 0)] = 0.0
        return np.real(np.fft.ifftn(hartree_fft))

    def _build_ewald_parameters(self) -> Dict[str, float]:
        min_lattice_length = min(float(np.linalg.norm(vector)) for vector in self.lattice_vectors)
        tolerance = float(self.config.get("ewald_tolerance", 1.0e-8))
        alpha = float(self.config.get("ewald_alpha", 5.0 / max(min_lattice_length, 1.0e-8)))
        sqrt_log = np.sqrt(-np.log(tolerance))
        real_cutoff = float(self.config.get("ewald_real_cutoff", sqrt_log / alpha))
        reciprocal_cutoff = float(self.config.get("ewald_reciprocal_cutoff", 2.0 * alpha * sqrt_log))
        return {
            "alpha": alpha,
            "real_cutoff": real_cutoff,
            "reciprocal_cutoff": reciprocal_cutoff,
            "tolerance": tolerance,
        }

    def _generate_translation_vectors(self, basis_vectors: np.ndarray, cutoff: float) -> list[np.ndarray]:
        if cutoff <= 0.0:
            return [np.zeros(3, dtype=float)]

        extents = [
            max(1, int(np.ceil(cutoff / max(np.linalg.norm(basis_vectors[axis]), 1.0e-12))) + 1)
            for axis in range(3)
        ]
        vectors: list[np.ndarray] = []
        for i in range(-extents[0], extents[0] + 1):
            for j in range(-extents[1], extents[1] + 1):
                for k in range(-extents[2], extents[2] + 1):
                    vector = i * basis_vectors[0] + j * basis_vectors[1] + k * basis_vectors[2]
                    if np.linalg.norm(vector) <= cutoff + 1.0e-12:
                        vectors.append(vector)
        return vectors

    def _compute_ion_ion_energy(self) -> float:
        alpha = self.ewald_parameters["alpha"]
        real_cutoff = self.ewald_parameters["real_cutoff"]
        reciprocal_cutoff = self.ewald_parameters["reciprocal_cutoff"]
        total_charge = float(np.sum(self.ionic_charges))

        real_translations = self._generate_translation_vectors(self.lattice_vectors, real_cutoff)
        real_energy = 0.0
        for atom_i in range(self.site_count):
            for atom_j in range(self.site_count):
                for translation in real_translations:
                    delta = self.cartesian_positions[atom_i] - self.cartesian_positions[atom_j] + translation
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1.0e-12:
                        continue
                    real_energy += (
                        self.ionic_charges[atom_i]
                        * self.ionic_charges[atom_j]
                        * float(special.erfc(alpha * distance))
                        / distance
                    )
        real_energy *= 0.5

        reciprocal_translations = self._generate_translation_vectors(self.reciprocal_lattice_vectors, reciprocal_cutoff)
        reciprocal_energy = 0.0
        for reciprocal_vector in reciprocal_translations:
            g_squared = float(np.dot(reciprocal_vector, reciprocal_vector))
            if g_squared <= 1.0e-12:
                continue
            structure_factor = np.sum(
                self.ionic_charges
                * np.exp(-1.0j * (self.cartesian_positions @ reciprocal_vector))
            )
            reciprocal_energy += (
                np.exp(-g_squared / (4.0 * alpha * alpha))
                * abs(structure_factor) ** 2
                / g_squared
            )
        reciprocal_energy *= 2.0 * np.pi / self.cell_volume

        self_energy = -alpha / np.sqrt(np.pi) * float(np.sum(self.ionic_charges * self.ionic_charges))
        background_energy = -np.pi * total_charge * total_charge / (2.0 * alpha * alpha * self.cell_volume)
        return float(real_energy + reciprocal_energy + self_energy + background_energy)

    def initial_density(self) -> np.ndarray:
        configured_density = self.initial_density_values
        if isinstance(configured_density, Sequence) and not isinstance(configured_density, (str, bytes)):
            raw_density = np.array(configured_density, dtype=float)
            return self._normalize_density(raw_density)

        if self._initial_density_cache is None:
            initial_noise = self._build_initial_noise_grid() + 1.0e-8
            self._initial_density_cache = self._normalize_density(initial_noise)
        return self._initial_density_cache.copy()

    def iterative_terms(self, density: np.ndarray | Sequence[float] | None = None) -> Dict[str, np.ndarray]:
        density_array = self._normalize_density(self.initial_density() if density is None else density)
        hartree_potential = self.hartree_scale * self._solve_poisson(density_array)
        xc_terms = evaluate_exchange_correlation_potential(
            density_array,
            self.grid_spacing,
            self.xc_functional,
        )
        exchange_energy_density = self.xc_scale * xc_terms["exchange_energy_density"]
        correlation_energy_density = self.xc_scale * xc_terms["correlation_energy_density"]
        exchange_potential = self.xc_scale * xc_terms["exchange"]
        correlation_potential = self.xc_scale * xc_terms["correlation"]
        exchange_correlation_potential = exchange_potential + correlation_potential
        effective_potential = self.ionic_potential + hartree_potential + exchange_correlation_potential

        return {
            "density": density_array,
            "ionic_potential": self.ionic_potential.copy(),
            "hartree_potential": hartree_potential,
            "exchange_energy_density": exchange_energy_density,
            "correlation_energy_density": correlation_energy_density,
            "exchange_potential": exchange_potential,
            "correlation_potential": correlation_potential,
            "exchange_correlation_potential": exchange_correlation_potential,
            "effective_potential": effective_potential,
        }

    def build(
        self,
        density: np.ndarray | Sequence[float] | None = None,
        k_point: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        iterative_terms = self.iterative_terms(density)
        selected_k_point = [float(value) for value in (self.k_point if k_point is None else k_point)]
        return {
            "model": "real-space Kohn-Sham prototype with local and nonlocal SG15 pseudopotential",
            "boundary_conditions": "periodic",
            "basis": "real-space grid",
            "xc_functional": self.xc_functional,
            "k_grid": list(self.k_grid),
            "k_points": self.k_points,
            "k_weights": self.k_weights,
            "k_point": selected_k_point,
            "lattice_vectors": self.lattice_vectors.tolist(),
            "reciprocal_lattice_vectors": self.reciprocal_lattice_vectors.tolist(),
            "elements": self.elements,
            "ionic_charges": self.ionic_charges.tolist(),
            "fractional_positions": self.fractional_positions,
            "cartesian_positions": self.cartesian_positions.tolist(),
            "grid_shape": list(self.grid_shape),
            "grid_spacing": self.grid_spacing.tolist(),
            "volume": self.cell_volume,
            "volume_element": self.volume_element,
            "fractional_grid": self.fractional_grid,
            "cartesian_grid": self.cartesian_grid,
            "density": iterative_terms["density"],
            "pseudopotentials": {
                element: {
                    "file_path": pseudopotential.file_path,
                    "functional": pseudopotential.functional,
                    "z_valence": pseudopotential.z_valence,
                    "projector_count": len(pseudopotential.projectors),
                }
                for element, pseudopotential in self.pseudopotentials.items()
            },
            "pseudopotential_warnings": self.pseudopotential_warnings,
            "ewald": {
                **self.ewald_parameters,
                "ion_ion_energy": self.ion_ion_energy,
            },
            "nonlocal_projector_matrix": self.nonlocal_projector_matrix,
            "nonlocal_d_matrix": self.nonlocal_d_matrix,
            "nonlocal_projector_metadata": self.nonlocal_projector_metadata,
            "potentials": {
                "ionic": iterative_terms["ionic_potential"],
                "hartree": iterative_terms["hartree_potential"],
                "exchange": iterative_terms["exchange_potential"],
                "correlation": iterative_terms["correlation_potential"],
                "exchange_correlation": iterative_terms["exchange_correlation_potential"],
                "effective": iterative_terms["effective_potential"],
            },
            "energy_densities": {
                "exchange": iterative_terms["exchange_energy_density"],
                "correlation": iterative_terms["correlation_energy_density"],
            },
        }


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
    result = hamiltonian.build(hamiltonian.initial_density())
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())