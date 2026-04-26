from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from hamiltonian import Hamiltonian, read_input_json
from wavefunction import WaveFunction


def normalize_density(density: np.ndarray, total_electrons: float, volume_element: float) -> np.ndarray:
    clipped_density = np.clip(density, 0.0, None)
    if clipped_density.size == 0 or total_electrons <= 0.0:
        return clipped_density

    density_sum = float(clipped_density.sum() * volume_element)
    if density_sum <= 0.0:
        return np.full_like(clipped_density, total_electrons / (clipped_density.size * volume_element))

    return clipped_density * (total_electrons / density_sum)


def _update_positions_in_config(config: Dict[str, Any], fractional_positions: np.ndarray) -> None:
    updated_positions = []
    for entry, position in zip(config["pos"], fractional_positions.tolist()):
        if isinstance(entry, dict):
            updated_entry = dict(entry)
            updated_entry["position"] = position
            updated_positions.append(updated_entry)
            continue

        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            if len(entry) == 4 and isinstance(entry[0], str):
                updated_positions.append([entry[0], float(position[0]), float(position[1]), float(position[2])])
                continue
            if len(entry) >= 3 and not isinstance(entry[0], str):
                updated_positions.append([float(position[0]), float(position[1]), float(position[2]), *entry[3:]])
                continue

        raise ValueError("无法更新 pos 原子坐标，输入格式不受支持。")

    config["pos"] = updated_positions


def _build_displaced_config(
    base_config: Dict[str, Any],
    base_fractional_positions: np.ndarray,
    lattice_vectors: np.ndarray,
    atom_index: int,
    axis: int,
    displacement_cartesian: float,
) -> Dict[str, Any]:
    displaced_config = copy.deepcopy(base_config)
    displaced_positions = np.array(base_fractional_positions, dtype=float)
    cartesian_shift = np.zeros(3, dtype=float)
    cartesian_shift[axis] = displacement_cartesian
    fractional_shift = cartesian_shift @ np.linalg.inv(lattice_vectors)
    displaced_positions[atom_index] = displaced_positions[atom_index] + fractional_shift
    _update_positions_in_config(displaced_config, displaced_positions)
    return displaced_config


def _build_strained_config(
    base_config: Dict[str, Any],
    base_lattice_vectors: np.ndarray,
    strain_tensor: np.ndarray,
) -> Dict[str, Any]:
    strained_config = copy.deepcopy(base_config)
    deformation_gradient = np.eye(3) + strain_tensor
    strained_lattice_vectors = base_lattice_vectors @ deformation_gradient.T
    strained_config["lattice_vectors"] = strained_lattice_vectors.tolist()
    return strained_config


def _prepare_property_config(
    base_config: Dict[str, Any],
    initial_density: np.ndarray,
) -> Dict[str, Any]:
    property_config = copy.deepcopy(base_config)
    property_config["initial_density"] = np.array(initial_density, dtype=float).tolist()
    property_config["show_progress"] = False
    property_config["show_plot"] = False
    property_config["show_density_evolution"] = False
    property_config["compute_forces"] = False
    property_config["compute_stress"] = False
    if "property_scf_max_iterations" in property_config:
        property_config["scf_max_iterations"] = int(property_config["property_scf_max_iterations"])
    if "property_scf_tolerance" in property_config:
        property_config["scf_tolerance"] = float(property_config["property_scf_tolerance"])
    return property_config


def _property_progress(label: str, current: int, total: int, show_progress: bool) -> None:
    if not show_progress:
        return
    sys.stderr.write(f"\r{label} {current}/{total}")
    sys.stderr.flush()


def _finish_property_progress(show_progress: bool) -> None:
    if not show_progress:
        return
    sys.stderr.write("\n")
    sys.stderr.flush()


def _solve_k_mesh(
    hamiltonian: Hamiltonian,
    wavefunction: WaveFunction,
    current_density: np.ndarray,
) -> Dict[str, Any]:
    k_hamiltonians: list[Dict[str, Any]] = []
    k_solvers: list[Dict[str, Any]] = []
    for k_point in hamiltonian.k_points:
        k_hamiltonian = hamiltonian.build(current_density, k_point=k_point)
        k_solver = wavefunction.solve(k_hamiltonian)
        k_hamiltonians.append(k_hamiltonian)
        k_solvers.append(k_solver)

    k_occupations = wavefunction.assign_kpoint_occupations(k_solvers, hamiltonian.k_weights)
    total_density = np.zeros_like(current_density)
    energy_expectations: list[Dict[str, float]] = []
    for k_hamiltonian, k_solver, occupations, weight in zip(
        k_hamiltonians,
        k_solvers,
        k_occupations,
        hamiltonian.k_weights,
    ):
        wavefunction.attach_occupations(k_solver, occupations)
        k_density = wavefunction.solve_density(
            k_hamiltonian,
            k_solver,
            occupations=occupations,
            k_weight=weight,
        )
        total_density += np.array(k_density["grid_density"], dtype=float)
        energy_expectations.append(
            wavefunction.compute_energy_expectations(
                k_hamiltonian,
                k_solver,
                occupations=occupations,
                k_weight=weight,
            )
        )

    normalized_density = normalize_density(total_density, hamiltonian.total_electrons, hamiltonian.volume_element)
    reference_hamiltonian = k_hamiltonians[0]
    density_result = wavefunction.density_from_grid_density(reference_hamiltonian, normalized_density)
    return {
        "density": normalized_density,
        "density_result": density_result,
        "k_hamiltonians": k_hamiltonians,
        "k_solvers": k_solvers,
        "k_occupations": k_occupations,
        "energy_expectations": energy_expectations,
    }


def _build_energy_summary(
    reference_hamiltonian: Dict[str, Any],
    density: np.ndarray,
    energy_expectations: Sequence[Dict[str, float]],
) -> Dict[str, Any]:
    density = np.array(density, dtype=float)
    potentials = reference_hamiltonian["potentials"]
    energy_densities = reference_hamiltonian["energy_densities"]
    volume_element = float(reference_hamiltonian["volume_element"])

    kinetic_energy = float(sum(item["kinetic"] for item in energy_expectations))
    nonlocal_energy = float(sum(item["nonlocal"] for item in energy_expectations))
    band_energy = float(sum(item["band"] for item in energy_expectations))
    local_ionic_energy = float(np.sum(density * np.array(potentials["ionic"], dtype=float)) * volume_element)
    hartree_potential_energy = float(np.sum(density * np.array(potentials["hartree"], dtype=float)) * volume_element)
    exchange_potential_energy = float(np.sum(density * np.array(potentials["exchange"], dtype=float)) * volume_element)
    correlation_potential_energy = float(np.sum(density * np.array(potentials["correlation"], dtype=float)) * volume_element)
    exchange_energy = float(np.sum(density * np.array(energy_densities["exchange"], dtype=float)) * volume_element)
    correlation_energy = float(np.sum(density * np.array(energy_densities["correlation"], dtype=float)) * volume_element)
    hartree_energy = 0.5 * hartree_potential_energy
    xc_energy = exchange_energy + correlation_energy
    xc_potential_energy = exchange_potential_energy + correlation_potential_energy
    ion_ion_energy = float(reference_hamiltonian["ewald"]["ion_ion_energy"])

    total_energy = (
        kinetic_energy
        + local_ionic_energy
        + nonlocal_energy
        + hartree_energy
        + xc_energy
        + ion_ion_energy
    )
    total_from_band = band_energy - hartree_energy - xc_potential_energy + xc_energy + ion_ion_energy
    band_reconstruction = (
        kinetic_energy
        + local_ionic_energy
        + nonlocal_energy
        + hartree_potential_energy
        + xc_potential_energy
    )

    return {
        "band_energy": band_energy,
        "kinetic_energy": kinetic_energy,
        "local_ionic_energy": local_ionic_energy,
        "nonlocal_energy": nonlocal_energy,
        "hartree_energy": hartree_energy,
        "hartree_potential_energy": hartree_potential_energy,
        "exchange_energy": exchange_energy,
        "correlation_energy": correlation_energy,
        "xc_energy": xc_energy,
        "exchange_potential_energy": exchange_potential_energy,
        "correlation_potential_energy": correlation_potential_energy,
        "xc_potential_energy": xc_potential_energy,
        "ion_ion_energy": ion_ion_energy,
        "total_energy": total_energy,
        "total_from_band": total_from_band,
        "band_reconstruction_error": float(band_energy - band_reconstruction),
        "total_energy_consistency_error": float(total_energy - total_from_band),
    }


def _run_scf_cycle(config: Dict[str, Any]) -> Dict[str, Any]:
    hamiltonian = Hamiltonian(config)
    wavefunction = WaveFunction(config, total_electrons=hamiltonian.total_electrons)

    max_iterations = int(config.get("scf_max_iterations", 60))
    tolerance = float(config.get("scf_tolerance", 1.0e-6))
    mixing_beta = float(config.get("mixing_beta", 0.35))
    momentum_beta = float(config.get("momentum_beta", 0.80))
    show_progress = bool(config.get("show_progress", True))

    density = hamiltonian.initial_density()
    velocity = np.zeros_like(density)
    converged = False
    history: list[Dict[str, Any]] = []
    density_frames: list[Dict[str, Any]] = []

    initial_hamiltonian_data = hamiltonian.build(density)
    initial_density_result = wavefunction.density_from_grid_density(initial_hamiltonian_data, density)
    density_frames.append(
        {
            "iteration": 0,
            "residual": None,
            "band_energy": None,
            "total_energy": None,
            "site_density": initial_density_result["site_density"],
            "grid_density": np.array(initial_density_result["grid_density"], dtype=np.float32),
        }
    )

    for iteration in range(1, max_iterations + 1):
        k_mesh_result = _solve_k_mesh(hamiltonian, wavefunction, density)
        reference_hamiltonian = k_mesh_result["k_hamiltonians"][0]
        energy_summary = _build_energy_summary(
            reference_hamiltonian,
            k_mesh_result["density"],
            k_mesh_result["energy_expectations"],
        )

        target_density = np.array(k_mesh_result["density"], dtype=float)
        correction = target_density - density
        velocity = momentum_beta * velocity + mixing_beta * correction
        mixed_density = normalize_density(
            density + velocity,
            hamiltonian.total_electrons,
            hamiltonian.volume_element,
        )
        mixed_density_result = wavefunction.density_from_grid_density(initial_hamiltonian_data, mixed_density)
        residual = float(np.max(np.abs(mixed_density - density)))

        history.append(
            {
                "iteration": iteration,
                "residual": residual,
                "band_energy": energy_summary["band_energy"],
                "total_energy": energy_summary["total_energy"],
                "density_min": float(np.min(mixed_density)),
                "density_max": float(np.max(mixed_density)),
            }
        )
        density_frames.append(
            {
                "iteration": iteration,
                "residual": residual,
                "band_energy": energy_summary["band_energy"],
                "total_energy": energy_summary["total_energy"],
                "site_density": mixed_density_result["site_density"],
                "grid_density": np.array(mixed_density_result["grid_density"], dtype=np.float32),
            }
        )

        density = mixed_density
        if show_progress:
            update_scf_progress(
                iteration=iteration,
                max_iterations=max_iterations,
                residual=residual,
                tolerance=tolerance,
                converged=residual < tolerance,
            )

        if residual < tolerance:
            converged = True
            break

    if show_progress:
        finish_scf_progress()

    final_k_mesh = _solve_k_mesh(hamiltonian, wavefunction, density)
    final_hamiltonian = final_k_mesh["k_hamiltonians"][0]
    final_density = final_k_mesh["density_result"]
    final_energies = _build_energy_summary(
        final_hamiltonian,
        final_k_mesh["density"],
        final_k_mesh["energy_expectations"],
    )
    final_solver = {
        "basis": "real-space grid",
        "k_grid": list(hamiltonian.k_grid),
        "k_points": hamiltonian.k_points,
        "k_weights": hamiltonian.k_weights,
        "eigenvalues_by_k": [solver_result["eigenvalues"] for solver_result in final_k_mesh["k_solvers"]],
        "occupations_by_k": [solver_result["occupations"] for solver_result in final_k_mesh["k_solvers"]],
        "band_energy": float(final_energies["band_energy"]),
    }
    return {
        "hamiltonian_object": hamiltonian,
        "wavefunction_object": wavefunction,
        "electron_count": hamiltonian.total_electrons,
        "k_grid": list(hamiltonian.k_grid),
        "k_points": hamiltonian.k_points,
        "k_weights": hamiltonian.k_weights,
        "converged": converged,
        "iterations": len(history),
        "history": history,
        "density_frames": density_frames,
        "final_density": final_density,
        "final_hamiltonian": final_hamiltonian,
        "final_solver": final_solver,
        "final_energies": final_energies,
        "final_k_mesh": final_k_mesh,
    }


def _compute_numerical_forces(
    base_config: Dict[str, Any],
    base_result: Dict[str, Any],
) -> Dict[str, Any]:
    show_progress = bool(base_config.get("show_progress", True))
    displacement = float(base_config.get("force_displacement", 1.0e-3))
    base_density = np.array(base_result["final_density"]["grid_density"], dtype=float)
    base_fractional_positions = np.array(base_result["final_hamiltonian"]["fractional_positions"], dtype=float)
    lattice_vectors = np.array(base_result["final_hamiltonian"]["lattice_vectors"], dtype=float)
    atom_count = base_fractional_positions.shape[0]

    forces = np.zeros((atom_count, 3), dtype=float)
    evaluations: list[Dict[str, Any]] = []
    total_evaluations = atom_count * 3
    evaluation_index = 0

    for atom_index in range(atom_count):
        for axis in range(3):
            evaluation_index += 1
            _property_progress("Forces", evaluation_index, total_evaluations, show_progress)

            plus_config = _prepare_property_config(base_config, base_density)
            plus_config = _build_displaced_config(
                plus_config,
                base_fractional_positions,
                lattice_vectors,
                atom_index,
                axis,
                displacement,
            )
            plus_result = _run_scf_cycle(plus_config)

            minus_config = _prepare_property_config(base_config, base_density)
            minus_config = _build_displaced_config(
                minus_config,
                base_fractional_positions,
                lattice_vectors,
                atom_index,
                axis,
                -displacement,
            )
            minus_result = _run_scf_cycle(minus_config)

            energy_plus = float(plus_result["final_energies"]["total_energy"])
            energy_minus = float(minus_result["final_energies"]["total_energy"])
            force_value = -(energy_plus - energy_minus) / (2.0 * displacement)
            forces[atom_index, axis] = force_value
            evaluations.append(
                {
                    "atom_index": atom_index,
                    "axis": axis,
                    "displacement": displacement,
                    "energy_plus": energy_plus,
                    "energy_minus": energy_minus,
                    "force": force_value,
                }
            )

    _finish_property_progress(show_progress)
    return {
        "method": "numerical central difference on fully relaxed total energy",
        "displacement": displacement,
        "cartesian_forces": forces,
        "net_force": np.sum(forces, axis=0),
        "evaluations": evaluations,
    }


def _strain_tensor_from_component(component: int, strain_step: float) -> np.ndarray:
    strain_tensor = np.zeros((3, 3), dtype=float)
    if component < 3:
        strain_tensor[component, component] = strain_step
        return strain_tensor

    if component == 3:
        strain_tensor[1, 2] = 0.5 * strain_step
        strain_tensor[2, 1] = 0.5 * strain_step
    elif component == 4:
        strain_tensor[0, 2] = 0.5 * strain_step
        strain_tensor[2, 0] = 0.5 * strain_step
    else:
        strain_tensor[0, 1] = 0.5 * strain_step
        strain_tensor[1, 0] = 0.5 * strain_step
    return strain_tensor


def _compute_numerical_stress(
    base_config: Dict[str, Any],
    base_result: Dict[str, Any],
) -> Dict[str, Any]:
    show_progress = bool(base_config.get("show_progress", True))
    strain_step = float(base_config.get("stress_strain_step", 2.5e-3))
    base_density = np.array(base_result["final_density"]["grid_density"], dtype=float)
    base_lattice_vectors = np.array(base_result["final_hamiltonian"]["lattice_vectors"], dtype=float)
    base_volume = float(base_result["final_hamiltonian"]["volume"])

    stress_tensor = np.zeros((3, 3), dtype=float)
    evaluations: list[Dict[str, Any]] = []
    for component in range(6):
        _property_progress("Stress", component + 1, 6, show_progress)
        strain_plus = _strain_tensor_from_component(component, strain_step)
        strain_minus = _strain_tensor_from_component(component, -strain_step)

        plus_config = _prepare_property_config(base_config, base_density)
        plus_config = _build_strained_config(plus_config, base_lattice_vectors, strain_plus)
        plus_result = _run_scf_cycle(plus_config)

        minus_config = _prepare_property_config(base_config, base_density)
        minus_config = _build_strained_config(minus_config, base_lattice_vectors, strain_minus)
        minus_result = _run_scf_cycle(minus_config)

        energy_plus = float(plus_result["final_energies"]["total_energy"])
        energy_minus = float(minus_result["final_energies"]["total_energy"])
        stress_component = (energy_plus - energy_minus) / (2.0 * strain_step * base_volume)
        if component < 3:
            stress_tensor[component, component] = stress_component
        elif component == 3:
            stress_tensor[1, 2] = stress_component
            stress_tensor[2, 1] = stress_component
        elif component == 4:
            stress_tensor[0, 2] = stress_component
            stress_tensor[2, 0] = stress_component
        else:
            stress_tensor[0, 1] = stress_component
            stress_tensor[1, 0] = stress_component

        evaluations.append(
            {
                "component": component,
                "strain_step": strain_step,
                "energy_plus": energy_plus,
                "energy_minus": energy_minus,
                "stress_component": stress_component,
            }
        )

    _finish_property_progress(show_progress)
    return {
        "method": "numerical central difference with homogeneous strain",
        "strain_step": strain_step,
        "stress_tensor": stress_tensor,
        "symmetry_error": float(np.max(np.abs(stress_tensor - stress_tensor.T))),
        "evaluations": evaluations,
    }


def _build_consistency_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    final_density = np.array(result["final_density"]["grid_density"], dtype=float)
    volume_element = float(result["final_hamiltonian"]["volume_element"])
    integrated_density = float(np.sum(final_density) * volume_element)
    checks = {
        "integrated_density": integrated_density,
        "electron_count_error": float(integrated_density - result["electron_count"]),
        "band_reconstruction_error": float(result["final_energies"]["band_reconstruction_error"]),
        "total_energy_consistency_error": float(result["final_energies"]["total_energy_consistency_error"]),
    }
    if result.get("forces") is not None:
        checks["net_force"] = np.array(result["forces"]["net_force"], dtype=float)
    if result.get("stress") is not None:
        checks["stress_symmetry_error"] = float(result["stress"]["symmetry_error"])
    return checks


def solve_scf(config: Dict[str, Any]) -> Dict[str, Any]:
    result = _run_scf_cycle(config)

    compute_forces = bool(config.get("compute_forces", False))
    compute_stress = bool(config.get("compute_stress", False))
    result["forces"] = _compute_numerical_forces(config, result) if compute_forces else None
    result["stress"] = _compute_numerical_stress(config, result) if compute_stress else None
    result["consistency_checks"] = _build_consistency_checks(result)
    return result


def select_density_points(
    density_grid: np.ndarray,
    cartesian_grid: np.ndarray,
    density_quantile: float,
    max_points: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_density = density_grid.reshape(-1)
    flat_points = cartesian_grid.reshape(-1, 3)
    threshold = float(np.quantile(flat_density, density_quantile)) if flat_density.size else 0.0
    mask = flat_density >= threshold
    selected_points = flat_points[mask]
    selected_density = flat_density[mask]

    if selected_points.size == 0:
        selected_points = flat_points
        selected_density = flat_density

    if selected_density.shape[0] > max_points:
        if np.allclose(selected_density, selected_density[0]):
            sampled_indices = np.linspace(0, selected_density.shape[0] - 1, max_points, dtype=int)
        else:
            sampled_indices = np.argsort(selected_density)[-max_points:]
            sampled_indices = np.sort(sampled_indices)
        selected_points = selected_points[sampled_indices]
        selected_density = selected_density[sampled_indices]

    density_span = float(selected_density.max() - selected_density.min()) if selected_density.size else 0.0
    if density_span <= 0.0:
        point_sizes = np.full(selected_density.shape, 30.0)
    else:
        point_sizes = 20.0 + 100.0 * (selected_density - selected_density.min()) / density_span

    return selected_points, selected_density, point_sizes


def prepare_density_frames(
    density_frames: Sequence[Dict[str, Any]],
    cartesian_grid: np.ndarray,
    density_quantile: float,
) -> tuple[list[Dict[str, Any]], float]:
    prepared_frames: list[Dict[str, Any]] = []
    color_max = 0.0

    for frame in density_frames:
        selected_points, selected_density, point_sizes = select_density_points(
            np.array(frame["grid_density"], dtype=float),
            cartesian_grid,
            density_quantile,
        )
        if selected_density.size:
            color_max = max(color_max, float(selected_density.max()))

        prepared_frames.append(
            {
                "iteration": int(frame["iteration"]),
                "residual": frame.get("residual"),
                "band_energy": frame.get("band_energy"),
                "total_energy": frame.get("total_energy"),
                "selected_points": selected_points,
                "selected_density": selected_density,
                "point_sizes": point_sizes,
            }
        )

    return prepared_frames, color_max if color_max > 0.0 else 1.0


def build_repeat_translations(
    lattice_vectors: np.ndarray,
    repeat_count: int,
) -> list[np.ndarray]:
    sanitized_repeat_count = max(int(repeat_count), 1)
    translations: list[np.ndarray] = []
    for i in range(sanitized_repeat_count):
        for j in range(sanitized_repeat_count):
            for k in range(sanitized_repeat_count):
                translations.append(i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2])
    return translations


def repeat_periodic_points(points: np.ndarray, translations: Sequence[np.ndarray]) -> np.ndarray:
    if points.size == 0:
        return points
    return np.vstack([points + translation for translation in translations])


def repeat_periodic_scalars(values: np.ndarray, repeat_factor: int) -> np.ndarray:
    if values.size == 0:
        return values
    return np.tile(values, max(int(repeat_factor), 1))


def render_density_frame(
    axis: Any,
    prepared_frame: Dict[str, Any],
    color_max: float,
    lattice_vectors: np.ndarray,
    cell_corners: np.ndarray,
    edges: Sequence[tuple[int, int]],
    cartesian_positions: np.ndarray,
    repeat_count: int,
) -> None:
    axis.cla()

    selected_points = prepared_frame["selected_points"]
    selected_density = prepared_frame["selected_density"]
    point_sizes = prepared_frame["point_sizes"]
    translations = build_repeat_translations(lattice_vectors, repeat_count)
    repeated_points = repeat_periodic_points(selected_points, translations)
    repeated_density = repeat_periodic_scalars(selected_density, len(translations))
    repeated_point_sizes = repeat_periodic_scalars(point_sizes, len(translations))
    repeated_positions = repeat_periodic_points(cartesian_positions, translations)
    repeated_cell_corners = np.vstack([cell_corners + translation for translation in translations])
    min_corner = repeated_cell_corners.min(axis=0)
    max_corner = repeated_cell_corners.max(axis=0)

    axis.scatter(
        repeated_points[:, 0],
        repeated_points[:, 1],
        repeated_points[:, 2],
        c=repeated_density,
        cmap="inferno",
        vmin=0.0,
        vmax=color_max,
        s=repeated_point_sizes,
        alpha=0.60,
        linewidths=0.0,
    )

    for translation in translations:
        translated_corners = cell_corners + translation
        for start_index, end_index in edges:
            segment = translated_corners[[start_index, end_index]]
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#0f4c81", linewidth=1.3)

    axis.scatter(
        repeated_positions[:, 0],
        repeated_positions[:, 1],
        repeated_positions[:, 2],
        color="black",
        s=80.0,
        depthshade=False,
        label="atomic sites",
    )

    axis.set_xlim(min_corner[0], max_corner[0])
    axis.set_ylim(min_corner[1], max_corner[1])
    axis.set_zlim(min_corner[2], max_corner[2])
    axis.set_box_aspect(np.maximum(max_corner - min_corner, 1.0e-6))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("SCF Electron Density Evolution", pad=18)


def render_iteration_info(
    info_axis: Any,
    frame_index: int,
    frame_count: int,
    prepared_frame: Dict[str, Any],
    electron_count: float | None,
    k_grid: Sequence[int] | None,
    repeat_count: int,
) -> None:
    info_axis.cla()
    info_axis.axis("off")

    residual = prepared_frame.get("residual")
    band_energy = prepared_frame.get("band_energy")
    total_energy = prepared_frame.get("total_energy")
    iteration = prepared_frame["iteration"]

    lines = [
        "SCF Frame Info",
        f"Frame: {frame_index + 1}/{frame_count}",
        f"SCF iteration: {iteration}",
    ]
    if electron_count is not None:
        lines.append(f"Electron count: {electron_count:.3f}")
    if k_grid is not None:
        lines.append("K grid: " + " x ".join(str(int(component)) for component in k_grid))
    lines.append("Repeat cells: " + " x ".join([str(int(repeat_count))] * 3))
    if residual is None:
        lines.append("Residual: initial guess")
    else:
        lines.append(f"Residual: {float(residual):.3e}")
    if band_energy is None:
        lines.append("Band energy: initial guess")
    else:
        lines.append(f"Band energy: {float(band_energy):.6f}")
    if total_energy is None:
        lines.append("Total energy: initial guess")
    else:
        lines.append(f"Total energy: {float(total_energy):.6f}")

    lines.extend(
        [
            "",
            "Visual Legend",
            "Black points: atomic sites",
            "Blue edges: unit cell",
            "Inferno colors: density magnitude",
        ]
    )

    info_axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=info_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f5f7fb", "edgecolor": "#d8dee8"},
    )


def render_iteration_progress(
    progress_axis: Any,
    frame_index: int,
    frame_count: int,
    prepared_frame: Dict[str, Any],
    repeat_count: int,
) -> None:
    progress_axis.cla()
    progress_axis.axis("off")

    residual = prepared_frame.get("residual")
    if residual is None:
        label = "Initial density"
    else:
        label = f"Iteration {prepared_frame['iteration']} | residual {float(residual):.3e}"

    progress_axis.text(
        0.0,
        0.88,
        "SCF Evolution",
        transform=progress_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#16324f",
    )
    progress_axis.text(
        1.0,
        0.88,
        label,
        transform=progress_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#16324f",
    )
    progress_axis.text(
        0.0,
        0.16,
        (
            f"Drag the sliders below to inspect SCF frames and periodic repeats. "
            f"Frame: {frame_index + 1}/{frame_count}, repeat: {repeat_count}x{repeat_count}x{repeat_count}"
        ),
        transform=progress_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#46607a",
    )


def initialize_frame_slider(
    slider_axis: Any,
    frame_count: int,
    initial_frame_index: int,
) -> Any:
    from matplotlib.widgets import Slider

    slider_axis.set_facecolor("#eef2f7")
    frame_slider = Slider(
        ax=slider_axis,
        label="Frame",
        valmin=1,
        valmax=frame_count,
        valinit=initial_frame_index + 1,
        valstep=1,
        color="#0f4c81",
        initcolor="none",
    )
    frame_slider.valtext.set_text(f"{initial_frame_index + 1}/{frame_count}")
    return frame_slider


def initialize_repeat_slider(
    slider_axis: Any,
    repeat_max: int,
    initial_repeat_count: int,
) -> Any:
    from matplotlib.widgets import Slider

    slider_axis.set_facecolor("#eef2f7")
    repeat_slider = Slider(
        ax=slider_axis,
        label="Repeat",
        valmin=1,
        valmax=max(int(repeat_max), 1),
        valinit=max(int(initial_repeat_count), 1),
        valstep=1,
        color="#3a6b8f",
        initcolor="none",
    )
    current_repeat = max(int(initial_repeat_count), 1)
    repeat_slider.valtext.set_text(f"{current_repeat}x{current_repeat}x{current_repeat}")
    return repeat_slider


def update_scf_progress(
    iteration: int,
    max_iterations: int,
    residual: float,
    tolerance: float,
    converged: bool,
) -> None:
    progress_width = 28
    ratio = 1.0 if max_iterations <= 0 else min(iteration / max_iterations, 1.0)
    filled_width = int(progress_width * ratio)
    bar = "#" * filled_width + "-" * (progress_width - filled_width)
    status = "converged" if converged else "running"
    sys.stderr.write(
        f"\rSCF [{bar}] {iteration}/{max_iterations} residual={residual:.3e} tol={tolerance:.1e} {status}"
    )
    sys.stderr.flush()


def finish_scf_progress() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()


def write_visualization(
    hamiltonian_data: Dict[str, Any],
    density_result: Dict[str, Any],
    output_path: Path,
    density_frames: Sequence[Dict[str, Any]] | None = None,
    electron_count: float | None = None,
    k_grid: Sequence[int] | None = None,
    show_plot: bool = True,
    show_density_evolution: bool = True,
    density_quantile: float = 0.82,
    save_figure: bool = True,
    repeat_count: int = 1,
    repeat_max: int = 1,
) -> None:
    import matplotlib

    if show_plot:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")

    from matplotlib import cm
    from matplotlib import colors
    from matplotlib import pyplot as plt

    cartesian_grid = np.array(density_result["cartesian_grid"], dtype=float)
    lattice_vectors = np.array(hamiltonian_data["lattice_vectors"], dtype=float)
    cartesian_positions = np.array(hamiltonian_data["cartesian_positions"], dtype=float)

    frames_to_show = list(density_frames) if density_frames else [
        {
            "iteration": 0,
            "residual": None,
            "band_energy": None,
            "total_energy": None,
            "grid_density": np.array(density_result["grid_density"], dtype=np.float32),
        }
    ]
    prepared_frames, color_max = prepare_density_frames(frames_to_show, cartesian_grid, density_quantile)

    initial_repeat_count = max(int(repeat_count), 1)
    repeat_slider_max = max(int(repeat_max), initial_repeat_count, 1)

    figure = plt.figure(figsize=(12.5, 9.4), dpi=180)
    grid_spec = figure.add_gridspec(
        2,
        2,
        height_ratios=[14, 3.0],
        width_ratios=[15, 5],
        hspace=0.16,
        wspace=0.12,
    )
    bottom_spec = grid_spec[1, :].subgridspec(3, 1, height_ratios=[0.7, 1.0, 1.0], hspace=0.08)
    axis = figure.add_subplot(grid_spec[0, 0], projection="3d")
    info_axis = figure.add_subplot(grid_spec[0, 1])
    progress_axis = figure.add_subplot(bottom_spec[0, 0])
    frame_slider_axis = figure.add_subplot(bottom_spec[1, 0])
    repeat_slider_axis = figure.add_subplot(bottom_spec[2, 0])

    cell_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            lattice_vectors[0],
            lattice_vectors[1],
            lattice_vectors[2],
            lattice_vectors[0] + lattice_vectors[1],
            lattice_vectors[0] + lattice_vectors[2],
            lattice_vectors[1] + lattice_vectors[2],
            lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    color_mapper = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=color_max), cmap="inferno")
    color_mapper.set_array([])
    figure.colorbar(color_mapper, ax=axis, fraction=0.032, pad=0.03, shrink=0.82, label="density")

    current_frame_index = len(prepared_frames) - 1
    current_repeat_count = initial_repeat_count

    def update_frame(frame_index: int, repeat_value: int) -> None:
        render_density_frame(
            axis,
            prepared_frames[frame_index],
            color_max,
            lattice_vectors,
            cell_corners,
            edges,
            cartesian_positions,
            repeat_value,
        )
        render_iteration_info(
            info_axis,
            frame_index,
            len(prepared_frames),
            prepared_frames[frame_index],
            electron_count,
            k_grid,
            repeat_value,
        )
        render_iteration_progress(
            progress_axis,
            frame_index,
            len(prepared_frames),
            prepared_frames[frame_index],
            repeat_value,
        )

    update_frame(current_frame_index, current_repeat_count)

    if show_density_evolution and len(prepared_frames) > 1:
        frame_slider = initialize_frame_slider(frame_slider_axis, len(prepared_frames), current_frame_index)

        def on_slider_change(value: float) -> None:
            nonlocal current_frame_index
            frame_index = max(0, min(len(prepared_frames) - 1, int(round(value)) - 1))
            current_frame_index = frame_index
            update_frame(current_frame_index, current_repeat_count)
            frame_slider.valtext.set_text(f"{frame_index + 1}/{len(prepared_frames)}")
            figure.canvas.draw_idle()

        frame_slider.on_changed(on_slider_change)
    else:
        frame_slider_axis.axis("off")
        frame_slider_axis.text(
            0.0,
            0.5,
            "Single frame available" if show_density_evolution else "Density evolution slider disabled",
            transform=frame_slider_axis.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            color="#46607a",
        )

    if repeat_slider_max > 1:
        repeat_slider = initialize_repeat_slider(repeat_slider_axis, repeat_slider_max, current_repeat_count)

        def on_repeat_slider_change(value: float) -> None:
            nonlocal current_repeat_count
            current_repeat_count = max(1, int(round(value)))
            update_frame(current_frame_index, current_repeat_count)
            repeat_slider.valtext.set_text(
                f"{current_repeat_count}x{current_repeat_count}x{current_repeat_count}"
            )
            figure.canvas.draw_idle()

        repeat_slider.on_changed(on_repeat_slider_change)
    else:
        repeat_slider_axis.axis("off")
        repeat_slider_axis.text(
            0.0,
            0.5,
            "Repeat slider disabled (repeat_max <= 1)",
            transform=repeat_slider_axis.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            color="#46607a",
        )

    figure.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.08)
    if save_figure:
        figure.savefig(output_path)

    if show_plot:
        plt.show(block=True)

    plt.close(figure)


def make_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    final_density = result["final_density"]
    final_hamiltonian = result["final_hamiltonian"]
    final_solver = result["final_solver"]

    serializable = {
        "electron_count": result["electron_count"],
        "k_grid": result["k_grid"],
        "k_points": result["k_points"],
        "k_weights": result["k_weights"],
        "converged": result["converged"],
        "iterations": result["iterations"],
        "history": result["history"],
        "energies": result["final_energies"],
        "consistency_checks": result["consistency_checks"],
        "xc_functional": final_hamiltonian["xc_functional"],
        "elements": final_hamiltonian["elements"],
        "ionic_charges": final_hamiltonian["ionic_charges"],
        "pseudopotentials": final_hamiltonian["pseudopotentials"],
        "pseudopotential_warnings": final_hamiltonian["pseudopotential_warnings"],
        "ewald": final_hamiltonian["ewald"],
        "site_density": final_density["site_density"],
        "grid_shape": final_density["grid_shape"],
        "eigenvalues_by_k": final_solver["eigenvalues_by_k"],
        "occupations_by_k": final_solver["occupations_by_k"],
        "band_energy": final_solver["band_energy"],
        "lattice_vectors": final_hamiltonian["lattice_vectors"],
        "reciprocal_lattice_vectors": final_hamiltonian["reciprocal_lattice_vectors"],
        "cartesian_positions": final_hamiltonian["cartesian_positions"],
        "fractional_positions": final_hamiltonian["fractional_positions"],
        "volume": float(final_hamiltonian["volume"]),
        "volume_element": float(final_hamiltonian["volume_element"]),
        "potentials": {
            "ionic": np.array(final_hamiltonian["potentials"]["ionic"], dtype=float).tolist(),
            "hartree": np.array(final_hamiltonian["potentials"]["hartree"], dtype=float).tolist(),
            "exchange": np.array(final_hamiltonian["potentials"]["exchange"], dtype=float).tolist(),
            "correlation": np.array(final_hamiltonian["potentials"]["correlation"], dtype=float).tolist(),
            "exchange_correlation": np.array(final_hamiltonian["potentials"]["exchange_correlation"], dtype=float).tolist(),
            "effective": np.array(final_hamiltonian["potentials"]["effective"], dtype=float).tolist(),
        },
        "energy_densities": {
            "exchange": np.array(final_hamiltonian["energy_densities"]["exchange"], dtype=float).tolist(),
            "correlation": np.array(final_hamiltonian["energy_densities"]["correlation"], dtype=float).tolist(),
        },
    }

    if result.get("forces") is not None:
        serializable["forces"] = {
            "method": result["forces"]["method"],
            "displacement": result["forces"]["displacement"],
            "cartesian_forces": np.array(result["forces"]["cartesian_forces"], dtype=float).tolist(),
            "net_force": np.array(result["forces"]["net_force"], dtype=float).tolist(),
        }
    if result.get("stress") is not None:
        serializable["stress"] = {
            "method": result["stress"]["method"],
            "strain_step": result["stress"]["strain_step"],
            "stress_tensor": np.array(result["stress"]["stress_tensor"], dtype=float).tolist(),
            "symmetry_error": float(result["stress"]["symmetry_error"]),
        }
    return serializable


def _resolve_project_name(config: Dict[str, Any], default_name: str) -> str:
    project_name = str(config.get("project_name", config.get("output_prefix", default_name))).strip()
    return project_name or default_name


def _save_density_npy(density_result: Dict[str, Any], output_path: Path) -> None:
    density_grid = np.array(density_result["grid_density"], dtype=float)
    np.save(output_path, density_grid)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    input_path = Path(arguments[0]) if arguments else Path("INPUT.json")
    config = read_input_json(input_path)
    project_name = _resolve_project_name(config, input_path.stem)
    show_plot = bool(config.get("show_plot", True))
    show_density_evolution = bool(config.get("show_density_evolution", True))
    save_result_json = bool(config.get("save_result_json", True))
    save_density_npy = bool(config.get("save_density_npy", False))
    save_density_plot = bool(config.get("save_density_plot", True))
    density_quantile = float(config.get("density_visualization_quantile", 0.82))
    visualization_repeat_count = int(config.get("visualization_repeat_count", 1))
    visualization_repeat_max = int(config.get("visualization_repeat_max", visualization_repeat_count))

    result = solve_scf(config)
    output_dir = input_path.parent
    result_path = output_dir / f"{project_name}_result.json"
    plot_path = output_dir / f"{project_name}_density_3d.png"
    density_npy_path = output_dir / f"{project_name}_density.npy"

    if save_density_plot or show_plot:
        write_visualization(
            result["final_hamiltonian"],
            result["final_density"],
            plot_path,
            density_frames=result["density_frames"],
            electron_count=result["electron_count"],
            k_grid=result["k_grid"],
            show_plot=show_plot,
            show_density_evolution=show_density_evolution,
            density_quantile=density_quantile,
            save_figure=save_density_plot,
            repeat_count=visualization_repeat_count,
            repeat_max=visualization_repeat_max,
        )

    if save_density_npy:
        _save_density_npy(result["final_density"], density_npy_path)

    serializable_result = make_serializable(result)
    serializable_result["project_name"] = project_name
    serializable_result["outputs"] = {
        "result_json": result_path.name if save_result_json else None,
        "density_plot": plot_path.name if save_density_plot else None,
        "density_npy": density_npy_path.name if save_density_npy else None,
    }
    serializable_result["visualization"] = {
        "density_plot": str(plot_path.name) if save_density_plot else None,
        "view": "3d density evolution with draggable slider" if show_density_evolution else "3d density",
        "frame_count": len(result["density_frames"]) if (save_density_plot or show_plot) else 0,
        "show_plot": show_plot,
        "show_density_evolution": show_density_evolution,
        "repeat_count": visualization_repeat_count,
        "repeat_max": visualization_repeat_max,
    }
    if save_result_json:
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable_result, handle, indent=2)

    print(json.dumps(serializable_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())