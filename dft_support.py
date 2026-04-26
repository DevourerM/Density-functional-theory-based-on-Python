from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Sequence
from xml.etree import ElementTree

import numpy as np
from scipy import special


@dataclass(frozen=True)
class AtomSite:
    element: str
    fractional_position: tuple[float, float, float]


@dataclass(frozen=True)
class NonlocalProjectorRadial:
    beta_index: int
    angular_momentum: int
    radial_values: tuple[float, ...]
    cutoff_radius: float
    cutoff_radius_index: int


@dataclass(frozen=True)
class NormConservingPseudopotential:
    element: str
    functional: str
    z_valence: float
    radial_grid: tuple[float, ...]
    local_potential: tuple[float, ...]
    projectors: tuple[NonlocalProjectorRadial, ...]
    dij: tuple[tuple[float, ...], ...]
    file_path: str

    def evaluate_local(self, radius: np.ndarray | float) -> np.ndarray:
        radius_array = np.asarray(radius, dtype=float)
        radial_grid = np.asarray(self.radial_grid, dtype=float)
        local_potential = np.asarray(self.local_potential, dtype=float)
        clipped_radius = np.clip(radius_array, radial_grid[0], radial_grid[-1])
        interpolated = np.interp(clipped_radius, radial_grid, local_potential)
        coulomb_tail = -self.z_valence / np.maximum(radius_array, 1.0e-8)
        return np.where(radius_array <= radial_grid[-1], interpolated, coulomb_tail)

    def evaluate_projector_radial(self, projector_index: int, radius: np.ndarray | float) -> np.ndarray:
        radius_array = np.asarray(radius, dtype=float)
        projector = self.projectors[projector_index]
        radial_grid = np.asarray(self.radial_grid, dtype=float)
        radial_values = np.asarray(projector.radial_values, dtype=float)
        clipped_radius = np.clip(radius_array, radial_grid[0], radial_grid[-1])
        interpolated = np.interp(clipped_radius, radial_grid, radial_values)
        radial_part = interpolated / np.maximum(radius_array, 1.0e-8)
        return np.where(radius_array <= projector.cutoff_radius, radial_part, 0.0)


def parse_atom_sites(config: Dict[str, Any]) -> list[AtomSite]:
    atom_sites: list[AtomSite] = []
    default_element = str(config.get("default_element", "X"))

    for entry in config.get("pos", []):
        if isinstance(entry, dict):
            element = str(entry.get("element", default_element))
            position = entry.get("position")
            if not isinstance(position, Sequence) or len(position) != 3:
                raise ValueError("原子 position 必须是长度为 3 的序列。")
            fractional_position = tuple(float(value) for value in position)
            atom_sites.append(AtomSite(element=element, fractional_position=fractional_position))
            continue

        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
            raise ValueError("pos 中的每个原子必须是序列或字典。")

        if len(entry) == 4 and isinstance(entry[0], str):
            atom_sites.append(
                AtomSite(
                    element=str(entry[0]),
                    fractional_position=(float(entry[1]), float(entry[2]), float(entry[3])),
                )
            )
            continue

        if len(entry) >= 3 and not isinstance(entry[0], str):
            atom_sites.append(
                AtomSite(
                    element=default_element,
                    fractional_position=(float(entry[0]), float(entry[1]), float(entry[2])),
                )
            )
            continue

        raise ValueError("pos 中的原子格式不合法。请使用 [\"Ar\", x, y, z] 或 {element, position}。")

    return atom_sites


def infer_k_grid(config: Dict[str, Any]) -> tuple[int, int, int]:
    configured_grid = config.get("k_grid", config.get("K_grid", config.get("k_mesh", [1, 1, 1])))
    if not isinstance(configured_grid, Sequence) or isinstance(configured_grid, (str, bytes)):
        raise ValueError("k_grid 必须是长度为 3 的序列。")
    if len(configured_grid) != 3:
        raise ValueError("k_grid 必须包含 3 个维度。")
    return tuple(max(int(value), 1) for value in configured_grid)


def generate_monkhorst_pack_grid(k_grid: Sequence[int]) -> tuple[list[list[float]], list[float]]:
    axes = [
        [((2 * index + 1) - axis_points) / (2.0 * axis_points) for index in range(axis_points)]
        for axis_points in k_grid
    ]
    k_points = [
        [float(kx), float(ky), float(kz)]
        for kx in axes[0]
        for ky in axes[1]
        for kz in axes[2]
    ]
    weight = 1.0 / max(len(k_points), 1)
    weights = [weight for _ in k_points]
    return k_points, weights


def find_local_pseudopotential_file(
    element: str,
    functional: str,
    pseudopotential_dir: Path,
) -> Path:
    normalized_functional = functional.upper()
    candidates = [
        pseudopotential_dir / f"{element}_ONCV_{normalized_functional}-1.0.upf",
    ]
    if normalized_functional != "PBE":
        candidates.append(pseudopotential_dir / f"{element}_ONCV_PBE-1.0.upf")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"未找到元素 {element} 的 {normalized_functional} 赝势文件，搜索目录: {pseudopotential_dir}。"
    )


def _parse_numeric_node(node: ElementTree.Element | None) -> np.ndarray:
    if node is None:
        return np.array([], dtype=float)
    return np.fromstring(node.text or "", sep=" ", dtype=float)


@lru_cache(maxsize=128)
def load_norm_conserving_pseudopotential(file_path: str) -> NormConservingPseudopotential:
    path = Path(file_path)
    root = ElementTree.parse(path).getroot()

    header = root.find("PP_HEADER")
    radial_grid_node = root.find("./PP_MESH/PP_R")
    local_node = root.find("PP_LOCAL")
    nonlocal_node = root.find("PP_NONLOCAL")
    if header is None or radial_grid_node is None or local_node is None:
        raise ValueError(f"UPF 文件 {path} 缺少必要的 PP_HEADER / PP_R / PP_LOCAL 段。")

    radial_grid = _parse_numeric_node(radial_grid_node)
    local_potential_ry = _parse_numeric_node(local_node)
    if radial_grid.size == 0 or local_potential_ry.size == 0:
        raise ValueError(f"UPF 文件 {path} 的 PP_R 或 PP_LOCAL 为空。")
    if radial_grid.size != local_potential_ry.size:
        raise ValueError(f"UPF 文件 {path} 的 PP_R 与 PP_LOCAL 长度不一致。")

    projector_nodes = []
    dij_matrix = np.zeros((0, 0), dtype=float)
    if nonlocal_node is not None:
        beta_nodes = [node for node in nonlocal_node if node.tag.startswith("PP_BETA")]
        projector_nodes = sorted(beta_nodes, key=lambda node: int(node.attrib.get("index", node.tag.split(".")[-1])))
        dij_values = _parse_numeric_node(nonlocal_node.find("PP_DIJ"))
        if projector_nodes:
            projector_count = len(projector_nodes)
            if dij_values.size != projector_count * projector_count:
                raise ValueError(f"UPF 文件 {path} 的 PP_DIJ 大小与投影子数不匹配。")
            dij_matrix = dij_values.reshape((projector_count, projector_count)) * 0.5

    projectors: list[NonlocalProjectorRadial] = []
    for projector_node in projector_nodes:
        radial_values = _parse_numeric_node(projector_node)
        if radial_values.size != radial_grid.size:
            raise ValueError(f"UPF 文件 {path} 的 {projector_node.tag} 长度与 PP_R 不一致。")
        projectors.append(
            NonlocalProjectorRadial(
                beta_index=int(projector_node.attrib.get("index", projector_node.tag.split(".")[-1])),
                angular_momentum=int(projector_node.attrib.get("angular_momentum", "0")),
                radial_values=tuple(radial_values.tolist()),
                cutoff_radius=float(projector_node.attrib.get("cutoff_radius", radial_grid[-1])),
                cutoff_radius_index=int(projector_node.attrib.get("cutoff_radius_index", radial_grid.size)),
            )
        )

    return NormConservingPseudopotential(
        element=str(header.attrib.get("element", path.stem.split("_")[0])),
        functional=str(header.attrib.get("functional", "PBE")).upper(),
        z_valence=float(header.attrib.get("z_valence", "0.0")),
        radial_grid=tuple(radial_grid.tolist()),
        local_potential=tuple((0.5 * local_potential_ry).tolist()),
        projectors=tuple(projectors),
        dij=tuple(tuple(row.tolist()) for row in dij_matrix),
        file_path=str(path),
    )


def load_local_pseudopotential(file_path: str) -> NormConservingPseudopotential:
    return load_norm_conserving_pseudopotential(file_path)


def spherical_harmonic(
    angular_momentum: int,
    magnetic_number: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    if hasattr(special, "sph_harm"):
        return special.sph_harm(magnetic_number, angular_momentum, phi, theta)
    return special.sph_harm_y(angular_momentum, magnetic_number, theta, phi)


def _compute_gradients(field: np.ndarray, grid_spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradients = []
    for axis, spacing in enumerate(grid_spacing):
        gradient = (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * float(spacing))
        gradients.append(gradient)
    return gradients[0], gradients[1], gradients[2]


def _compute_divergence(vector_fields: Sequence[np.ndarray], grid_spacing: Sequence[float]) -> np.ndarray:
    divergence = np.zeros_like(vector_fields[0], dtype=float)
    for axis, spacing in enumerate(grid_spacing):
        divergence += (np.roll(vector_fields[axis], -1, axis=axis) - np.roll(vector_fields[axis], 1, axis=axis)) / (2.0 * float(spacing))
    return divergence


def _pw92_correlation_energy_density(density: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
    positive_density = np.clip(density, 1.0e-14, None)
    rs = np.cbrt(3.0 / (4.0 * np.pi * positive_density))
    a = 0.0310907
    alpha1 = 0.21370
    beta1 = 7.5957
    beta2 = 3.5876
    beta3 = 1.6382
    beta4 = 0.49294
    denominator = 2.0 * a * (
        beta1 * np.sqrt(rs)
        + beta2 * rs
        + beta3 * rs ** 1.5
        + beta4 * rs * rs
    )
    return -2.0 * a * (1.0 + alpha1 * rs) * np.log1p(1.0 / np.maximum(denominator, 1.0e-14))


def _lda_exchange_energy_density(density: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
    positive_density = np.clip(density, 1.0e-14, None)
    return -(3.0 / 4.0) * np.cbrt(3.0 / np.pi) * np.cbrt(positive_density)


def _pbe_exchange_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    positive_density = np.clip(density, 1.0e-14, None)
    positive_sigma = np.clip(sigma, 0.0, None)
    grad_norm = np.sqrt(positive_sigma)
    kf = np.cbrt(3.0 * np.pi * np.pi * positive_density)
    s = grad_norm / (2.0 * kf * positive_density + 1.0e-14)
    kappa = 0.804
    mu = 0.2195149727645171
    enhancement = 1.0 + kappa - kappa / (1.0 + mu * s * s / kappa)
    return _lda_exchange_energy_density(positive_density) * enhancement


def _pbe_correlation_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    positive_density = np.clip(density, 1.0e-14, None)
    positive_sigma = np.clip(sigma, 0.0, None)
    grad_norm = np.sqrt(positive_sigma)
    epsilon_c_lda = _pw92_correlation_energy_density(positive_density)
    gamma = 0.031090690869654895
    beta = 0.06672455060314922
    kf = np.cbrt(3.0 * np.pi * np.pi * positive_density)
    screening = np.sqrt(4.0 * kf / np.pi)
    t = grad_norm / (2.0 * screening * positive_density + 1.0e-14)
    a_parameter = beta / gamma / (np.exp(-epsilon_c_lda / gamma) - 1.0 + 1.0e-14)
    numerator = (beta / gamma) * t * t * (1.0 + a_parameter * t * t)
    denominator = 1.0 + a_parameter * t * t + a_parameter * a_parameter * t ** 4
    h = gamma * np.log1p(numerator / np.maximum(denominator, 1.0e-14))
    return epsilon_c_lda + h


def _functional_derivative(
    density: np.ndarray,
    grid_spacing: Sequence[float],
    energy_density_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    positive_density = np.clip(density, 1.0e-14, None)
    gradients = _compute_gradients(positive_density, grid_spacing)
    sigma = sum(component * component for component in gradients)

    epsilon = energy_density_function(positive_density, sigma)
    energy_per_volume = positive_density * epsilon

    density_step = 1.0e-6 * np.maximum(positive_density, 1.0)
    density_plus = positive_density + density_step
    density_minus = np.maximum(positive_density - density_step, 1.0e-14)
    epsilon_plus = energy_density_function(density_plus, sigma)
    epsilon_minus = energy_density_function(density_minus, sigma)
    energy_plus = density_plus * epsilon_plus
    energy_minus = density_minus * epsilon_minus
    dfdn = (energy_plus - energy_minus) / np.maximum(density_plus - density_minus, 1.0e-14)

    sigma_step = 1.0e-6 * np.maximum(sigma, 1.0)
    sigma_plus = sigma + sigma_step
    sigma_minus = np.maximum(sigma - sigma_step, 0.0)
    epsilon_sigma_plus = energy_density_function(positive_density, sigma_plus)
    epsilon_sigma_minus = energy_density_function(positive_density, sigma_minus)
    energy_sigma_plus = positive_density * epsilon_sigma_plus
    energy_sigma_minus = positive_density * epsilon_sigma_minus
    dfdsigma = (energy_sigma_plus - energy_sigma_minus) / np.maximum(sigma_plus - sigma_minus, 1.0e-14)

    vector_fields = [2.0 * dfdsigma * gradient for gradient in gradients]
    divergence_term = _compute_divergence(vector_fields, grid_spacing)
    potential = dfdn - divergence_term
    return epsilon, potential


def evaluate_exchange_correlation_potential(
    density: np.ndarray,
    grid_spacing: Sequence[float],
    functional: str,
) -> Dict[str, np.ndarray]:
    normalized_functional = functional.upper()
    if normalized_functional not in {"LDA", "PBE"}:
        raise ValueError(f"暂不支持交换关联泛函 {functional}，仅支持 LDA 或 PBE。")

    if normalized_functional == "LDA":
        exchange_energy_density, exchange_potential = _functional_derivative(
            density,
            grid_spacing,
            _lda_exchange_energy_density,
        )
        correlation_energy_density, correlation_potential = _functional_derivative(
            density,
            grid_spacing,
            _pw92_correlation_energy_density,
        )
    else:
        exchange_energy_density, exchange_potential = _functional_derivative(
            density,
            grid_spacing,
            _pbe_exchange_energy_density,
        )
        correlation_energy_density, correlation_potential = _functional_derivative(
            density,
            grid_spacing,
            _pbe_correlation_energy_density,
        )

    return {
        "exchange_energy_density": exchange_energy_density,
        "correlation_energy_density": correlation_energy_density,
        "exchange": exchange_potential,
        "correlation": correlation_potential,
        "total": exchange_potential + correlation_potential,
    }
