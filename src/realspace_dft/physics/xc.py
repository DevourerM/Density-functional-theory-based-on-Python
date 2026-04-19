"""LDA/PBE 交换关联能量与势。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.models import DensityGrid, PotentialField, RealSpaceGrid
from ..mesh.differential import 计算梯度模平方, 计算笛卡尔散度, 计算笛卡尔梯度

PI = np.pi
THREE_PI_SQUARED = 3.0 * PI * PI
CX = 0.75 * (3.0 / PI) ** (1.0 / 3.0)
PBE_KAPPA = 0.804
PBE_MU = 0.2195149727645171
PBE_BETA = 0.06672455060314922
PW92_A = 0.031090690869654895
PW92_ALPHA1 = 0.21370
PW92_BETA1 = 7.5957
PW92_BETA2 = 3.5876
PW92_BETA3 = 1.6382
PW92_BETA4 = 0.49294
PBE_GAMMA = (1.0 - np.log(2.0)) / (PI * PI)


@dataclass(slots=True, frozen=True)
class ExchangeCorrelationData:
    """保存 XC 势与能量密度。"""

    potential: PotentialField
    energy_density_hartree_per_bohr3: np.ndarray
    total_energy_hartree: float
    model_name: str


def _safe_density(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1.0e-14, None)


def _lda_exchange_energy_density(density: np.ndarray) -> np.ndarray:
    safe_density = _safe_density(density)
    return -CX * np.power(safe_density, 4.0 / 3.0)


def _pw92_correlation_energy_per_particle(density: np.ndarray) -> np.ndarray:
    safe_density = _safe_density(density)
    rs = np.power(3.0 / (4.0 * PI * safe_density), 1.0 / 3.0)
    sqrt_rs = np.sqrt(rs)
    denominator = 2.0 * PW92_A * (
        PW92_BETA1 * sqrt_rs
        + PW92_BETA2 * rs
        + PW92_BETA3 * rs * sqrt_rs
        + PW92_BETA4 * rs * rs
    )
    return -2.0 * PW92_A * (1.0 + PW92_ALPHA1 * rs) * np.log1p(1.0 / denominator)


def _lda_correlation_energy_density(density: np.ndarray) -> np.ndarray:
    safe_density = _safe_density(density)
    return safe_density * _pw92_correlation_energy_per_particle(safe_density)


def _pbe_exchange_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    safe_density = _safe_density(density)
    safe_sigma = np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None)
    s2 = safe_sigma / (4.0 * np.power(THREE_PI_SQUARED, 2.0 / 3.0) * np.power(safe_density, 8.0 / 3.0))
    enhancement_factor = 1.0 + PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s2 / PBE_KAPPA)
    return _lda_exchange_energy_density(safe_density) * enhancement_factor


def _pbe_correlation_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    safe_density = _safe_density(density)
    safe_sigma = np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None)
    epsilon_c_lda = _pw92_correlation_energy_per_particle(safe_density)
    kf = np.power(THREE_PI_SQUARED * safe_density, 1.0 / 3.0)
    ks = np.sqrt(4.0 * kf / PI)
    gradient_norm = np.sqrt(safe_sigma)
    t = gradient_norm / (2.0 * ks * safe_density)
    t2 = t * t
    a_factor = (PBE_BETA / PBE_GAMMA) / (np.exp(-epsilon_c_lda / PBE_GAMMA) - 1.0)
    h_correction = PBE_GAMMA * np.log(
        1.0
        + (PBE_BETA / PBE_GAMMA) * t2 * (1.0 + a_factor * t2) / (1.0 + a_factor * t2 + a_factor * a_factor * t2 * t2)
    )
    return safe_density * (epsilon_c_lda + h_correction)


def 计算交换关联能量密度(
    density: np.ndarray,
    sigma: np.ndarray,
    xc_functional: str,
) -> tuple[np.ndarray, str]:
    """计算单位体积 XC 能量密度。"""

    if xc_functional == "LDA":
        energy_density = _lda_exchange_energy_density(density) + _lda_correlation_energy_density(density)
        return energy_density, "LDA exchange + PW92 correlation"
    if xc_functional == "PBE":
        energy_density = _pbe_exchange_energy_density(density, sigma) + _pbe_correlation_energy_density(density, sigma)
        return energy_density, "PBE exchange + PBE correlation"
    raise ValueError(f"不支持的交换关联类型: {xc_functional}")


def _有限差分偏导(
    density: np.ndarray,
    sigma: np.ndarray,
    xc_functional: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """对 XC 能量密度进行数值偏导。"""

    safe_density = _safe_density(density)
    safe_sigma = np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None)
    delta_density = np.maximum(1.0e-8, 1.0e-5 * safe_density)
    delta_sigma = np.maximum(1.0e-10, 1.0e-5 * (safe_sigma + 1.0e-8))

    energy_density, model_name = 计算交换关联能量密度(safe_density, safe_sigma, xc_functional)

    energy_density_plus_n, _ = 计算交换关联能量密度(safe_density + delta_density, safe_sigma, xc_functional)
    density_minus = np.maximum(safe_density - delta_density, 1.0e-14)
    energy_density_minus_n, _ = 计算交换关联能量密度(density_minus, safe_sigma, xc_functional)
    derivative_density = (energy_density_plus_n - energy_density_minus_n) / ((safe_density + delta_density) - density_minus)

    sigma_plus = safe_sigma + delta_sigma
    sigma_minus = np.maximum(safe_sigma - delta_sigma, 0.0)
    energy_density_plus_sigma, _ = 计算交换关联能量密度(safe_density, sigma_plus, xc_functional)
    energy_density_minus_sigma, _ = 计算交换关联能量密度(safe_density, sigma_minus, xc_functional)
    derivative_sigma = (energy_density_plus_sigma - energy_density_minus_sigma) / (sigma_plus - sigma_minus + 1.0e-20)

    return derivative_density, derivative_sigma, model_name


def 计算交换关联数据(
    real_space_grid: RealSpaceGrid,
    density_grid: DensityGrid,
    xc_functional: str,
) -> ExchangeCorrelationData:
    """计算 XC 势和 XC 总能。"""

    density_values = density_grid.values
    gradient = 计算笛卡尔梯度(density_values, real_space_grid)
    sigma = 计算梯度模平方(density_values, real_space_grid)
    energy_density, model_name = 计算交换关联能量密度(density_values, sigma, xc_functional)
    derivative_density, derivative_sigma, _ = _有限差分偏导(density_values, sigma, xc_functional)

    flux = tuple(2.0 * derivative_sigma * gradient_component for gradient_component in gradient)
    potential_values = derivative_density - 计算笛卡尔散度(flux, real_space_grid)
    total_energy_hartree = float(np.sum(energy_density) * real_space_grid.volume_element_bohr3)

    return ExchangeCorrelationData(
        potential=PotentialField(
            name="交换关联势",
            values=np.asarray(potential_values, dtype=np.float64),
            note=model_name,
            metadata={"请求泛函": xc_functional, "当前实现": model_name},
        ),
        energy_density_hartree_per_bohr3=np.asarray(energy_density, dtype=np.float64),
        total_energy_hartree=total_energy_hartree,
        model_name=model_name,
    )
