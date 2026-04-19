"""SCF 电荷密度混合算法。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config.exceptions import DFTInputError
from ..core.models import MixingConfig


class BaseDensityMixer(ABC):
    """电荷密度混合器的统一接口。"""

    def __init__(self, mixing_config: MixingConfig):
        self.mixing_config = mixing_config

    @abstractmethod
    def mix(self, input_density: np.ndarray, output_density: np.ndarray) -> np.ndarray:
        """将输入密度与新输出密度混合成下一轮的密度。"""


class LinearDensityMixer(BaseDensityMixer):
    """最简单的线性密度混合。"""

    def mix(self, input_density: np.ndarray, output_density: np.ndarray) -> np.ndarray:
        density_in = np.asarray(input_density, dtype=np.float64)
        density_out = np.asarray(output_density, dtype=np.float64)
        residual = density_out - density_in
        return density_in + self.mixing_config.linear_coefficient * residual


def create_density_mixer(mixing_config: MixingConfig) -> BaseDensityMixer:
    """根据输入配置构造密度混合器。"""

    normalized_method = mixing_config.method.strip()
    if normalized_method == "linear":
        return LinearDensityMixer(mixing_config)
    raise DFTInputError(f"不支持的密度混合方法: {mixing_config.method}")


创建密度混合器 = create_density_mixer
