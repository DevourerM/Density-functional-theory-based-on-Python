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


class DIISDensityMixer(BaseDensityMixer):
    """基于残差最小化的密度 DIIS 混合。"""

    def __init__(self, mixing_config: MixingConfig):
        super().__init__(mixing_config)
        self._output_history: list[np.ndarray] = []
        self._residual_history: list[np.ndarray] = []

    def mix(self, input_density: np.ndarray, output_density: np.ndarray) -> np.ndarray:
        density_in = np.asarray(input_density, dtype=np.float64)
        density_out = np.asarray(output_density, dtype=np.float64)
        residual = (density_out - density_in).reshape(-1).copy()
        output_flat = density_out.reshape(-1).copy()

        self._output_history.append(output_flat)
        self._residual_history.append(residual)

        history_size = self.mixing_config.diis_history_steps
        if len(self._output_history) > history_size:
            self._output_history.pop(0)
            self._residual_history.pop(0)

        if len(self._output_history) < 2:
            return density_in + self.mixing_config.linear_coefficient * (density_out - density_in)

        diis_dimension = len(self._residual_history)
        b_matrix = np.empty((diis_dimension + 1, diis_dimension + 1), dtype=np.float64)
        b_matrix[:-1, :-1] = 0.0
        for row_index in range(diis_dimension):
            for column_index in range(diis_dimension):
                b_matrix[row_index, column_index] = float(
                    np.vdot(self._residual_history[row_index], self._residual_history[column_index]).real
                )

        b_matrix[:-1, -1] = -1.0
        b_matrix[-1, :-1] = -1.0
        b_matrix[-1, -1] = 0.0

        right_hand_side = np.zeros(diis_dimension + 1, dtype=np.float64)
        right_hand_side[-1] = -1.0

        try:
            coefficients = np.linalg.solve(b_matrix, right_hand_side)[:-1]
        except np.linalg.LinAlgError:
            return density_in + self.mixing_config.linear_coefficient * (density_out - density_in)

        mixed_flat = np.zeros_like(output_flat)
        for coefficient, density_history in zip(coefficients, self._output_history, strict=True):
            mixed_flat += coefficient * density_history
        return mixed_flat.reshape(density_in.shape)


def create_density_mixer(mixing_config: MixingConfig) -> BaseDensityMixer:
    """根据输入配置构造密度混合器。"""

    normalized_method = mixing_config.method.strip()
    if normalized_method == "linear":
        return LinearDensityMixer(mixing_config)
    if normalized_method == "DIIS":
        return DIISDensityMixer(mixing_config)
    raise DFTInputError(f"不支持的密度混合方法: {mixing_config.method}")


创建密度混合器 = create_density_mixer
