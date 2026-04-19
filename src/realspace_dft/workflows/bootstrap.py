"""把输入解析、赝势读取与哈密顿初始化组织成统一入口。"""

from __future__ import annotations

import math
from pathlib import Path

from ..config.exceptions import DFTInputError
from ..config.input_parser import 加载输入参数
from ..core.models import InputConfig, PseudopotentialInfo, RuntimeContext
from ..mesh.density import 初始化电荷密度网格
from ..mesh.geometry import 构造实空间网格
from ..physics.ewald import 计算离子离子Ewald能
from ..physics.hamiltonian import 构造哈密顿算符
from ..physics.pseudopotential import 载入所需赝势


def _计算总价电子数(
    config: InputConfig,
    pseudopotentials: dict[str, PseudopotentialInfo],
) -> float:
    """根据元素个数与赝势的 z_valence 计算总价电子数。"""

    total_valence_electrons = 0.0
    for element, count in config.species_counts.items():
        total_valence_electrons += count * pseudopotentials[element].z_valence
    return total_valence_electrons


def _估算最小占据轨道数(total_valence_electrons: float) -> int:
    """默认按非自旋极化情形估算最小占据轨道数。"""

    return math.ceil(total_valence_electrons / 2.0 - 1.0e-12)


def 初始化计算上下文(input_path: str | Path = "INPUT.json") -> RuntimeContext:
    """执行初始化阶段所需的全部读取、校验与初始网格构造。"""

    config = 加载输入参数(input_path)
    pseudopotential_dir, pseudopotentials = 载入所需赝势(config)
    real_space_grid = 构造实空间网格(config.crystal, config.numerical.grid)
    total_valence_electrons = _计算总价电子数(config, pseudopotentials)
    minimum_occupied_bands = _估算最小占据轨道数(total_valence_electrons)
    ion_ion_ewald_hartree = 计算离子离子Ewald能(config.crystal, pseudopotentials)

    if config.numerical.nbands < minimum_occupied_bands:
        raise DFTInputError(
            "输入的轨道数 nbands={} 小于体系所需的最小占据轨道数 {}。"
            "当前按非自旋极化、每个轨道最多容纳 2 个电子估算，"
            "总价电子数为 {:.6f}。".format(
                config.numerical.nbands,
                minimum_occupied_bands,
                total_valence_electrons,
            )
        )

    density_grid = 初始化电荷密度网格(real_space_grid, total_valence_electrons)
    hamiltonian_components, hamiltonian = 构造哈密顿算符(
        config=config,
        real_space_grid=real_space_grid,
        density_grid=density_grid,
        pseudopotentials=pseudopotentials,
    )

    return RuntimeContext(
        config=config,
        pseudopotential_dir=pseudopotential_dir,
        pseudopotentials=pseudopotentials,
        real_space_grid=real_space_grid,
        total_valence_electrons=total_valence_electrons,
        minimum_occupied_bands=minimum_occupied_bands,
        ion_ion_ewald_hartree=ion_ion_ewald_hartree,
        density_grid=density_grid,
        hamiltonian_components=hamiltonian_components,
        hamiltonian=hamiltonian,
    )
