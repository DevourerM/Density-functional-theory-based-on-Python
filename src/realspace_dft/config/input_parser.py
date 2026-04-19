"""输入参数 JSON 的读取与校验。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import ALLOWED_MIXING_METHODS, ALLOWED_XC_FUNCTIONALS, ANGSTROM_TO_BOHR
from .exceptions import DFTInputError
from ..core.models import (
    AtomSite,
    CrystalStructure,
    GridSpec,
    InputConfig,
    MixingConfig,
    NumericalSettings,
    SCFSettings,
)


def _read_required(mapping: Mapping[str, Any], *keys: str) -> Any:
    """读取必填字段，并在缺失时给出清晰的路径提示。"""

    current: Any = mapping
    traversed_keys: list[str] = []
    for key in keys:
        traversed_keys.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise DFTInputError(f"缺少必填字段: {'/'.join(traversed_keys)}")
        current = current[key]
    return current


def _read_optional(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """读取可选字段，缺失时返回默认值。"""

    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _parse_positive_float(value: Any, field_name: str, *, allow_zero: bool = False) -> float:
    """把输入转为有限浮点数，并检查正性。"""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DFTInputError(f"字段 {field_name} 必须是浮点数。") from exc

    if not math.isfinite(parsed):
        raise DFTInputError(f"字段 {field_name} 必须是有限实数。")

    if allow_zero:
        if parsed < 0.0:
            raise DFTInputError(f"字段 {field_name} 不能为负数。")
    elif parsed <= 0.0:
        raise DFTInputError(f"字段 {field_name} 必须大于 0。")

    return parsed


def _parse_positive_int(value: Any, field_name: str, *, minimum: int = 1) -> int:
    """把输入转为整数并检查下界。"""

    if isinstance(value, bool):
        raise DFTInputError(f"字段 {field_name} 必须是整数，不能是布尔值。")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DFTInputError(f"字段 {field_name} 必须是整数。") from exc

    if parsed < minimum:
        raise DFTInputError(f"字段 {field_name} 必须大于等于 {minimum}。")

    return parsed


def _normalize_element_symbol(symbol: Any) -> str:
    """把元素符号规范化为常见写法，例如 si -> Si。"""

    if not isinstance(symbol, str):
        raise DFTInputError("原子元素字段必须是字符串。")

    stripped = symbol.strip()
    if not re.fullmatch(r"[A-Za-z]{1,2}", stripped):
        raise DFTInputError(f"元素符号 {symbol!r} 格式不合法。")

    return stripped[0].upper() + stripped[1:].lower()


def _convert_length_to_bohr(value: float, unit: str) -> float:
    """把输入的晶格常数统一转为 bohr。"""

    normalized_unit = unit.strip().lower()
    if normalized_unit in {"bohr", "au", "a.u."}:
        return value
    if normalized_unit in {"angstrom", "ang", "a"}:
        return value * ANGSTROM_TO_BOHR
    raise DFTInputError(f"不支持的晶格常数单位: {unit}")


def _parse_normalized_lattice_vectors(raw_vectors: Any) -> np.ndarray:
    """解析并检查归一化晶格基矢。"""

    try:
        vectors = np.asarray(raw_vectors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DFTInputError("归一化晶格基矢必须是 3x3 数值数组。") from exc

    if vectors.shape != (3, 3):
        raise DFTInputError("归一化晶格基矢必须是 3x3 数组。")

    if not np.isfinite(vectors).all():
        raise DFTInputError("归一化晶格基矢中不能出现 NaN 或无穷大。")

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1.0e-12):
        raise DFTInputError("归一化晶格基矢不能出现零向量。")

    if np.any(np.abs(norms - 1.0) > 1.0e-6):
        raise DFTInputError("归一化晶格基矢的每个基矢长度都必须接近 1。")

    return vectors


def _parse_atom_sites(raw_sites: Any) -> tuple[AtomSite, ...]:
    """解析原子列表，要求每个原子都提供元素与分数坐标。"""

    if not isinstance(raw_sites, list) or not raw_sites:
        raise DFTInputError("原子分数坐标必须是非空列表。")

    parsed_sites: list[AtomSite] = []
    for index, raw_site in enumerate(raw_sites, start=1):
        if not isinstance(raw_site, Mapping):
            raise DFTInputError(f"原子分数坐标中的第 {index} 项必须是对象。")

        element = _normalize_element_symbol(_read_required(raw_site, "元素"))
        coordinates = _read_required(raw_site, "坐标")

        try:
            fractional_position = tuple(float(value) for value in coordinates)
        except (TypeError, ValueError) as exc:
            raise DFTInputError(f"第 {index} 个原子的坐标必须是三个数值。") from exc

        if len(fractional_position) != 3:
            raise DFTInputError(f"第 {index} 个原子的坐标必须包含 3 个分量。")

        if not all(math.isfinite(value) for value in fractional_position):
            raise DFTInputError(f"第 {index} 个原子的坐标中包含非法数值。")

        parsed_sites.append(AtomSite(element=element, fractional_position=fractional_position))

    return tuple(parsed_sites)


def _parse_mixing_config(root: Mapping[str, Any]) -> MixingConfig:
    """解析密度混合配置，并根据所选方法检查参数。"""

    method_raw = str(_read_required(root, "SCF控制", "密度混合", "方法")).strip()
    if method_raw.lower() == "linear":
        method = "linear"
    elif method_raw.upper() == "DIIS":
        method = "DIIS"
    else:
        allowed_methods = ", ".join(sorted(ALLOWED_MIXING_METHODS))
        raise DFTInputError(f"密度混合方法仅支持: {allowed_methods}")

    linear_coefficient = _parse_positive_float(
        _read_optional(root, "SCF控制", "密度混合", "线性混合系数", default=0.3),
        "SCF控制/密度混合/线性混合系数",
    )
    if linear_coefficient > 1.0:
        raise DFTInputError("线性混合系数必须在 (0, 1] 区间内。")

    diis_history_steps = _parse_positive_int(
        _read_optional(root, "SCF控制", "密度混合", "DIIS历史步数", default=6),
        "SCF控制/密度混合/DIIS历史步数",
        minimum=2,
    )

    return MixingConfig(
        method=method,
        linear_coefficient=linear_coefficient,
        diis_history_steps=diis_history_steps,
    )


def _parse_grid_spec(root: Mapping[str, Any]) -> GridSpec:
    """解析三维实空间网格划分。"""

    shape = (
        _parse_positive_int(
            _read_required(root, "数值设置", "空间网格划分", "x方向网格数"),
            "数值设置/空间网格划分/x方向网格数",
            minimum=2,
        ),
        _parse_positive_int(
            _read_required(root, "数值设置", "空间网格划分", "y方向网格数"),
            "数值设置/空间网格划分/y方向网格数",
            minimum=2,
        ),
        _parse_positive_int(
            _read_required(root, "数值设置", "空间网格划分", "z方向网格数"),
            "数值设置/空间网格划分/z方向网格数",
            minimum=2,
        ),
    )
    return GridSpec(shape=shape)


def _validate_crystal(crystal: CrystalStructure) -> None:
    """检查晶胞体积是否有效。"""

    if crystal.cell_volume_bohr3 <= 1.0e-10:
        raise DFTInputError("晶胞体积接近 0，请检查晶格常数和基矢是否线性相关。")


def 读取输入文件(input_path: str | Path) -> dict[str, Any]:
    """从 JSON 文件读取原始输入字典。"""

    resolved_input_path = Path(input_path).expanduser().resolve()
    if not resolved_input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {resolved_input_path}")

    with resolved_input_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, dict):
        raise DFTInputError("输入 JSON 的顶层结构必须是对象。")

    return raw_data


def 加载输入参数(input_path: str | Path = "INPUT.json") -> InputConfig:
    """读取并校验输入参数，返回结构化配置对象。"""

    resolved_input_path = Path(input_path).expanduser().resolve()
    root = 读取输入文件(resolved_input_path)

    pseudopotential_dir_hint = str(_read_required(root, "路径设置", "赝势文件夹路径")).strip()
    if not pseudopotential_dir_hint:
        raise DFTInputError("路径设置/赝势文件夹路径 不能为空。")

    lattice_constant_value = _parse_positive_float(
        _read_required(root, "晶体结构", "晶格常数", "数值"),
        "晶体结构/晶格常数/数值",
    )
    lattice_constant_unit = str(_read_required(root, "晶体结构", "晶格常数", "单位")).strip()
    if not lattice_constant_unit:
        raise DFTInputError("晶体结构/晶格常数/单位 不能为空。")

    crystal = CrystalStructure(
        lattice_constant_bohr=_convert_length_to_bohr(lattice_constant_value, lattice_constant_unit),
        lattice_constant_unit=lattice_constant_unit,
        normalized_lattice_vectors=_parse_normalized_lattice_vectors(
            _read_required(root, "晶体结构", "归一化晶格基矢")
        ),
        atom_sites=_parse_atom_sites(_read_required(root, "晶体结构", "原子分数坐标")),
    )
    _validate_crystal(crystal)

    scf_settings = SCFSettings(
        max_iterations=_parse_positive_int(
            _read_required(root, "SCF控制", "最大迭代次数"),
            "SCF控制/最大迭代次数",
        ),
        scf_tolerance=_parse_positive_float(
            _read_required(root, "SCF控制", "SCF收敛阈值"),
            "SCF控制/SCF收敛阈值",
        ),
        wavefunction_tolerance=_parse_positive_float(
            _read_required(root, "SCF控制", "波函数收敛阈值"),
            "SCF控制/波函数收敛阈值",
        ),
        mixing=_parse_mixing_config(root),
    )

    xc_functional = str(_read_required(root, "数值设置", "交换关联类型")).strip().upper()
    if xc_functional not in ALLOWED_XC_FUNCTIONALS:
        allowed_functionals = ", ".join(sorted(ALLOWED_XC_FUNCTIONALS))
        raise DFTInputError(f"交换关联类型仅支持: {allowed_functionals}")

    numerical_settings = NumericalSettings(
        grid=_parse_grid_spec(root),
        xc_functional=xc_functional,
        nbands=_parse_positive_int(
            _read_required(root, "数值设置", "轨道数"),
            "数值设置/轨道数",
        ),
    )

    return InputConfig(
        input_path=resolved_input_path,
        pseudopotential_dir_hint=pseudopotential_dir_hint,
        crystal=crystal,
        scf=scf_settings,
        numerical=numerical_settings,
    )


load_input_config = 加载输入参数
