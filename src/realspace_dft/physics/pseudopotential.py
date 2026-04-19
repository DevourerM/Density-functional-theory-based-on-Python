"""UPF 赝势文件的定位与头部信息读取。"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..config.exceptions import DFTInputError, PseudopotentialNotFoundError
from ..core.models import InputConfig, NonlocalProjectorRadial, PseudopotentialInfo


def _normalize_pseudopotential_dir(candidate: Path) -> Path | None:
    """把用户给定的路径解析到真正存放 .upf 文件的目录。"""

    if not candidate.exists() or not candidate.is_dir():
        return None

    direct_upf_files = list(candidate.glob("*.upf"))
    if direct_upf_files:
        return candidate.resolve()

    nested_upf_dirs = [
        child.resolve()
        for child in candidate.iterdir()
        if child.is_dir() and list(child.glob("*.upf"))
    ]
    if len(nested_upf_dirs) == 1:
        return nested_upf_dirs[0]
    if len(nested_upf_dirs) > 1:
        raise DFTInputError(
            f"赝势路径 {candidate} 下包含多个带 .upf 文件的子目录，请在输入文件中指向具体目录。"
        )
    return None


def _load_upf_root(file_path: str | Path) -> tuple[Path, ET.Element]:
    """读取并解析 UPF 文件根节点。"""

    resolved_path = Path(file_path).resolve()

    try:
        xml_tree = ET.parse(resolved_path)
    except ET.ParseError as exc:
        raise DFTInputError(f"无法解析 UPF 文件 {resolved_path}: {exc}") from exc

    return resolved_path, xml_tree.getroot()


def _extract_header_info(root: ET.Element, resolved_path: Path) -> PseudopotentialInfo:
    """从 XML 根节点中提取赝势头部元数据。"""

    header = root.find("PP_HEADER")
    if header is None:
        raise DFTInputError(f"UPF 文件 {resolved_path} 中缺少 PP_HEADER 节点。")

    element = header.attrib.get("element", "").strip() or resolved_path.stem.split("_")[0]
    functional = header.attrib.get("functional", "").strip().upper()
    if not functional:
        file_name_parts = resolved_path.stem.split("_")
        if len(file_name_parts) >= 3:
            functional = file_name_parts[2].split("-")[0].upper()

    z_valence_raw = header.attrib.get("z_valence", "")
    if not z_valence_raw:
        raise DFTInputError(f"UPF 文件 {resolved_path} 中缺少 z_valence 字段。")

    return PseudopotentialInfo(
        file_path=resolved_path,
        element=element,
        functional=functional,
        z_valence=float(z_valence_raw),
        pseudo_type=header.attrib.get("pseudo_type", "unknown").strip(),
    )


def _extract_real_array(
    root: ET.Element,
    resolved_path: Path,
    xpath: str,
    node_name: str,
) -> np.ndarray:
    """从指定 XML 节点中读取实数数组。"""

    node = root.find(xpath)
    if node is None or node.text is None:
        raise DFTInputError(f"UPF 文件 {resolved_path} 中缺少 {node_name} 数据。")

    values = np.fromstring(node.text, sep=" ", dtype=np.float64)
    if values.size == 0:
        raise DFTInputError(f"UPF 文件 {resolved_path} 的 {node_name} 数据为空。")
    return values


def _extract_nonlocal_projectors(root: ET.Element, resolved_path: Path) -> tuple[NonlocalProjectorRadial, ...]:
    """提取非局域 projector 的径向部分。"""

    nonlocal_root = root.find("./PP_NONLOCAL")
    if nonlocal_root is None:
        return ()

    projectors: list[NonlocalProjectorRadial] = []
    for child in nonlocal_root:
        if not child.tag.startswith("PP_BETA"):
            continue
        if child.text is None:
            raise DFTInputError(f"UPF 文件 {resolved_path} 中的 {child.tag} 数据为空。")

        radial_values = np.fromstring(child.text, sep=" ", dtype=np.float64)
        if radial_values.size == 0:
            raise DFTInputError(f"UPF 文件 {resolved_path} 中的 {child.tag} 数组为空。")

        projectors.append(
            NonlocalProjectorRadial(
                index=int(child.attrib.get("index", len(projectors) + 1)),
                angular_momentum=int(child.attrib.get("angular_momentum", "0")),
                radial_values=radial_values,
                cutoff_radius_bohr=float(child.attrib.get("cutoff_radius", "0.0")),
            )
        )

    return tuple(sorted(projectors, key=lambda projector: projector.index))


def _extract_dij_matrix_hartree(
    root: ET.Element,
    resolved_path: Path,
    projector_count: int,
) -> np.ndarray | None:
    """提取非局域耦合矩阵，并统一转换为 Hartree。"""

    nonlocal_root = root.find("./PP_NONLOCAL/PP_DIJ")
    if nonlocal_root is None or nonlocal_root.text is None:
        return None

    dij_values_ry = np.fromstring(nonlocal_root.text, sep=" ", dtype=np.float64)
    if dij_values_ry.size != projector_count * projector_count:
        raise DFTInputError(
            f"UPF 文件 {resolved_path} 的 PP_DIJ 维度与 projector 数量不匹配。"
        )
    return 0.5 * dij_values_ry.reshape((projector_count, projector_count))


def 解析赝势目录(path_hint: str, input_path: Path) -> Path:
    """解析输入文件中的赝势目录，并允许在项目内自动回溯定位。"""

    expanded_path = Path(os.path.expandvars(os.path.expanduser(path_hint)))
    candidate_paths: list[Path] = []

    if expanded_path.is_absolute():
        candidate_paths.append(expanded_path)
    else:
        candidate_paths.append((input_path.parent / expanded_path).resolve())
        candidate_paths.append((Path.cwd() / expanded_path).resolve())

    for candidate_path in candidate_paths:
        normalized = _normalize_pseudopotential_dir(candidate_path)
        if normalized is not None:
            return normalized

    target_dir_name = expanded_path.name
    matched_dirs: list[Path] = []
    for matched_path in input_path.parent.rglob(target_dir_name):
        normalized = _normalize_pseudopotential_dir(matched_path)
        if normalized is not None:
            matched_dirs.append(normalized)

    unique_matches = list(dict.fromkeys(matched_dirs))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        joined_matches = ", ".join(str(path) for path in unique_matches)
        raise DFTInputError(
            f"根据输入路径回溯找到了多个候选赝势目录，请在 INPUT.json 中明确指定其一: {joined_matches}"
        )

    raise PseudopotentialNotFoundError(f"未找到赝势目录: {path_hint}")


def 读取赝势头信息(file_path: str | Path) -> PseudopotentialInfo:
    """从 UPF 文件头部读取元素、泛函和价电子数等信息。"""

    resolved_path, root = _load_upf_root(file_path)
    return _extract_header_info(root, resolved_path)


def 读取赝势完整信息(file_path: str | Path) -> PseudopotentialInfo:
    """读取后续构造哈密顿量所需的完整赝势信息。"""

    resolved_path, root = _load_upf_root(file_path)
    header_info = _extract_header_info(root, resolved_path)
    radial_grid_bohr = _extract_real_array(root, resolved_path, "./PP_MESH/PP_R", "PP_R")
    local_potential_ry = _extract_real_array(root, resolved_path, "./PP_LOCAL", "PP_LOCAL")
    nonlocal_projectors = _extract_nonlocal_projectors(root, resolved_path)
    dij_matrix_hartree = _extract_dij_matrix_hartree(root, resolved_path, len(nonlocal_projectors))

    if radial_grid_bohr.size != local_potential_ry.size:
        raise DFTInputError(
            f"UPF 文件 {resolved_path} 中 PP_R 与 PP_LOCAL 的长度不一致。"
        )

    for projector in nonlocal_projectors:
        if projector.radial_values.size != radial_grid_bohr.size:
            raise DFTInputError(
                f"UPF 文件 {resolved_path} 中非局域 projector 长度与径向网格长度不一致。"
            )

    return PseudopotentialInfo(
        file_path=header_info.file_path,
        element=header_info.element,
        functional=header_info.functional,
        z_valence=header_info.z_valence,
        pseudo_type=header_info.pseudo_type,
        radial_grid_bohr=radial_grid_bohr,
        local_potential_hartree=0.5 * local_potential_ry,
        nonlocal_projectors=nonlocal_projectors,
        dij_matrix_hartree=dij_matrix_hartree,
    )


def _build_pseudopotential_index(
    pseudopotential_dir: Path,
) -> dict[tuple[str, str], list[PseudopotentialInfo]]:
    """按 元素 + 泛函 为键建立赝势索引。"""

    upf_files = sorted(pseudopotential_dir.glob("*.upf"))
    if not upf_files:
        raise PseudopotentialNotFoundError(f"赝势目录中没有找到 .upf 文件: {pseudopotential_dir}")

    index: dict[tuple[str, str], list[PseudopotentialInfo]] = defaultdict(list)
    for upf_file in upf_files:
        info = 读取赝势头信息(upf_file)
        index[(info.element, info.functional)].append(info)
    return dict(index)


def 载入所需赝势(config: InputConfig) -> tuple[Path, dict[str, PseudopotentialInfo]]:
    """按照输入中的元素种类和交换关联类型载入所需赝势。"""

    pseudopotential_dir = 解析赝势目录(config.pseudopotential_dir_hint, config.input_path)
    index = _build_pseudopotential_index(pseudopotential_dir)

    loaded_pseudopotentials: dict[str, PseudopotentialInfo] = {}
    for element in config.species:
        matches = index.get((element, config.numerical.xc_functional), [])

        if not matches:
            available_functionals = sorted(
                functional
                for (indexed_element, functional) in index
                if indexed_element == element
            )
            if available_functionals:
                raise PseudopotentialNotFoundError(
                    f"元素 {element} 没有找到 {config.numerical.xc_functional} 赝势。"
                    f"当前目录中可用泛函: {', '.join(available_functionals)}"
                )
            raise PseudopotentialNotFoundError(
                f"赝势目录 {pseudopotential_dir} 中没有元素 {element} 对应的赝势文件。"
            )

        if len(matches) > 1:
            matched_names = ", ".join(match.file_path.name for match in matches)
            raise DFTInputError(
                f"元素 {element} 和泛函 {config.numerical.xc_functional} 匹配到多个赝势文件: {matched_names}"
            )

        loaded_pseudopotentials[element] = 读取赝势完整信息(matches[0].file_path)

    return pseudopotential_dir, loaded_pseudopotentials
