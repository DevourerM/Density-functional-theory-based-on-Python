"""最小完整计算流程编排。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.models import RuntimeContext
from ..reporting.persistence import 保存最终电荷密度TXT
from ..reporting.tensorboard import 写入SCF过程到TensorBoard, 清空日志目录
from .bootstrap import 初始化计算上下文
from .scf import SCFResult, 运行SCF循环


@dataclass(slots=True)
class PipelineResult:
    """保存一次完整计算流程的输出。"""

    context: RuntimeContext
    scf_result: SCFResult
    log_dir: Path
    density_txt_path: Path

def _解析输出路径(base_dir: Path, path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()



def 执行完整计算流程(
    input_path: str | Path = "INPUT.json",
    *,
    log_dir: str | Path = "logs",
    output_dir: str | Path = "outputs",
    clear_logs: bool = True,
) -> PipelineResult:
    """执行初始化、SCF、TensorBoard 写出和最终密度导出。"""

    resolved_input_path = Path(input_path).resolve()
    base_dir = resolved_input_path.parent
    resolved_log_dir = _解析输出路径(base_dir, log_dir)
    resolved_output_dir = _解析输出路径(base_dir, output_dir)

    if clear_logs:
        清空日志目录(resolved_log_dir)
    else:
        resolved_log_dir.mkdir(parents=True, exist_ok=True)

    context = 初始化计算上下文(resolved_input_path)
    scf_result = 运行SCF循环(initial_context=context)
    写入SCF过程到TensorBoard(resolved_log_dir, scf_result)
    density_txt_path = 保存最终电荷密度TXT(resolved_output_dir, scf_result)
    return PipelineResult(
        context=context,
        scf_result=scf_result,
        log_dir=resolved_log_dir,
        density_txt_path=density_txt_path,
    )
