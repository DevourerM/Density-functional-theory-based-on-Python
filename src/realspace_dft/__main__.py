"""命令行入口：执行完整实空间 DFT 计算流程。"""

from __future__ import annotations

import argparse

from .config.exceptions import DFTInputError
from .workflows.pipeline import 执行完整计算流程



def main() -> None:
    """执行完整计算并打印简要摘要。"""

    parser = argparse.ArgumentParser(description="执行实空间 DFT SCF 计算")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="INPUT.json",
        help="输入参数 JSON 文件路径",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="TensorBoard 日志目录，默认相对输入文件目录创建 logs",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="结果输出目录，默认相对输入文件目录创建 outputs",
    )
    args = parser.parse_args()

    try:
        pipeline_result = 执行完整计算流程(
            input_path=args.input_path,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            clear_logs=True,
        )
    except (DFTInputError, FileNotFoundError) as exc:
        raise SystemExit(f"运行失败: {exc}") from exc

    print(pipeline_result.context.summary())
    print()
    print(pipeline_result.scf_result.summary())
    print()
    print(f"TensorBoard 日志目录: {pipeline_result.log_dir}")
    print(f"最终电荷密度文件: {pipeline_result.density_file_path}")
    print(f"SCF 摘要文件: {pipeline_result.summary_file_path}")


if __name__ == "__main__":
    main()
