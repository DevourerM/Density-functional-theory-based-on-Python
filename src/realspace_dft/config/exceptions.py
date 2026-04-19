"""项目内统一使用的异常类型。"""


class DFTInputError(ValueError):
    """表示输入参数缺失、格式错误或物理上不合理。"""


class PseudopotentialNotFoundError(FileNotFoundError):
    """表示未找到指定元素或交换关联泛函对应的赝势文件。"""


class EigenSolverError(RuntimeError):
    """表示波函数本征求解阶段的异常。"""


class UnsupportedEigenSolverError(EigenSolverError):
    """表示请求了当前尚未实现的本征求解器。"""
