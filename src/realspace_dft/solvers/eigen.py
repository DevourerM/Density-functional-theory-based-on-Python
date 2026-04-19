"""基于 LOBPCG 的最小块本征求解。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lobpcg

from ..config.constants import ALLOWED_EIGENSOLVER_METHODS
from ..config.exceptions import DFTInputError, UnsupportedEigenSolverError
from ..core.models import ComplexArray, FloatArray

if TYPE_CHECKING:
    from ..core.models import RealSpaceGrid

HamiltonianLike = LinearOperator | np.ndarray


@dataclass(slots=True, frozen=True)
class IterativeEigenSolverConfig:
    """控制波函数块迭代求解的参数。"""

    method: str = "LOBPCG"
    tolerance: float = 1.0e-8
    max_iterations: int = 200
    largest: bool = False
    random_seed: int = 0
    verbosity_level: int = 0

    @property
    def normalized_method(self) -> str:
        """返回大写标准化后的求解器方法名。"""

        return self.method.strip().upper()


@dataclass(slots=True)
class IterativeEigenSolverResult:
    """保存本征求解得到的本征值与波函数。"""

    solver_method: str
    eigenvalues_hartree: FloatArray
    eigenvectors: ComplexArray
    final_residual_norms: FloatArray | None
    lambda_history: tuple[FloatArray, ...]
    residual_history: tuple[FloatArray, ...]
    grid_shape: tuple[int, int, int] | None = None
    volume_element_bohr3: float | None = None

    @property
    def nbands(self) -> int:
        """返回求解得到的波函数数目。"""

        return int(self.eigenvectors.shape[1])

    @property
    def wavefunctions(self) -> ComplexArray:
        """按 (nbands, 维度) 返回波函数向量。"""

        return self.eigenvectors.T.copy()

    @property
    def converged(self) -> bool:
        """根据最终残差判断是否达到收敛。"""

        if self.final_residual_norms is None:
            return False
        return bool(np.all(np.isfinite(self.final_residual_norms)))

    def wavefunctions_as_grids(self) -> ComplexArray:
        """把波函数恢复成 (nx, ny, nz, nbands) 形状。"""

        if self.grid_shape is None:
            raise ValueError("当前结果不包含网格形状，无法恢复三维波函数。")

        return self.eigenvectors.reshape(self.grid_shape + (self.nbands,), order="C")

    def overlap_matrix(self) -> ComplexArray:
        """返回波函数的重叠矩阵。"""

        overlap = self.eigenvectors.conj().T @ self.eigenvectors
        if self.volume_element_bohr3 is not None:
            overlap = self.volume_element_bohr3 * overlap
        return overlap


class BaseBandEigenSolver(ABC):
    """所有块本征求解器的公共抽象接口。"""

    def __init__(self, config: IterativeEigenSolverConfig | None = None):
        self.config = config or IterativeEigenSolverConfig()

    @abstractmethod
    def solve(
        self,
        hamiltonian: HamiltonianLike,
        nbands: int,
    ) -> IterativeEigenSolverResult:
        """求解最低若干个本征态。"""



def _as_linear_operator(hamiltonian: HamiltonianLike) -> LinearOperator:
    """把输入统一包装成 LinearOperator。"""

    base_operator = hamiltonian if isinstance(hamiltonian, LinearOperator) else aslinearoperator(hamiltonian)
    wrapper_dtype = np.result_type(base_operator.dtype or np.float64, np.complex128)

    def matvec(vector: np.ndarray) -> np.ndarray:
        flattened_vector = np.asarray(vector, dtype=wrapper_dtype).reshape(-1)
        return np.asarray(base_operator.matvec(flattened_vector), dtype=wrapper_dtype).reshape(-1)

    def matmat(matrix: np.ndarray) -> np.ndarray:
        matrix_array = np.asarray(matrix, dtype=wrapper_dtype)
        if matrix_array.ndim == 1:
            return matvec(matrix_array)

        try:
            multiplied = base_operator.matmat(matrix_array)
            return np.asarray(multiplied, dtype=wrapper_dtype)
        except Exception:
            return np.column_stack([matvec(matrix_array[:, index]) for index in range(matrix_array.shape[1])])

    return LinearOperator(
        shape=base_operator.shape,
        matvec=matvec,
        matmat=matmat,
        dtype=wrapper_dtype,
    )



def _extract_grid(hamiltonian: HamiltonianLike) -> RealSpaceGrid | None:
    """尝试从哈密顿量对象中提取网格信息。"""

    grid = getattr(hamiltonian, "grid", None)
    if grid is None:
        return None
    return grid



def _prepare_initial_subspace(
    operator: LinearOperator,
    nbands: int,
    random_seed: int,
) -> np.ndarray:
    """构造并正交化随机初始子空间。"""

    if operator.shape[0] != operator.shape[1]:
        raise DFTInputError("哈密顿量必须是方阵算符。")

    dimension = operator.shape[0]
    if nbands <= 0:
        raise DFTInputError("nbands 必须大于 0。")
    if nbands >= dimension:
        raise DFTInputError("nbands 必须严格小于哈密顿量维度，LOBPCG 才能稳定工作。")

    operator_dtype = np.dtype(operator.dtype or np.float64)
    is_complex = np.issubdtype(operator_dtype, np.complexfloating)
    rng = np.random.default_rng(random_seed)
    block = rng.standard_normal((dimension, nbands))
    if is_complex:
        block = block + 1j * rng.standard_normal((dimension, nbands))

    q_matrix, _ = np.linalg.qr(block.astype(operator_dtype, copy=False), mode="reduced")
    return q_matrix.astype(operator_dtype, copy=False)



def _physically_normalize_eigenvectors(
    eigenvectors: np.ndarray,
    grid: RealSpaceGrid | None,
) -> np.ndarray:
    """若可用，则按实空间体积元对波函数进行物理归一化。"""

    if grid is None:
        return np.asarray(eigenvectors, dtype=np.complex128)

    normalized = np.asarray(eigenvectors, dtype=np.complex128).copy()
    norms = np.sqrt(grid.volume_element_bohr3 * np.sum(np.abs(normalized) ** 2, axis=0))
    safe_norms = np.where(norms > 1.0e-14, norms, 1.0)
    normalized /= safe_norms[np.newaxis, :]
    return normalized



def _coerce_real_array(values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """把理论上应为实数的结果安全转换为 float 数组。"""

    real_like_values = np.real_if_close(np.asarray(values), tol=1000)
    if np.iscomplexobj(real_like_values):
        real_like_values = real_like_values.real
    return np.asarray(real_like_values, dtype=np.float64)


class LOBPCGBandEigenSolver(BaseBandEigenSolver):
    """基于 LOBPCG 的块本征态求解器。"""

    def solve(
        self,
        hamiltonian: HamiltonianLike,
        nbands: int,
    ) -> IterativeEigenSolverResult:
        operator = _as_linear_operator(hamiltonian)
        initial_block = _prepare_initial_subspace(
            operator=operator,
            nbands=nbands,
            random_seed=self.config.random_seed,
        )

        eigenvalues, eigenvectors, lambda_history, residual_history = lobpcg(
            operator,
            initial_block,
            largest=self.config.largest,
            tol=self.config.tolerance,
            maxiter=self.config.max_iterations,
            verbosityLevel=self.config.verbosity_level,
            retLambdaHistory=True,
            retResidualNormsHistory=True,
        )

        order = np.argsort(eigenvalues)
        if self.config.largest:
            order = order[::-1]

        sorted_eigenvalues = _coerce_real_array(eigenvalues[order])
        sorted_eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
        grid = _extract_grid(hamiltonian)
        normalized_eigenvectors = _physically_normalize_eigenvectors(sorted_eigenvectors, grid)

        final_residual_norms = None
        if residual_history:
            final_residual_norms = _coerce_real_array(residual_history[-1])[order]

        return IterativeEigenSolverResult(
            solver_method="LOBPCG",
            eigenvalues_hartree=sorted_eigenvalues,
            eigenvectors=normalized_eigenvectors,
            final_residual_norms=final_residual_norms,
            lambda_history=tuple(_coerce_real_array(values) for values in lambda_history),
            residual_history=tuple(_coerce_real_array(values) for values in residual_history),
            grid_shape=None if grid is None else grid.shape,
            volume_element_bohr3=None if grid is None else grid.volume_element_bohr3,
        )



def create_band_eigensolver(
    config: IterativeEigenSolverConfig | None = None,
) -> BaseBandEigenSolver:
    """根据配置创建块本征求解器。"""

    solver_config = config or IterativeEigenSolverConfig()
    normalized_method = solver_config.normalized_method
    if normalized_method not in ALLOWED_EIGENSOLVER_METHODS:
        allowed_methods = ", ".join(sorted(ALLOWED_EIGENSOLVER_METHODS))
        raise DFTInputError(f"本征求解器方法仅支持: {allowed_methods}")
    if normalized_method != "LOBPCG":
        raise UnsupportedEigenSolverError(f"未知的本征求解器方法: {normalized_method}")
    return LOBPCGBandEigenSolver(solver_config)



def 求解本征波函数(
    hamiltonian: HamiltonianLike,
    nbands: int,
    *,
    solver_config: IterativeEigenSolverConfig | None = None,
) -> IterativeEigenSolverResult:
    """以哈密顿量为输入，求解最低若干个 band 的波函数。"""

    solver = create_band_eigensolver(solver_config)
    return solver.solve(hamiltonian, nbands)


本征求解配置 = IterativeEigenSolverConfig
本征求解结果 = IterativeEigenSolverResult
LOBPCG波函数求解器 = LOBPCGBandEigenSolver
创建波函数求解器 = create_band_eigensolver
