"""结果保存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..workflows.scf import SCFResult


def 保存最终电荷密度TXT(output_dir: str | Path, scf_result: SCFResult) -> Path:
    """把最终电荷密度保存为带坐标的 txt 文件。"""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    grid = scf_result.final_hamiltonian.grid
    density_values = scf_result.final_density.values
    x_grid, y_grid, z_grid = grid.cartesian_coordinate_arrays_bohr

    index_grid = np.indices(grid.shape, dtype=np.int64)
    table = np.column_stack(
        [
            index_grid[0].reshape(-1, order="C"),
            index_grid[1].reshape(-1, order="C"),
            index_grid[2].reshape(-1, order="C"),
            x_grid.reshape(-1, order="C"),
            y_grid.reshape(-1, order="C"),
            z_grid.reshape(-1, order="C"),
            density_values.reshape(-1, order="C"),
        ]
    )

    output_path = resolved_output_dir / "final_density.txt"
    np.savetxt(
        output_path,
        table,
        fmt=["%d", "%d", "%d", "%.12e", "%.12e", "%.12e", "%.12e"],
        header=(
            "ix iy iz x_bohr y_bohr z_bohr density_bohr^-3\n"
            f"grid_shape={grid.shape} dvol_bohr3={grid.volume_element_bohr3:.12e} "
            f"integrated_electrons={scf_result.final_density.integrated_electrons:.12e}"
        ),
        comments="# ",
    )
    return output_path