# Density-functional-theory-based-on-Python

这是一个面向周期固体自洽场闭环的三维实空间 Kohn-Sham DFT 原型程序。当前版本主路径包含显式 k 点、UPF 局域与非局域赝势、FFT Hartree 势、交换关联势、LOBPCG 本征求解、费米分数占据和线性密度混合。

## 当前能力

- 三维周期性实空间均匀网格离散
- 固定 7 点有限差分动能算符
- UPF 局域赝势读取与插值
- UPF 非局域 projector 读取、球谐展开与 Kleinman-Bylander 低秩作用
- Hartree 势 FFT 泊松求解
- LDA 与 PBE 交换关联势/能量
- LOBPCG 波函数迭代求解
- 显式 reduced k 点与权重输入
- 费米-狄拉克分数占据与费米能搜索
- 线性密度混合
- 基于密度残差和能级变化的最小 SCF 收敛判据
- TensorBoard 迭代可视化
- 最终电荷密度 txt 导出

## 当前限制

当前版本明确不包含以下增强功能：

- DIIS 混合
- Davidson 求解器
- Ewald 离子-离子常数项
- 自旋、力、应力、结构优化和对称性约简

如果你的目标是跑通一个最小、直接、可读的实空间 SCF 流程，这个版本就是当前仓库的主实现。

## 安装

建议先创建虚拟环境，再在仓库根目录安装为可编辑模式：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

项目当前核心依赖为：

- numpy
- scipy
- tensorboardX

## 输入文件

默认输入文件为仓库根目录下的 INPUT.json。当前最小输入项包括：

- 赝势目录路径
- 晶格常数与归一化晶格基矢
- 原子分数坐标
- SCF 最大步数与统一收敛阈值
- 线性混合系数
- 实空间网格划分
- 交换关联类型
- 轨道数 nbands
- 显式 k 点列表和权重
- 费米展宽与自旋简并度

仓库中提供了 [INPUT.json](INPUT.json) 和 [Ref.json](Ref.json) 作为模板。

## 运行

安装完成后，在仓库根目录运行：

```bash
python -m realspace_dft INPUT.json --log-dir logs --output-dir outputs
```

如果你没有执行 `pip install -e .`，也可以在仓库根目录临时指定源码目录：

```bash
set PYTHONPATH=src
python -m realspace_dft INPUT.json --log-dir logs --output-dir outputs
```

程序会自动执行以下流程：

1. 读取 INPUT.json
2. 载入所需 UPF 赝势
3. 构造实空间网格与初始密度
4. 对每个 k 点构造包含 Bloch 边界的局域与非局域 Kohn-Sham 哈密顿量
5. 对全部 k 点迭代求解波函数、费米能和电荷密度，直到满足 SCF 收敛判据
6. 写出 TensorBoard 日志和最终电荷密度 txt
7. 在终端打印初始化摘要和 SCF 摘要

## 输出内容

当前程序会在终端输出摘要，同时生成 TensorBoard 日志和最终电荷密度 txt。终端会打印：

- 输入与赝势信息
- 总价电子数与最小占据轨道数
- 网格规模和初始电荷积分
- SCF 是否收敛
- k 点数与费米能
- 最终密度残差
- 最终最大能级变化
- 最终总能
- 最终电荷积分

TensorBoard 用法：

```bash
tensorboard --logdir logs
```

默认输出文件：

- outputs/final_density.txt
	- 列顺序为 ix, iy, iz, x_bohr, y_bohr, z_bohr, density_bohr^-3

## 测试

仓库当前提供了初始化、哈密顿量、本征求解器和最小 SCF 工作流的回归测试。运行方式：

```bash
python -m unittest discover -s tests -v
```

## 备注

仓库自带的 SG15 赝势目录为 PBE 版本，因此示例输入默认使用 PBE。代码仍然保留 LDA/PBE 两种交换关联选项，但 SCF 算法本身已经收缩为最小版本。
