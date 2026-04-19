# Density-functional-theory-based-on-Python

这是一个基于 Python 的三维实空间 Kohn-Sham DFT SCF 程序。当前版本已经可以作为一个可直接运行的单 Γ 点周期性 DFT SCF 软件使用，支持从 INPUT.json 读取结构与数值参数，执行自洽迭代，并输出 TensorBoard 日志、最终电荷密度和总能摘要。

## 当前能力

- 三维周期性实空间网格离散
- 矩阵自由动能算符
- UPF 局域赝势读取与插值
- UPF 非局域 Kleinman-Bylander projector 项
- Hartree 势 FFT 泊松求解
- LDA 与 PBE 交换关联势/能量
- LOBPCG 波函数迭代求解
- 线性混合与 DIIS 密度混合
- SCF 过程中记录密度残差、波函数残差、总能变化
- 离子-离子 Ewald 常数项，输出可比较的晶体总能
- TensorBoard 可视化
- 最终电荷密度和 SCF 摘要保存

## 当前限制

当前版本仍然是教学型/研究原型实现，已经可用于小体系的 SCF 试算，但还没有以下能力：

- 仅支持单 Γ 点，不支持一般 k 点采样
- 仅支持非自旋极化，不支持自旋分辨密度
- 未实现原子力、应力、结构优化和分子动力学
- Davidson 求解器接口已预留，但当前未实现
- 未实现 smearing、费米能搜索和金属体系专用占据策略

如果你的目标是做小型绝缘体/半导体单胞的实空间 SCF 试算，这个版本已经可以直接跑通完整流程。

## 安装

建议先创建虚拟环境，再在仓库根目录安装为可编辑模式：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

项目依赖由 pyproject.toml 管理，当前核心依赖为：

- numpy
- scipy
- tensorboardX

## 输入文件

默认输入文件为仓库根目录下的 INPUT.json。当前输入项包括：

- 赝势目录路径
- 晶格常数与晶格基矢
- 原子分数坐标
- SCF 最大步数与收敛阈值
- 波函数求解阈值
- 密度混合方法与参数
- 实空间网格划分
- 交换关联泛函类型
- 轨道数 nbands

示例输入可直接参考仓库中的 INPUT.json。

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
4. 计算离子-离子 Ewald 常数项
5. 构造 Kohn-Sham 哈密顿量
6. 迭代求解波函数与电荷密度，直到满足 SCF 收敛判据
7. 写出日志、最终电荷密度和 SCF 摘要

## 输出内容

### 终端摘要

程序会打印：

- 输入与赝势信息
- 总价电子数
- 非局域 projector 数
- 离子-离子 Ewald 能
- SCF 是否收敛
- 最终电子总能
- 最终晶体总能

### TensorBoard 日志

每次运行前默认会清空 logs 目录，然后写入新的 event 文件。可视化命令：

```bash
tensorboard --logdir logs
```

当前会记录的主要标量包括：

- density residual
- mixed density change
- wavefunction residual
- electronic total energy
- crystal total energy
- kinetic / ionic local / nonlocal / Hartree / XC / Ewald 分项

### 输出文件

默认写入 outputs 目录：

- final_density.npz
	- 最终电荷密度
	- 网格形状
	- 电子数积分
	- 最终电子总能
	- 最终晶体总能
	- 离子-离子 Ewald 常数项
- scf_summary.json
	- 收敛状态
	- 迭代步数
	- 最终能量与能量分项
	- 最终电荷积分

## 总能约定

当前总能输出分为两层：

- 电子总能：动能、电子-离子局域项、非局域项、Hartree 项和 XC 项之和
- 晶体总能：电子总能再加上离子-离子 Ewald 常数项

其中 Ewald 项采用周期性点电荷求和，并包含与 `G=0` 库仑分量去除一致的均匀中性背景修正。这一点会影响该常数项本身的数值符号，但不影响当前代码内部总能定义的一致性。

## 测试

仓库当前提供了初始化、哈密顿量、本征求解器、Ewald 总能和 SCF 工作流的回归测试。运行方式：

```bash
python -m unittest discover -s tests -v
```

## 建议的下一步

如果要继续把它推进成更完整的 DFT 软件，优先顺序建议是：

1. 加入费米能搜索与分数占据
2. 加入 k 点采样
3. 加入原子力与结构优化
4. 加入自旋极化
5. 对 PBE 势和非局域 projector 做更系统的数值验证
