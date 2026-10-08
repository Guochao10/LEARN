# LEARN 项目说明与工作状态

本文档供维护本仓库的协作者和代码代理使用。项目状态记录截至 2026-10-08，实验结果和 Git 工作区可能继续变化；进行修改、提交或运行命令前，必须重新核对实际文件与状态。

## 项目定位

- 工作目录：`/data/home/gcjiang3/qsm/LEARN`。
- 当前维护的 PyTorch 实现是 LEARN-IMG，用于校正多回波 GRE 复数图像的运动伪影。输入和输出张量均为 `[B, 2, E, H, W]`，其中 `2` 为实部和虚部；当前 HDF5 实验有 4 个回波。
- `legacy_tf/` 归档原始 TensorFlow 代码，其中包含 LEARN-IMG 和 LEARN-BIO。PyTorch 分支尚未实现 LEARN-BIO，也没有 BIO 的 `S0`、`R2*` 输出及信号模型损失。分析当前 PyTorch 代码时，默认不要混入旧实现。
- 运动数据生成由相邻项目 `/data/home/gcjiang3/qsm/radial-qsm-motion` 负责；本仓库使用已生成的数据，不负责重写其生成流程。

## 目录与入口

| 路径 | 用途 |
| --- | --- |
| `README.md`、`pyproject.toml` | 项目说明、安装依赖及 `src` 打包配置。根目录已合并为一份 README。 |
| `configs/` | 各实验独立的 YAML。`data.format` 必须明确指定为 `h5` 或 `mat`。 |
| `scripts/train_learn_img.py` | LEARN-IMG 正式训练、验证、选最佳权重和续训。 |
| `scripts/evaluate_learn_img.py` | 对配对 HDF5 测试集一次完成推理、指标计算和预览图保存，不写入中间 `.mat` 预测体。 |
| `scripts/overfit_one_h5_slice.py` | 对一个训练配对的一层四回波复数图像反复拟合，用于单样本诊断。 |
| `scripts/smoke_test_learn_img.py` | 不依赖真实数据的前向、反向及 checkpoint 冒烟检查。 |
| `src/learn_motion/models/learn_img.py` | LEARN-IMG 各向异性 U-Net；只在图像平面池化，保留回波维。支持残差校正。 |
| `src/learn_motion/data/h5_dataset.py` | 当前实验使用的 HDF5 配对数据集。 |
| `src/learn_motion/data/mat_dataset.py`、`mat_io.py` | 为原始 MATLAB 示例数据保留的 MAT 加载路径。 |
| `src/learn_motion/factory.py` | 根据配置建立模型、数据集和 DataLoader。 |
| `src/learn_motion/losses.py`、`metrics.py`、`checkpoint.py` | 训练损失、幅值指标和权重存取。 |
| `tests/` | 小规模自动化检查；各文件职责见 `tests/README.md`。 |
| `docs/` | 正式训练说明、实验记录、笔记及 Xu 等人的 2022 年论文。 |
| `results/` | 各实验的权重、历史记录、评估表和预览图。 |

`scripts/infer_learn_img.py` 已删除；当前 HDF5 测试评估使用 `evaluate_learn_img.py`，不采用“先导出 `.mat` 再读取评估”的流程。

## 当前数据与实验约定

- HDF5 数据根目录：`/data1/gcjiang3/generated/HNRepeatability_RadialMotion_v1`。数据集通过 `splits.json` 和 `manifest.csv` 按受试者划分，并读取运动污染与无运动参考的配对图像。
- 训练集为 `sub-01` 至 `sub-07`，验证集为 `sub-08`，测试集为 `sub-09`、`sub-10`。
- 当前选用解剖第 113 至 176 层，共 64 层；Python 范围为 `[112, 176]`。每张切片包含 4 个回波，HDF5 图像维度为 `[Z, 2, 4, H, W]`。
- 正式 HDF5 配置使用 `shared_input_scale: true`，使污染输入和参考使用相同缩放系数。是否在训练损失中使用脑掩膜由 `data.use_brain_mask` 决定。
- `configs/learn_img_original.yaml` 使用 `data.format: mat`，仅用于原始 MATLAB 数据路径。不要把它当作当前 HDF5 正式实验配置。

## 已完成工作与结果快照

以下数值来自各目录的 `test-evaluation/summary.json`，均为 10 个配对案例、640 张测试切片的总体复数 MSE。`checkpoint_epoch` 为所用最佳权重对应的轮数。

| 实验配置 | 最佳轮数 | 输入 MSE | 预测 MSE | 主要设置 |
| --- | ---: | ---: | ---: | --- |
| `hn_radial_learn_img_formal.yaml` | 12 | 0.466189 | 0.358168 | 非残差、无脑掩膜、32 基础通道。 |
| `hn_radial_learn_img_residual.yaml` | 27 | 0.466189 | 0.349583 | 残差校正、无脑掩膜、32 基础通道。 |
| `hn_radial_learn_img_residual_masked.yaml` | 14 | 0.266395 | 0.195908 | 残差校正、脑掩膜损失、32 基础通道。 |
| `hn_radial_learn_img_residual_masked_wide.yaml` | 21 | 0.266395 | 0.194202 | 残差校正、脑掩膜损失、64 基础通道。 |

- 残差模式采用 `prediction = input - correction`，输出层零初始化，使初始预测为输入本身。
- 脑掩膜训练损失 `masked_mse` 按掩膜内元素数归一化；测试脚本的复数 MSE 则是先乘掩膜、再按整幅图像元素数求均值。训练损失和测试 MSE 不能直接按绝对数值比较；无掩膜与有掩膜实验的测试 MSE 口径也不同。
- 已有正式基线、残差、残差加掩膜和扩大通道数四组独立配置与结果。先前观察到预测图像过度平滑、轻污染案例可能过度校正；这些现象需要结合逐案例指标与预览图判断，不能只凭总体 MSE 下结论。
- `results/single-slice-diagnostic/` 目前有两组第 141 层、各 1000 步的单切片诊断输出。它们用于判断单样本拟合能力，不能作为泛化性能证据。
- 已将 MAT 数据集及其测试重命名为 `mat_dataset.py`、`test_mat_dataset.py`；HDF5 和 MAT 数据集仍同时受工厂支持。`README_PYTORCH.md` 已合并进根目录 `README.md`。

## 运行与维护约束

- 正确的 Python 环境为 `/data1/gcjiang3/envs/LEARN/bin/python`。在本仓库根目录运行测试：`PYTHONPATH=. /data1/gcjiang3/envs/LEARN/bin/pytest -q`。最近一次完整检查为 `11 passed`；修改后应重新运行相关测试。
- 每个新实验使用独立 YAML 和独立输出目录，不覆盖已有结果，不一次改变多个实验因素。不要自动启动长时间训练；用户通常在 tmux 中手动运行。
- 提供训练命令前，先核对当前工作目录、配置文件、目标输出目录、tmux 会话和 GPU 占用。
- `.gitignore` 忽略 `.pt`、`.pth`、`.ckpt`、`.h5` 和 `results/` 下的 `.log`。指标 JSON/CSV、`history.jsonl` 和预览图可以纳入 Git；不要提交模型权重、训练日志或原始数据。
- 工作区可能有未提交的代码、文档和结果。当前快照包括 README 合并、MAT 数据集重命名、脚本中文注释、单切片诊断输出，以及 `infer_learn_img.py` 的删除。操作前查看 `git status --short`、`git diff` 和 `git diff --cached`；不要覆盖用户修改或使用破坏性 Git 命令。
