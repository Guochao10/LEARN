# 测试目录

本目录用于检查 LEARN 的模型, 数据读取, checkpoint 和脚本入口. 正常运行训练或测试评估时不会自动执行这些测试. 测试使用临时生成的小数据, 不读取正式实验数据, 也不写入现有的 `results/` 实验目录.

| 文件 | 用途 |
| --- | --- |
| `test_learn_img_model.py` | 检查模型输出形状, 反向传播, 回波维度保留, 残差模式初始恒等输出和非法图像尺寸处理. |
| `test_h5_dataset.py` | 检查当前 HDF5 数据路径的配对清单, 切片选择, 脑掩膜, 归一化, 共享缩放系数和 DataLoader 输出. |
| `test_formal_training.py` | 用临时 HDF5 数据在 CPU 上运行一轮训练和一次续训, 检查权重, 训练历史, 验证预览图和调度器状态. |
| `test_evaluate_learn_img.py` | 用临时测试数据检查推理与评估一次完成, 报告和预览图正确生成, 且不保存 `.mat` 或 `.h5` 中间预测文件. |
| `test_overfit_one_h5_slice.py` | 检查单切片诊断使用的掩膜 MSE 和幅值梯度误差. 不运行完整的单切片过拟合训练. |
| `test_checkpoint.py` | 检查模型权重保存和加载后预测一致, 并核对 checkpoint 中的轮数和最佳指标. |
| `test_complex_conversion.py` | 检查复数数组与实部/虚部双通道张量之间的转换能往返还原. |
| `test_mat_dataset.py` | 检查旧 MATLAB `.mat` 数据加载器的切片, 回波和空间轴顺序. 当前 HDF5 正式实验不使用此加载器, 但配置工厂仍支持 MAT 数据路径. |

在 `LEARN` 根目录运行全部测试:

```bash
cd /data/home/gcjiang3/qsm/LEARN
PYTHONPATH=. /data1/gcjiang3/envs/LEARN/bin/pytest -q
```
