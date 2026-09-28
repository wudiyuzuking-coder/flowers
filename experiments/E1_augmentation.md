# 实验编号：E1

## 实验目的

在保持模型、优化器、学习率、batch size、seed、FC weight decay、epoch 数和类别数量不变的前提下，验证保守数据增强相对 E0 Baseline 的作用。

## E0 和 E1 的变量差异

- 对照实验：E0
- 唯一主要变量：训练集是否启用额外数据增强
- E0：`RandomCropDecodeResize`
- E1：`RandomCropDecodeResize` + `RandomHorizontalFlip(prob=0.5)` + `RandomRotation(degrees=10)` + `RandomColorAdjust(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05))`
- `RandomColorAdjust` 是 MindSpore 中用于实现 ColorJitter 的对应算子。
- validation 和 test 均只使用确定性 decode、resize、类型转换和 HWC2CHW，不使用随机增强。

## 配置

- 启动命令：`python ms3.py --experiment-id E1`
- 数据划分：70% train / 10% validation / 20% test；seed=42；正式实验前待生成不可变、按类别分层的 split manifest 及校验值
- Random seed：42
- 模型：自定义 ResNet18
- Batch size：32
- Epochs：10
- Optimizer：Adam
- Learning rate：0.0001
- Scheduler：none
- Weight decay：仅 FC weight，0.01
- Dropout：none（配置中的 `dropout_ratio` 未被模型使用）
- Class weight：none
- 类别数量：5
- 模型选择：每个 epoch 在 validation 上评估，以最高 validation accuracy 保存 best checkpoint
- Test 使用：训练结束后仅评估 validation 选出的 best checkpoint

## 待实验字段

- Git commit SHA：待实验
- 数据划分清单/校验值：待创建
- Python / MindSpore / CUDA / 驱动：待实验
- 训练设备：待实验
- 负责人：待分配
- Best epoch：待实验
- Train accuracy：待实验
- Validation accuracy：待实验
- Test accuracy：待实验
- Precision / Recall / F1：待实验
- Worst-class recall：待实验
- Train-test gap：待实验
- Checkpoint 引用/校验值：待实验
- 日志、曲线、混淆矩阵、错误案例：待实验

当前仅完成 E1 代码开发，未进行正式训练，不填写或推测任何结果。
