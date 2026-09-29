# 实验编号：E2

## 实验目的

在完整保留 E1 数据增强与其余训练、验证和测试流程的基础上，将 weight decay 从仅作用于 `fc.weight` 扩展到 Backbone 中适合衰减的 Conv2d/Dense weight，验证更全面的参数正则化是否改善泛化能力。

## 对照实验与唯一主要变量

- 对照实验：E1
- 唯一主要变量：weight decay 的作用范围与参数分组
- E1：`fc.weight=0.01`；其余可训练参数为 `0`
- E2：`fc.weight=0.01`；Backbone Conv2d/Dense weight 为 `0.0001`；BatchNorm 参数、bias 及其他参数为 `0`

## E2 Weight Decay 参数分组

1. FC weight：`net.fc.weight`，`weight_decay=0.01`
2. Backbone Conv/Dense weight：通过 `network.cells_and_names()` 枚举 Cell，并从 `nn.Conv2d` / `nn.Dense` Cell 取得 `weight` Parameter；排除 `net.fc.weight`，`weight_decay=0.0001`
3. No-decay：BatchNorm gamma/beta、所有 bias 以及其他未归入上述 weight 组的参数，`weight_decay=0`

参数选择依据 Cell 类型与 Parameter 对象身份，不依赖不同 MindSpore 版本可能变化的参数名称字符串。当前 ResNet18 Backbone 实际包含 Conv2d weight，不包含额外 Dense 层。

## 固定变量

- 模型：与 E1 完全相同的自定义 ResNet18 和 ResidualBlock
- 数据划分：与 E1 相同的 70% train / 10% validation / 20% test，seed=42；正式实验前仍需固定分层 split manifest
- 训练增强：与 E1 完全相同，即 `RandomCropDecodeResize`、`RandomHorizontalFlip(prob=0.5)`、`RandomRotation(degrees=10)`、`RandomColorAdjust(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05))`
- Validation/Test 预处理：与 E1 完全相同的确定性 decode、resize、HWC2CHW 和 float32 类型转换
- Loss：`SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")`
- Optimizer：Adam
- Learning rate：0.0001
- Batch size：32
- Epochs：10
- Random seed：42
- 类别数量：5
- Validation 选择 best checkpoint、checkpoint 保存和 Test 评估流程：与 E1 完全相同
- Scheduler：none
- Dropout：none（配置中的 `dropout_ratio` 未被模型使用）
- Early stopping：none
- Class weight：none
- Pretrained model / Fine-tuning：none

## 运行命令

```powershell
python ms3.py --experiment-id E2
```

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
- Precision：待实验
- Recall：待实验
- F1：待实验
- Worst-class recall：待实验
- Train-test gap：待实验
- Checkpoint 引用/校验值：待实验
- 日志、曲线、混淆矩阵、错误案例：待实验

当前仅完成 E2 代码开发，未进行正式训练，不填写或推测任何结果。
