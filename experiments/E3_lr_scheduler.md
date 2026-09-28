# 实验编号：E3

## 实验目的

在完整继承 E2 的模型、数据增强、weight decay 分组和评估流程的基础上，将固定学习率替换为预先确定的 cosine learning-rate schedule，验证训练后期逐渐减小更新步长是否改善泛化能力。

## 对照实验与唯一主要变量

- 对照实验：E2
- 唯一主要变量：Learning Rate Strategy
- E2：Adam 使用固定 learning rate `0.0001`
- E3：Adam 使用从 `0.0001` 平滑衰减到 `0.000001` 的逐 step cosine schedule
- Scheduler 不读取 validation 或 test 指标，不进行自适应调参

## Scheduler 配置

- 名称：Per-step Cosine Learning Rate Decay
- Initial learning rate：`0.0001`
- Final/min learning rate：`0.000001`
- 粒度：optimizer step
- Steps per epoch：运行时由 `de_train.get_dataset_size()` 获取
- Total decay steps：`steps_per_epoch * epoch_size`
- MindSpore 机制：将覆盖全部训练 step 的 cosine 数组转换为 1-D `mindspore.Tensor`，作为 `nn.Adam(..., learning_rate=...)` 的动态学习率输入
- 曲线：`results/curves/E3_lr_curve.png`，直接根据传给 Adam 的 schedule 数组生成，属于配置理论值，不是 callback 读取值

## 固定变量

- 模型：与 E2 完全相同的自定义 ResNet18 和 ResidualBlock
- 数据划分：与 E2 相同的 70% train / 10% validation / 20% test，seed=42；正式实验前仍需固定分层 split manifest
- 训练增强：与 E2 完全相同，即 `RandomCropDecodeResize`、`RandomHorizontalFlip(prob=0.5)`、`RandomRotation(degrees=10)`、`RandomColorAdjust(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05))`
- Validation/Test 预处理：与 E2 完全相同的确定性 decode、resize、HWC2CHW 和 float32 类型转换
- Loss：`SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")`
- Optimizer：Adam
- Batch size：32
- Epochs：10
- Random seed：42
- 类别数量：5
- FC weight decay：0.01
- Backbone Conv/Dense weight decay：0.0001
- BatchNorm / bias weight decay：0
- Validation 选择 best checkpoint、checkpoint 保存和 Test 评估流程：与 E2 完全相同
- Dropout：none（配置中的 `dropout_ratio` 未被模型使用）
- Early stopping：none
- Class weight：none
- Pretrained model / Fine-tuning：none

## 运行命令

```powershell
python ms3.py --experiment-id E3
```

## 待实验字段

- Git commit SHA：待实验
- 实际 steps per epoch / total decay steps：待实验环境确认
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
- 日志、训练曲线、混淆矩阵、错误案例：待实验

当前仅完成 E3 代码开发，未进行正式训练，不填写或推测任何结果。
