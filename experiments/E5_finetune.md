# 实验编号：E5

## 实验目的

在 E4 frozen-backbone 迁移学习基线之上，只解冻 MindCV ResNet18 的最后一个高层残差阶段 `layer4`，验证有限的 task-specific feature adaptation 是否优于完全冻结 Backbone。

E5 是 partial fine-tuning baseline，不是 full fine-tuning。选择 `layer4` 是因为当前数据集只有 3670 张图片：低层和中层特征更通用，而最高层语义特征更适合针对五类花卉进行有限适配；只解冻最后阶段也可降低过拟合和破坏预训练特征的风险。

## 对照实验与唯一核心变化

- 对照实验：E4。
- E4：全部 Backbone frozen，仅 classifier trainable。
- E5：`layer4` + classifier trainable，其余 Backbone frozen。
- 唯一核心变化：启用 `fine_tune=True`、`fine_tune_scope=layer4`，并为新解冻的预训练参数使用较小固定学习率。
- 不继承 E1 extra augmentation、E2 实验开关或 E3 cosine scheduler。

## Pretrained Model

- 模型：MindCV 0.3.0 `resnet18`。
- 权重：与 E4 完全相同的官方 ImageNet-1K `resnet18-1e65cd21.ckpt`。
- 新分类头：与 E4 完全相同的 512→5 `classifier`，`TruncatedNormal(0.02)` 初始化。
- 类别顺序：0 daisy、1 dandelion、2 roses、3 sunflowers、4 tulips。

## Frozen / Trainable Layers

Frozen：

- `conv1`
- `bn1`
- `layer1`
- `layer2`
- `layer3`

Trainable：

- `layer4` Conv weight（以及实际存在的 Conv bias）
- `layer4` BatchNorm gamma/beta
- `classifier.weight`
- `classifier.bias`

参数通过实际 Cell 结构与 Parameter 对象身份选择，不依赖参数名字符串匹配。程序断言 trainable parameter set 必须严格等于 `layer4` 选定参数与 classifier 参数的并集。

## BatchNorm 策略

- Frozen 部分 BatchNorm：gamma/beta frozen，running mean/variance frozen。
- `layer4` BatchNorm：gamma/beta trainable；running mean/variance frozen。
- 所有 BatchNorm 均设置 `use_batch_statistics=False`，使用 ImageNet checkpoint 保存的统计量，不在当前小数据集上重新估计。

该保守策略避免 batch size 32 和较小训练集导致 running statistics 不稳定，同时允许 `layer4` 的 affine 参数适应花卉分类任务。

## Optimizer 参数组

Optimizer 继续使用 Adam，不使用 scheduler。冻结参数不进入 optimizer。

1. `layer4` Conv/Dense weight：LR `0.00001`，weight decay `0.0001`
2. `layer4` BatchNorm gamma/beta 与 bias：LR `0.00001`，weight decay `0`
3. `classifier.weight`：LR `0.0001`，weight decay `0.01`
4. `classifier.bias`：LR `0.0001`，weight decay `0`

`layer4` 使用更小 LR，因为它来自预训练权重，只需要小幅适配；classifier 是新初始化参数，继续使用 E4 的 LR。`layer4` 的 `0.0001` weight decay 是迁移学习链中的轻量 fine-tuning 正则化配置，不解释为 E2 的实验贡献。

## Preprocessing

与 E4 完全一致：

- Train：`RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333))` → ImageNet Normalize → HWC2CHW → float32。
- Train-eval：Decode → Resize(256) → CenterCrop(224) → ImageNet Normalize → HWC2CHW → float32。
- Validation：与 Train-eval 相同，完全确定性。
- Test：与 Train-eval 相同，完全确定性。
- 不增加 HorizontalFlip、RandomRotation 或 RandomColorAdjust。

## 固定 Split

E5 不重新划分数据，继续读取并验证与 E4 完全相同的 manifest：

- `train.txt` SHA-256：`e3746f3898a0bf61e9082dad57e3b7da60cd0ad5d37cfef8df6c3f86a48e7aee`
- `val.txt` SHA-256：`cd1d81e02b8b4a8089258f48a37d692805d3cdf642ea54350964493db3daec95`
- `test.txt` SHA-256：`560d77f600d243e2ea1f084bebd49d8b1dbb7ce65ee2b2fa1ecce82adc4136b3`

## Evaluation 与 Checkpoint

- 继续使用公共 `evaluation.py`，不修改指标公式。
- 每个 epoch 记录确定性 train-eval 与 validation loss/accuracy。
- 只由 validation accuracy 选择 `checkpoints/E5/E5_best.ckpt`。
- 训练结束后加载 best checkpoint，再对完整 test manifest 评估一次。
- 输出 Accuracy、Macro Precision/Recall/F1、per-class metrics、worst-class recall、confusion matrix、train-test gap、best epoch、best validation accuracy 和 best-epoch train accuracy。

## 固定配置

- Loss：`SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')`
- Optimizer：Adam
- Batch size：32
- Epochs：10
- Random seed：42
- Scheduler：none
- Extra augmentation：false
- Full-backbone fine-tuning：false

## 运行与审计

```powershell
python ms3.py --experiment-id E5
```

启动时打印总参数量、trainable/frozen 参数量、`layer4` 与 classifier 的参数名、参数量、LR 和 weight decay 策略，并断言 optimizer 只覆盖 `layer4 + classifier`。

正式训练前还需在 MindSpore 环境执行非正式 1-batch smoke audit：确认 `layer3` 参数不变，`layer4` Conv 参数与 `classifier.weight` 均发生更新。该检查只验证冻结逻辑，不属于实验结果。

## 待实验结果

- Git commit SHA：待实验
- Python / MindSpore / MindCV / CUDA / 驱动：待实验
- 训练设备与运行时长：待实验
- Pretrained checkpoint 来源与校验信息：待实验
- 1-batch freeze/update smoke audit：待实验
- Best epoch：待实验
- Train accuracy：待实验
- Validation accuracy：待实验
- Test accuracy：待实验
- Macro Precision / Recall / F1：待实验
- Per-class Precision / Recall / F1 / Support：待实验
- Worst-class recall：待实验
- Train-test gap：待实验
- Checkpoint、日志、曲线、混淆矩阵和错误案例：待实验

当前仅完成 E5 代码开发，不进行正式训练，不填写或推测任何结果。
