# 实验编号：E6

## 实验定位与目的

E6 是预注册的候选类别均衡实验。在 E5 partial fine-tuning 基础上，只把训练目标改为 class-weighted softmax cross entropy，观察类别均衡处理对 per-class recall、worst-class recall、macro recall 和 macro F1 的影响。

目前没有正式实验结果证明类别不平衡已构成性能瓶颈。E6 不预设 Overall Accuracy 或任何类别指标一定提高；少数类 recall 提高而 Overall Accuracy 不变或下降，也只能由正式结果判断。最终是否将 E6 纳入报告的核心改进链，应由正式诊断和对照实验决定。

## 对照实验与唯一主要变化

- 对照实验：E5。
- E5：普通 `SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')`。
- E6：逐样本 sparse softmax CE 乘以真实类别对应权重后取 batch mean。
- 唯一主要变化：Class-weighted classification loss。
- 模型 forward、fine-tune 范围、optimizer、preprocessing、split、evaluation、checkpoint 选择和 test 流程均保持 E5 不变。

## Class Weight 数据来源与公式

- 唯一数据来源：`splits/train.txt`。
- 禁止使用 validation/test 的类别数量或性能决定权重。
- 类别/标签顺序：0 daisy、1 dandelion、2 roses、3 sunflowers、4 tulips。
- 公式：`w_c = N / (K * n_c)`，其中 `N` 为 train 样本数，`K=5`，`n_c` 为类别 c 的 train 样本数。
- counts 与 weights 均在启动时从 train manifest 自动统计并打印；weights 使用 float32。
- 程序校验每类 count 大于 0，weight 有限且大于 0，并校验 counts 与已验证的 train split 信息一致。
- Train class counts：运行时自动统计。
- Class weights：运行时自动计算。

MindSpore 2.2 的 [`nn.SoftmaxCrossEntropyWithLogits`](https://www.mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html) 官方接口只有 `sparse` 和 `reduction`，不直接提供多分类 class-weight 参数。因此 E6 使用最小自定义 `WeightedSoftmaxCrossEntropy`：先以 `reduction='none'` 得到 `[N]` 的逐样本 CE，再按 sparse label 通过 [`ops.gather`](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.gather.html) 取得 `[N]` 的 class weight，计算 `mean(weight[label] * CE)`。权重不乘到 logits 上。

## E5 固定配置

- Pretrained model：MindCV 0.3.0 ImageNet ResNet18，与 E5 相同。
- Fine-tune scope：仅 `layer4 + classifier`；`conv1`、`bn1`、`layer1`～`layer3` frozen。
- BatchNorm：所有 running statistics frozen；`layer4` BN gamma/beta trainable。
- Layer4 Conv/Dense weight：LR `0.00001`，weight decay `0.0001`。
- Layer4 BN gamma/beta 与 bias：LR `0.00001`，weight decay `0`。
- Classifier weight：LR `0.0001`，weight decay `0.01`。
- Classifier bias：LR `0.0001`，weight decay `0`。
- Optimizer：Adam；scheduler：none。
- Batch size：32；epochs：10；random seed：42。
- Extra augmentation：false。

## Preprocessing 与 Split

- Train：`RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333))` → ImageNet Normalize → HWC2CHW → float32。
- Train-eval / Validation / Test：Decode → Resize(256) → CenterCrop(224) → ImageNet Normalize → HWC2CHW → float32。
- 不增加 Flip、Rotation、ColorAdjust、Mixup 或 CutMix。
- 继续使用相同的 `splits/train.txt`、`splits/val.txt`、`splits/test.txt`，不得重新 split。
- Class weight 只读取 `train.txt`。

## Evaluation 与 Checkpoint

- `evaluation.py` 和全部指标公式保持不变。
- 每个 epoch 仍以 Validation Accuracy 选择 `checkpoints/E6/E6_best.ckpt`，不改用 Macro F1。
- 训练结束后加载 best validation checkpoint，再对完整 test manifest 评估一次。
- 继续输出 Accuracy、Macro Precision/Recall/F1、per-class metrics、worst-class recall、confusion matrix、train-test gap 和 best epoch。
- 公共 evaluation 中报告的 loss 仍为未加权 sparse CE，以维持 E5/E6 相同的评估口径；class weight 只改变训练目标。

## 运行命令

```powershell
python ms3.py --experiment-id E6
```

离线官方 checkpoint：

```powershell
python ms3.py --experiment-id E6 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
```

## 待实验结果

- Git commit SHA：待实验
- Python / MindSpore / MindCV / CUDA / 驱动：待实验
- 训练设备与运行时长：待实验
- Pretrained checkpoint 来源与校验信息：待实验
- Weighted loss 运行级 smoke test：待实验
- Best epoch：待实验
- Train accuracy：待实验
- Validation accuracy：待实验
- Test accuracy：待实验
- Macro Precision / Recall / F1：待实验
- Per-class Precision / Recall / F1 / Support：待实验
- Worst-class recall：待实验
- Train-test gap：待实验
- Checkpoint、日志、曲线、混淆矩阵和错误案例：待实验

当前只完成 E6 代码、静态验证和候选实验文档；不进行正式训练，不填写或推测任何实验结果。
