# 实验编号：E4

## 实验目的

建立一条独立的迁移学习基线：使用 ImageNet 预训练视觉特征，冻结 ResNet18 Backbone，只训练新的 5 类分类头，并通过固定 validation 选择 best checkpoint 后在 test 上评估。

核心研究问题：在当前 3670 张五类花卉数据上，ImageNet pretrained frozen Backbone + 新分类头能够达到怎样的迁移学习效果？

## 实验类型与对照关系

- 实验类型：迁移学习基线。
- Baseline improvement track：E0 → E1 → E2 → E3。
- Transfer learning track：E4 → E5。
- E4 不继承 E1 extra augmentation、E2 Backbone weight decay 或 E3 cosine scheduler。
- E4 更换模型实现与初始化，并采用预训练权重配套 preprocessing，因此不是 E3 上的单变量增量实验，也不是与 E0～E3 的严格单变量消融。

## Pretrained Model 与权重

- 模型：MindCV 0.3.0 `resnet18`。
- 预训练任务：ImageNet-1K，原分类头为 1000 类。
- API 来源：[MindCV 官方仓库](https://github.com/mindspore-lab/mindcv) 的 `create_model`、ResNet 模型注册与 ImageNet transforms。
- 官方权重注册：MindCV `resnet.py` 中的 `resnet18-1e65cd21.ckpt`，来源域名为 `download.mindspore.cn`。
- 官方权重地址：<https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt>。
- 默认加载：`mindcv.create_model('resnet18', pretrained=True, num_classes=1000)`，需要首次联网下载或命中 MindCV 本地缓存。
- 离线加载：`--pretrained-checkpoint` 指向同一个官方 1000 类 checkpoint；使用 `strict_load=True`，任何缺失或多余参数均终止运行。
- 禁止 fallback：权重下载、读取或严格匹配失败时直接报错，不退回随机初始化。

## 冻结与可训练层

- Backbone：`conv1`、`bn1`、`layer1`～`layer4` 及全局池化特征提取全部冻结。
- BatchNorm：除参数冻结外，强制使用已保存的 running mean/variance，不在 E4 训练中更新统计量。
- 新分类头：`classifier` 从 512 维特征映射到 5 类，使用 `TruncatedNormal(0.02)` 初始化。
- 唯一可训练参数：`classifier.weight` 和 `classifier.bias`。
- 启动时打印总参数量、可训练参数量、冻结参数量、可训练参数名和 classifier 参数名；若发现任何 Backbone 参数可训练则终止。

## 类别与数据划分

- 类别顺序固定为：0 daisy、1 dandelion、2 roses、3 sunflowers、4 tulips。
- 不重新 split；继续读取并验证现有 manifest。
- `train.txt` SHA-256：`e3746f3898a0bf61e9082dad57e3b7da60cd0ad5d37cfef8df6c3f86a48e7aee`
- `val.txt` SHA-256：`cd1d81e02b8b4a8089258f48a37d692805d3cdf642ea54350964493db3daec95`
- `test.txt` SHA-256：`560d77f600d243e2ea1f084bebd49d8b1dbb7ce65ee2b2fa1ecce82adc4136b3`

## Input Preprocessing

预处理依据 MindCV 0.3.0 ImageNet transforms：输入尺寸 224×224，mean 为 `[0.485, 0.456, 0.406] × 255`，std 为 `[0.229, 0.224, 0.225] × 255`，通道顺序为 RGB 后转 CHW。

- Train：`RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333))` → ImageNet Normalize → HWC2CHW → float32。不开启 E1 的 flip、rotation 或 color adjustment。
- Train-eval：Decode → Resize(256) → CenterCrop(224) → ImageNet Normalize → HWC2CHW → float32。
- Validation：与 Train-eval 相同，完全确定性。
- Test：与 Train-eval 相同，完全确定性。

输入由 E0～E3 的 100×100、无明确 normalization 改为 224×224 ImageNet normalization，属于预训练模型兼容要求，也是 E4 与 E0～E3 不能视为严格单变量比较的原因之一。

## Optimizer 与固定配置

- Loss：`SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')`。
- Optimizer：Adam。
- Learning rate：固定 `0.0001`；scheduler 为 none。
- Batch size：32。
- Epochs：10。
- Random seed：42。
- `classifier.weight` weight decay：0.01。
- `classifier.bias` weight decay：0。
- Backbone：冻结，不进入 optimizer，不应用 E2 Backbone weight decay。
- Extra augmentation：false。
- Fine-tuning：false。

## Evaluation 与 Checkpoint

- 继续使用公共 `evaluation.py`，不另写指标公式。
- 每个 epoch 使用确定性 train-eval 和 validation 记录 loss/accuracy。
- 只以 validation accuracy 选择 `checkpoints/E4/E4_best.ckpt`。
- 训练结束后加载 best checkpoint，并仅在此时评估完整 test manifest。
- 输出 Accuracy、Macro Precision/Recall/F1、per-class metrics、worst-class recall、confusion matrix、train-test gap、best epoch、best validation accuracy 和 best-epoch train accuracy。

## 运行命令

联网或已有 MindCV 缓存：

```powershell
python ms3.py --experiment-id E4
```

离线官方 checkpoint：

```powershell
python ms3.py --experiment-id E4 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
```

## 待实验结果

- Git commit SHA：待实验
- Python / MindSpore / MindCV / CUDA / 驱动：待实验
- 训练设备与运行时长：待实验
- 权重下载或离线 checkpoint 校验信息：待实验
- Best epoch：待实验
- Train accuracy：待实验
- Validation accuracy：待实验
- Test accuracy：待实验
- Macro Precision / Recall / F1：待实验
- Per-class Precision / Recall / F1 / Support：待实验
- Worst-class recall：待实验
- Train-test gap：待实验
- Checkpoint、日志、曲线、混淆矩阵和错误案例：待实验

当前仅完成 E4 静态开发，不填写或推测任何实验结果。E5 fine-tuning 尚未实现；后续应基于同一模型构建接口调整冻结策略并单独开展实验。
