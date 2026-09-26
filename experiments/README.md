# 实验记录

本目录保存人工可读的实验卡片。每次正式实验使用一个 Markdown 文件，文件名采用 `<experiment_id>_<slug>.md`，例如 `E1_augmentation.md`。重复运行可在正文中建立 run 表，或在编号后增加 `R1`、`R2`，但 `results/metrics.csv` 中必须能唯一追溯每次记录。

不要填写不存在的数据。未完成字段保留“待实验/未生成”，并在 notes 中说明原因。

## 单次实验模板

```markdown
# 实验编号：E?

## 实验目的

<!-- 本实验要验证的假设。 -->

## 与上一实验相比修改了什么

- 对照实验：
- 唯一主要变量：
- 保持不变的变量：

## 配置

- Git commit SHA：
- 配置文件：
- 数据划分清单/校验值：
- Random seed：
- 模型：
- Batch size：
- Epochs：
- Optimizer：
- Learning rate：
- Scheduler：
- Augmentation：
- Weight decay：
- Dropout：
- Class weight：
- Python / MindSpore / 驱动：
- 训练设备：
- 负责人：

## 实验结果

- Best epoch：
- Train accuracy：
- Validation accuracy：
- Test accuracy：
- Precision（注明平均方式）：
- Recall（注明平均方式）：
- F1（注明平均方式）：
- Worst-class recall（含类别）：
- Train-test gap：
- Checkpoint 引用/校验值：
- 日志：

## 混淆矩阵

<!-- 文件路径、图表或“未生成”。 -->

## 训练曲线

<!-- Train/validation loss 与 accuracy；文件路径、图表或“未生成”。 -->

## 错误案例

<!-- 典型错例路径，以及真实类别、预测类别、置信度和分析。 -->

## 结论

<!-- 结论必须由指标与图表支持，同时记录异常和局限。 -->

## 是否进入下一阶段

- 决策：待决定 / 是 / 否 / 需要重复实验
- 理由：
```

字段口径和完整执行流程见 `docs/EXPERIMENT_GUIDE.md`。

