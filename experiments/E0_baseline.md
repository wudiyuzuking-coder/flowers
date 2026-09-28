# 实验编号：E0

## 实验目的

建立全组统一 Baseline，固定数据划分、随机种子、训练配置与评估口径，为 E1～E6 提供可比较的基准。

## 与上一实验相比修改了什么

- 对照实验：无
- 唯一主要变量：不适用（建立基准）
- 保持不变的变量：与 E1 使用相同模型、损失、优化器、学习率、batch size、seed、FC weight decay、epoch 数、类别数量、数据划分和评估流程

## 配置

- Git commit SHA：待实验
- 启动命令：`python ms3.py --experiment-id E0`
- 配置方式：`ms3.py` 中共享配置 + `experiment_id=E0`
- 数据划分清单/校验值：待创建；当前代码以 seed=42 划分 70% train / 10% validation / 20% test
- Random seed：42
- 模型：自定义 ResNet18
- Batch size：32
- Epochs：10
- Optimizer：Adam
- Learning rate：0.0001
- Scheduler：none
- Augmentation：训练集仅 `RandomCropDecodeResize`；validation/test 使用确定性 decode + resize + 类型转换 + HWC2CHW
- Weight decay：仅 FC weight，0.01
- Dropout：none（配置中的 `dropout_ratio` 未被模型使用）
- Class weight：none
- Python / MindSpore / 驱动：待确认
- 训练设备：待确认
- 负责人：待分配

## 实验结果

- Best epoch：待实验
- Train accuracy：待实验
- Validation accuracy：待实验
- Test accuracy：待实验
- Precision（注明平均方式）：待实验
- Recall（注明平均方式）：待实验
- F1（注明平均方式）：待实验
- Worst-class recall（含类别）：待实验
- Train-test gap：待实验
- Checkpoint 引用/校验值：待实验
- 日志：待实验

## 混淆矩阵

未生成。

## 训练曲线

未生成。

## 错误案例

未整理。

## 结论

待实验，不填写推测结论。

## 是否进入下一阶段

- 决策：待决定
- 理由：需先完成 E0 复现、评估与问题诊断。
