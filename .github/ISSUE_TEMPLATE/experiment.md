---
name: 实验任务
about: 规划、执行并记录一次可复现实验
title: "[Experiment] E?: "
labels: experiment
assignees: ""
---

## 实验编号

<!-- 例如 E1；重复运行请注明 run/seed。 -->

## 实验目的

<!-- 要验证的假设和成功/失败判断标准。 -->

## 修改变量

<!-- 原则上只写一个主要变量，并写明具体取值。 -->

## 固定变量

- 对照实验：
- Dataset split / manifest：
- Random seed：
- Model：
- Batch size：
- Epochs：
- Optimizer：
- Learning rate：
- 其他：

## 负责人

- 实现：
- 训练：
- 评估：
- 复现审核：

## 训练设备

<!-- GPU/CPU 型号、Python、MindSpore、CUDA/驱动等。 -->

## 输出内容

- [ ] 实验配置
- [ ] 训练日志
- [ ] Best checkpoint 引用与校验值
- [ ] Train/validation 曲线
- [ ] 混淆矩阵
- [ ] 分类别指标与 worst-class recall
- [ ] 错误案例
- [ ] 实验记录 Markdown
- [ ] `results/metrics.csv`

## 实验结果

<!-- 只填写真实结果；未运行时留空。注明指标平均方式。 -->

## 是否完成

- [ ] 训练完成
- [ ] 正式测试完成
- [ ] 结果复核完成
- [ ] 可由他人复现

