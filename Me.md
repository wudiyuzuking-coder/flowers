先把 Baseline 跑通并记录结果
先不要急着改模型。拿到 Train/Val Loss、Train/Val Accuracy、Test Accuracy，确认原始模型到底什么水平。

做 Baseline 问题诊断
重点看两件事：  
- 有没有过拟合
- 哪些类别识别差
同时让评估同学产出混淆矩阵、Precision / Recall / F1、Worst-class Recall。老师要求的重点就是先用证据定位问题，而不是直接说“准确率低”。 花卉图像识别模型改进研讨说明

根据诊断结果决定改进路线
不要一上来所有技巧全加。优先按类似下面的路线：
E0 Baseline → E1 数据增强 → E2 Weight Decay → E3 LR Scheduler → E4 预训练模型 → E5 Fine-tune → E6 类别加权（需要时）

柚子亲自负责核心模型部分
重点做迁移学习和 Fine-tuning，比如加载 ImageNet 预训练 backbone、替换分类头、冻结/解冻部分层。这部分最能体现核心技术贡献。 花卉图像识别模型改进研讨说明

设计消融实验
每次尽量只改一个主要变量，其余设置保持一致。你需要决定每个实验“为什么做、改什么、看什么指标”。老师明确强调消融实验，而且评分占比最高。 花卉图像识别模型改进研讨说明

把实验交给 5070Ti 同学统一跑
你负责给出配置和实验编号，对方负责批量训练、测试、保存 checkpoint 和日志。这样所有实验环境统一，数据才好比较。

盯住最终结果分析
不只是看 Accuracy，还要比较：
- Test Accuracy
- Precision / Recall / F1
- Worst-class Recall
- Train-Test Gap
- 混淆矩阵变化

最后负责技术结论和答辩主线
最终你要能讲清楚三句话：
原模型有什么问题 → 为什么这么改 → 实验是否证明改进有效。
这就是整个作业的核心。
