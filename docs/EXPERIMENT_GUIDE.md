# 实验管理与复现规范

## 1. 目的

本规范保证 E0～E6 在相同评估口径下可比较、可复查、可复现。正式结果必须能回答：改了什么、为什么改、其他条件是否固定、结果如何、结论是否由证据支持。

## 2. 实验原则

- E0 是全组唯一统一 Baseline；E0 未固定前，不开始正式消融对比。
- 每次实验尽量只修改一个主要变量。若必须同时修改多个变量，应说明原因，且不得把结果解释为某一个变量的独立贡献。
- 所有正式实验使用同一 Dataset Split，并保存不可变的 split manifest 或其校验值。
- 固定 Random Seed；需要重复实验时使用预先约定的一组 seed，并同时报告均值和波动，不挑选最好的一次。
- 数据划分必须在数据增强之前完成；测试集不得参与调参或 early stopping。
- 保证代码版本、配置、环境、数据版本、日志与 checkpoint 可以互相追溯。
- 不允许只记录“最高准确率”或只保存成功实验。
- 每个正式实验必须保留混淆矩阵、训练/验证曲线和分类别指标。
- 不得填写或推测尚未产生的实验数据；未测字段留空，并在 notes 说明原因。

## 3. 固定数据划分

正式 E0 前生成一次 train/validation/test 划分清单，后续实验只读取该清单。建议按类别分层划分，并记录：

- 数据集来源、版本和总样本数；
- 各类别原始数量；
- train/validation/test 各自样本数与类别分布；
- split seed；
- 清单文件路径和 SHA-256（或等价校验值）；
- 是否存在重复、损坏或泄漏样本。

当前代码统一读取 `splits/train.txt`、`splits/val.txt` 和 `splits/test.txt`，使用 seed 42 按类别分层生成约 70%/10%/20% 的固定划分。`splits/split_info.json` 保存各类别与各 split 数量以及 manifest SHA-256。所有正式实验必须复用这组文件；重新生成只能通过显式 `--force`，且一旦重生成，E0～后续对比实验必须全部重跑。

## 4. 每个实验必须记录的字段

| 字段 | 说明 |
|---|---|
| `experiment_id` | 唯一编号，如 E0、E1；重复运行可用 E1-R1 等 run 标识补充 |
| `model` | 模型结构、backbone、预训练权重来源/版本 |
| `dataset split` | split manifest 路径、版本或校验值 |
| `random seed` | Python、NumPy、MindSpore 及数据加载相关 seed |
| `batch size` | 训练 batch size |
| `epochs` | 最大训练轮数；若 early stopping，记录规则 |
| `optimizer` | 优化器及关键参数 |
| `learning rate` | 初始学习率，必要时含分组学习率 |
| `scheduler` | 名称、参数和更新粒度；没有则写 `none` |
| `augmentation` | 训练集增强及参数；验证/测试预处理单独说明 |
| `weight decay` | 数值及作用的参数范围 |
| `dropout` | 位置与比例；没有则写 `none` |
| `class weight` | 计算方法和具体权重；没有则写 `none` |
| `best epoch` | 按预先定义的验证指标选出的 epoch |
| `train accuracy` | best epoch 对应训练准确率 |
| `validation accuracy` | best epoch 对应验证准确率 |
| `test accuracy` | 固定 checkpoint 在测试集上的准确率 |
| `precision` | 明确 macro/micro/weighted 口径；推荐 macro 作为主表口径 |
| `recall` | 明确平均口径 |
| `f1` | 明确平均口径 |
| `worst class recall` | 各类别 recall 的最小值，并注明类别 |
| `train-test gap` | `train accuracy - test accuracy`，统一使用百分点或小数 |
| `checkpoint` | 本地路径、制品 ID/URL、文件名及校验值；权重不进 Git |
| `notes` | 异常、中断、环境差异、结论限制等 |

除上表外，还必须记录：运行日期、负责人、Git commit SHA、Python/MindSpore/驱动环境、训练设备、运行时长、配置文件路径、日志路径、混淆矩阵路径、曲线路径和错误案例路径。

## 5. 指标口径

- 主表中的 Precision、Recall、F1 必须统一平均方式；建议采用 macro，避免多数类掩盖少数类表现。
- 同时保留每一类的 Precision、Recall、F1 和 support。
- `worst_class_recall = min(per_class_recall)`，同时记录对应类别。
- `train_test_gap = train_acc - test_acc`，不得用不同 epoch 的训练指标随意拼接。
- best checkpoint 只能由 validation 指标选择；最终测试集只用于确定方案后的正式评估。
- 若重复多个 seed，应保留每次 run 的原始结果，并另行汇总 mean ± std。

## 6. 实验执行流程

1. 创建 Experiment Issue，明确目的、主要变量、固定变量、负责人和预期输出。
2. 从 `dev` 创建 `feature/experiment-<id>-<slug>` 分支。
3. 复制上一实验配置，修改目标变量并审查配置差异。
4. 记录 Git commit SHA、环境和数据 split 标识后开始训练。
5. 保存日志与 checkpoint；checkpoint 文件不提交 Git。
6. 以 validation 指标选择 best epoch/checkpoint，然后只对选定 checkpoint 做正式测试。
7. 生成混淆矩阵、训练/验证曲线、分类报告和错误案例。
8. 更新 `experiments/<id>_<slug>.md` 与 `results/metrics.csv`。
9. 提交 PR，由对应负责人检查变量隔离、指标口径和复现信息。

公共评估模块对完整 test manifest 仅遍历一次，统一采用固定类别顺序 `daisy`、`dandelion`、`roses`、`sunflowers`、`tulips`。无预测样本或无真实样本导致分母为零时，对应 precision、recall 或 F1 记为 0，不产生 NaN。Test 只能在 validation 选出 best checkpoint 后执行。

每个 epoch 结束后，使用 train manifest 的确定性预处理副本记录 train loss/accuracy，并使用 validation manifest 记录 validation loss/accuracy；这些只读评估不参与梯度更新。`best_epoch_train_accuracy` 必须取自 validation 选中 best checkpoint 的同一个 epoch，用于计算 `train_test_gap`。

## 7. 文件命名建议

```text
configs/E1_augmentation.yaml
experiments/E1_augmentation.md
results/confusion_matrix/E1_seed42.png
results/curves/E1_seed42.png
results/error_cases/E1_seed42/
checkpoints/E1_seed42_best.ckpt
logs/E1_seed42.log
```

经复核并用于分析或课程报告的最终混淆矩阵、曲线和代表性错例可以提交 Git。批量中间图放入 `results/tmp/`，原始训练日志放入 `results/logs/`，两者由 `.gitignore` 排除；不要一次提交所有中间产物。

## 8. E0～E6 的变量控制

| 实验 | 相对比较对象 | 预期主要变量 |
|---|---|---|
| E0 | 无 | 固定 Baseline、split 和评估口径 |
| E1 | E0 | 仅 Data Augmentation |
| E2 | E1 | 仅扩大 Weight Decay 作用范围（保留 FC weight=0.01，新增 Backbone weight decay） |
| E3 | E2 | 仅 Cosine LR Scheduler（Adam 不变，逐 step 衰减） |
| E4 | 独立迁移学习基线 | ImageNet pretrained ResNet18 + frozen Backbone + 新 5 类分类头；不继承 E1～E3 |
| E5 | E4 | 仅 partial fine-tuning `layer4`；其余 Backbone 保持冻结，所有 BatchNorm running statistics 固定 |
| E6 | E5 | 仅 Class-weighted Loss；权重只能由 train manifest 按 normalized inverse frequency 自动计算 |

实验关系必须写成两条链：Baseline improvement track 为 E0→E1→E2→E3；Transfer learning track 为 E4→E5→E6，其中 E6 是 conditional/candidate imbalance experiment。E4 使用不同模型实现、ImageNet 初始化及其配套输入预处理，不能简单视为在 E3 上只增加一个训练参数，也不能把它与 E0～E3 的差异解释为单一变量贡献。E5 以 E4 为直接对照，唯一核心变化是只解冻 `layer4` 做 partial fine-tuning；不得同时引入额外增强或 scheduler。E6 必须完整继承 E5，唯一主要变化是 class-weighted loss；权重只能由 `splits/train.txt` 按 `w_c = N / (K * n_c)` 自动统计，不得参考 validation/test 数量或表现。E4/E5/E6 必须复用同一 split manifest、统一 evaluation、Validation Accuracy best-checkpoint 选择和最终 test 流程。E6 是否纳入核心改进链，必须等待正式诊断和实验，不得预设有效。
