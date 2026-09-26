# 花卉图像识别模型改进

这是一个 5 人协作的课程项目。项目在现有花卉分类代码基础上，通过统一的 Baseline、问题诊断、单变量改进、消融实验和错误分析，形成可复现的模型改进结论与课程报告。

> 当前状态：仓库中保留了原始 `ms3.py`（训练）与 `ms3_val.py`（加载 checkpoint 后预测/评估）。本次仅补充项目管理结构和规范，没有修改模型实现或伪造实验结果。

## 项目目标

- 复现并固定统一的 E0 Baseline。
- 使用训练/验证曲线、混淆矩阵和分类别指标诊断问题。
- 逐项验证数据增强、正则化、学习率策略、迁移学习和类别不平衡处理。
- 使用相同数据划分、随机种子与评估口径完成可比实验。
- 保存配置、日志、曲线、混淆矩阵、checkpoint 引用和结论，支持他人复现。

## 当前技术栈

- Python
- MindSpore（`GRAPH_MODE`；现有脚本当前指定 `CPU`）
- NumPy
- Matplotlib
- EasyDict
- 数据集：5 类花卉图片（daisy、dandelion、roses、sunflowers、tulips）
- 当前模型代码：自定义 ResNet18 风格网络

依赖来自现有源码的实际 import。版本尚未锁定；完成团队环境验证后，应把可复现版本写入锁定文件或在实验记录中记录 Python、MindSpore、CUDA/驱动和设备版本。

## 目录结构

```text
.
├── .github/                     # Issue 与 Pull Request 模板
├── configs/                     # 实验配置约定（当前代码尚未接入配置加载）
├── docs/
│   ├── EXPERIMENT_GUIDE.md      # 实验记录和复现规范
│   └── GITHUB_WORKFLOW.md       # 分支、Commit、PR 协作规范
├── experiments/
│   ├── README.md                # 单次实验记录模板与命名规则
│   └── E0_baseline.md           # E0 空白记录
├── results/
│   ├── metrics.csv              # 实验总表（仅有表头）
│   ├── confusion_matrix/        # 经复核的混淆矩阵
│   ├── curves/                  # 经复核的训练/验证曲线
│   └── error_cases/             # 代表性错例可视化
├── flower_photos/               # 当前本地原始数据集，已在 .gitignore 中忽略
├── ms3.py                       # 现有训练脚本
├── ms3_val.py                   # 现有预测/评估脚本
├── Instruction.md               # 原项目实施方案
└── requirements.txt
```

没有移动原有源码或数据，以免破坏现有运行方式。

## 环境安装

建议使用独立虚拟环境。Windows PowerShell 示例：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

MindSpore 的 CPU/GPU 安装包与 Python、操作系统及加速环境存在兼容要求。正式训练机应先确定 RTX 5070 Ti 对应的可用环境，再锁定版本；不要只依赖未锁版本的 `requirements.txt` 作为复现依据。

## 启动与训练

现有训练入口：

```powershell
python ms3.py
```

现有加载 checkpoint 并预测/评估的入口：

```powershell
python ms3_val.py
```

运行前必须注意：

1. 两个脚本中的 `cfg.data_path` 仍是原开发机的绝对路径，并未自动指向仓库内的 `flower_photos/`。
2. `ms3_val.py` 中 `CKPT` 也是绝对路径。
3. 现有脚本指定 `device_target="CPU"`；GPU 训练前需要由训练负责人验证 MindSpore 与设备环境，并在专门分支提交可审查的兼容改动。
4. 当前代码尚不读取 `configs/`，其中配置文档不能直接驱动训练。
5. 当前数据流程只有 80% train / 20% test，没有独立 validation split。正式 E0 前需统一生成并固定数据划分清单。

这些问题会影响直接运行和实验可比性，因此本 README 不声称当前命令开箱即用。

## 实验体系 E0～E6

| 编号 | 方案 | 主要目的 |
|---|---|---|
| E0 | Baseline CNN | 建立统一基准并完成问题诊断 |
| E1 | E0 + Data Augmentation | 验证数据增强的贡献 |
| E2 | E1 + Weight Decay | 验证正则化的贡献 |
| E3 | E2 + LR Scheduler | 验证学习率调度的贡献 |
| E4 | Pretrained Model | 验证迁移学习的贡献 |
| E5 | E4 + Fine-tuning | 验证解冻微调的贡献 |
| E6 | E5 + Class Weight | 仅在类别不平衡明显时验证类别权重 |

注意：现有 `ms3.py` 实际实现的是自定义 ResNet18 风格网络，而计划中的 E0 被描述为 Baseline CNN。正式实验前应由技术负责人明确 E0 到底采用哪份实现，避免把不同模型混作同一基线。

## 实验结果记录规范

每个正式实验必须同时完成三处记录：

1. 在 `configs/` 保存本次可复现配置或配置快照。
2. 在 `experiments/<experiment_id>_<slug>.md` 记录目的、唯一主要变量、环境、输出与结论。
3. 在 `results/metrics.csv` 增加一行真实结果，并填写对应 checkpoint 的本地路径、制品编号或发布地址。

不得只登记最高准确率。必须保留 train/validation 曲线、混淆矩阵、分类别 Precision/Recall/F1、worst-class recall 和 train-test gap。经复核且用于分析/报告的图表可以提交 Git；批量中间图、模型权重、原始日志和本地数据集不提交。大文件需要共享时使用团队约定的制品存储，并在记录中保留引用和校验信息。详细要求见 [实验规范](docs/EXPERIMENT_GUIDE.md)。

## 五人分工

| 角色 | 主要职责 |
|---|---|
| 1. 模型改进与实验设计负责人 | Baseline 诊断、技术路线、迁移学习/Fine-tuning、消融设计、核心分析、模型代码 review |
| 2. GPU 训练与性能评测负责人 | RTX 3090  环境、批量训练、checkpoint、正式测试、重复实验 |
| 3. 数据处理与训练优化负责人 | 预处理、增强、Dropout/Weight Decay、Scheduler、Early Stopping、类别不平衡处理 |
| 4. 模型评估与错误分析负责人 | 混淆矩阵、Precision/Recall/F1、曲线、worst-class recall、错例可视化 |
| 5. 实验工程与结果复现负责人 | 配置、seed、超参数、日志、模型版本、`metrics.csv`、复现和材料整理 |

所有角色围绕同一实验链路协作，不把五部分割裂为互不兼容的独立版本。

## Git 协作流程

推荐长期分支关系：

```text
feature/*  ->  dev  ->  main
```

- `main`：稳定、可展示和可复现版本，禁止随意直接提交。
- `dev`：日常集成分支。
- `feature/model-*`、`feature/data-*`、`feature/eval-*`、`feature/experiment-*`：按任务开发。
- `fix/*`：缺陷修复。

功能完成后向 `dev` 提 PR；阶段成果验证完成后再由 `dev` 向 `main` 提 PR。合并前至少完成相关自测，核心模型修改必须由模型改进与实验设计负责人 review。完整流程见 [GitHub 协作规范](docs/GITHUB_WORKFLOW.md)。

## Commit 规范

格式：`<type>: <简短说明>`。一个 commit 尽量只解决一个问题。

- `feat:` 新功能或新模型能力
- `fix:` 缺陷修复
- `exp:` 实验配置、实验脚本或真实结果
- `refactor:` 不改变行为的重构
- `docs:` 文档
- `chore:` 工具、依赖和工程杂项

示例：

```text
exp: add E1 data augmentation config
feat: add pretrained backbone
fix: correct per-class recall calculation
```

## Pull Request 规范

- 关联对应 Issue，并写明修改目的和边界。
- 说明是否影响训练逻辑、是否改变实验变量和数据划分。
- 列出已执行的测试；未测试项必须说明原因。
- 有真实实验时更新实验记录和 `results/metrics.csv`；无结果时不要填写占位数值。
- 不提交 checkpoint、数据集、海量日志或其他大文件。
- 指定合适 Reviewer；核心模型修改必须由技术负责人审核。
