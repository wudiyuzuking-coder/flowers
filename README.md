# 花卉图像识别模型改进

这是一个 5 人协作的课程项目。项目在现有花卉分类代码基础上，通过统一的 Baseline、问题诊断、单变量改进、消融实验和错误分析，形成可复现的模型改进结论与课程报告。

> 当前状态：`ms3.py` 可通过 `--experiment-id E0|E1|E2|E3|E4|E5|E6` 运行两条实验链。E0～E3 使用同一套自定义 ResNet18；E4 是冻结 Backbone 的迁移学习基线；E5 完整继承 E4，并只解冻 `layer4` 做 partial fine-tuning；E6 是完整继承 E5、只增加 class-weighted loss 的候选类别均衡实验。所有实验统一读取固定分层 split manifest，并共用同一评估模块。尚未进行正式实验或填写实验结果。

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
- MindCV 0.3.0（E4/E5/E6 的官方 ImageNet pretrained ResNet18）
- 数据集：5 类花卉图片（daisy、dandelion、roses、sunflowers、tulips）
- 当前模型代码：自定义 ResNet18 风格网络

依赖来自现有源码的实际 import。E4 使用已核对 API 与权重注册表的 `mindcv==0.3.0`；MindCV 官方兼容表对应 MindSpore 2.2.10，但当前仓库尚未锁定设备相关的 MindSpore 安装版本。正式训练前必须在目标 GPU 环境验证这组版本，并记录 Python、MindSpore、MindCV、CUDA/驱动和设备版本。

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
├── splits/                      # 固定、分层的 train/val/test manifest 与校验信息
├── tools/
│   └── generate_split_manifest.py # 一次性生成固定 split
├── evaluation.py               # 公共指标、曲线与混淆矩阵实现
├── split_manifest.py           # manifest 生成、校验和数据源
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

首次正式实验前先生成一次固定 split：

```powershell
python tools/generate_split_manifest.py
```

生成器使用 seed 42，在每个类别内部独立打乱并按约 70%/10%/20% 分配。`splits/split_info.json` 记录类别分布与三个 manifest 的 SHA-256；如果文件已存在，默认拒绝覆盖，只有明确传入 `--force` 才会重新生成。

之后运行训练（默认 E0）：

```powershell
python ms3.py --experiment-id E0
python ms3.py --experiment-id E1
python ms3.py --experiment-id E2
python ms3.py --experiment-id E3
python ms3.py --experiment-id E4
python ms3.py --experiment-id E5
python ms3.py --experiment-id E6
```

E4/E5/E6 默认由 MindCV 下载或复用缓存中的官方 ResNet18 ImageNet checkpoint。无网络环境可预先取得同一个官方 1000 类 checkpoint，并严格离线加载：

```powershell
python ms3.py --experiment-id E4 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
python ms3.py --experiment-id E5 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
python ms3.py --experiment-id E6 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
```

离线 checkpoint 必须与 MindCV 0.3.0 的 1000 类 ResNet18 完整匹配；缺失文件或任何参数不匹配都会终止运行，不会静默退回随机初始化。

现有加载 checkpoint 并预测/评估的入口：

```powershell
python ms3_val.py
```

运行前必须注意：

1. `ms3.py` 使用仓库内相对路径 `./flower_photos`；`ms3_val.py` 仍保留原开发机的数据和 checkpoint 绝对路径。
2. `ms3.py` 指定 `device_target="GPU"`，`ms3_val.py` 指定 `CPU`；正式训练前需验证 MindSpore 与 GPU 环境。
3. `ms3.py` 当前通过命令行实验编号控制 E0/E1/E2/E3/E4/E5/E6，尚不读取 `configs/` 文件。
4. E0～E6 全部读取 `splits/train.txt`、`splits/val.txt` 和 `splits/test.txt`，训练启动时会验证 SHA-256、交集、全集覆盖、类别数量和本地数据漂移；E6 的 class weight 只从 `train.txt` 自动统计。
5. E4/E5/E6 依赖 MindCV 0.3.0；当前开发环境未完成 MindSpore/MindCV 运行级验证，正式训练前必须做 checkpoint 加载、冻结参数、weighted loss 和单批更新 smoke test。

这些问题会影响直接运行和实验可比性，因此本 README 不声称当前命令开箱即用。

## 实验体系 E0～E6

| 编号 | 方案 | 主要目的 |
|---|---|---|
| E0 | Baseline CNN | 建立统一基准并完成问题诊断 |
| E1 | E0 + Data Augmentation | 验证数据增强的贡献 |
| E2 | E1 + Wider Weight Decay | 验证将衰减从 FC weight 扩展到 Backbone Conv/Dense weight 的贡献 |
| E3 | E2 + Cosine LR Scheduler | 验证逐 step 动态学习率的贡献 |
| E4 | ImageNet Pretrained ResNet18 + Frozen Backbone | 建立只训练新 5 类头的迁移学习基线 |
| E5 | E4 + Partial Fine-tuning of `layer4` | 验证有限高层语义适配的贡献 |
| E6 | E5 + Class-weighted Loss | 候选实验：验证训练集类别权重对类别均衡指标的影响 |

实验结构分为两条链，而不是 E0→E4 的严格线性消融：

- Baseline improvement track：E0 → E1 → E2 → E3。
- Transfer learning track：E4 → E5 → E6（E6 为 conditional/candidate imbalance experiment）。

E0～E3 均使用 `ms3.py` 中同一份自定义 ResNet18 和 Adam。E1 不替换模型，只增加训练集数据增强；E2 不改变 E1 增强，仅将 weight decay 从 `fc.weight=0.01` 扩展到 Backbone Conv/Dense weight（`0.0001`）；E3 完整继承 E2，仅把固定 `0.0001` 学习率改为从 `0.0001` 到 `0.000001` 的逐 step cosine schedule。

E4 不继承 E1～E3 的增强、Backbone weight decay 或 cosine scheduler。它使用 MindCV ResNet18 的官方 ImageNet 权重，冻结 Backbone（包括固定 BatchNorm 统计），替换为 5 类分类头，并保持 Adam、固定 learning rate `0.0001` 与分类头 weight decay `0.01`。E4 使用该预训练权重配套的 224×224 与 ImageNet normalization，因此它是跨模型、跨初始化和配套预处理的迁移学习对照，不能把结果解释为 E3 上某个单变量的独立贡献。

E5 完整继承 E4 的模型、预训练权重、新分类头、输入预处理、split、evaluation 和 checkpoint 流程，唯一核心变化是解冻 `layer4`。E5 仍冻结 `conv1`、`bn1`、`layer1`～`layer3`，并固定所有 BatchNorm running statistics；`layer4` Conv weight 与 BN gamma/beta 使用固定 LR `0.00001`，classifier 使用固定 LR `0.0001`。E5 不启用 E1 augmentation、E2 实验开关或 E3 scheduler，是 partial fine-tuning 而不是 full fine-tuning。

E6 完整继承 E5，唯一主要变化是训练 loss 使用仅由 `splits/train.txt` 自动统计的 normalized inverse-frequency class weight，公式为 `w_c = N / (K * n_c)`。Validation Accuracy 仍是唯一 best-checkpoint 选择指标，evaluation 指标公式不变。E6 尚无正式结果，也尚未由诊断确认类别不平衡是性能瓶颈，因此只能作为 conditional/candidate 类别均衡实验，不能预设 Accuracy 或类别指标会提高。

训练期间统一记录确定性 train-evaluation 与 validation 的 loss/accuracy，并仍以 validation accuracy 选择 best checkpoint。加载 best checkpoint 后，公共评估模块遍历完整 test manifest，输出 Overall Accuracy、Macro Precision/Recall/F1、分类别指标、worst-class recall、train-test gap、JSON 和混淆矩阵。zero division 统一按 0 处理；`results/metrics.csv` 不会被训练脚本自动修改。

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
