# E0～E6 实验运行与结果交付手册

本文是花卉分类项目正式运行 E0～E6 的主操作手册。实验负责人应从仓库根目录执行所有命令。本手册不定义 E7，也不授权修改既有实验配置。

## 1. 实验总览

```text
Baseline improvement track

E0 Baseline
 ↓
E1 Extra Data Augmentation
 ↓
E2 Wider Weight Decay
 ↓
E3 Cosine Learning Rate Scheduler

Transfer learning track

E4 ImageNet Pretrained ResNet18 + Frozen Backbone
 ↓
E5 Partial Fine-tuning layer4
 ↓
E6 E5 + Class-weighted Loss
   (conditional / candidate imbalance experiment)
```

E4 不是 E3 的单变量后继实验：E4 同时采用 MindCV 模型实现、ImageNet 初始化和配套的 224×224 normalization，因此只能作为独立迁移学习基线。E6 目前是候选实验；是否纳入报告核心改进链，必须根据正式结果中的 per-class recall、Macro Recall、Macro F1、Worst-class Recall 和 Confusion Matrix 决定，不能预设有效。

## 2. 冻结配置：正式对比期间禁止改变

> **禁止修改后仍沿用 E0～E6 原编号。** 以下任一项发生变化，都必须创建新的实验 ID，并与原结果分开记录。

- split manifests
- random seed（当前为 42）
- batch size（当前为 32）
- epochs（当前为 10）
- augmentation 参数
- weight decay 数值和参数范围
- learning rate
- scheduler
- preprocessing 和 input size
- pretrained model/checkpoint 类型
- fine-tune scope
- class-weight 公式
- Validation Accuracy best-checkpoint 规则
- evaluation 指标公式

不要在正式运行前“顺手优化”任何参数，也不要用 test 结果反向修改配置。

## 3. 环境准备与环境记录

### 3.1 代码实际依赖

`requirements.txt` 当前包含：

- MindSpore（版本未锁定，且代码使用 `GRAPH_MODE` 和 `device_target="GPU"`）
- MindCV `0.3.0`（E4/E5/E6）
- NumPy
- Matplotlib
- EasyDict

还需要与 MindSpore GPU 包兼容的 NVIDIA GPU、驱动和 CUDA 运行环境。仓库没有证据证明某个特定 Python/CUDA/驱动组合已完整运行 E0～E6；MindCV 0.3.0 是代码明确检查的版本，但 MindSpore、Python、CUDA 与驱动仍必须在目标训练机实测。

建议使用独立虚拟环境：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

不要仅凭安装成功就声明 GPU 环境通过。正式训练前至少完成依赖导入、preflight 和项目相关 runtime smoke test。

### 3.2 每台训练机必须记录

```text
OS:
Python version:
MindSpore version:
MindCV version:
NumPy version:
Matplotlib version:
EasyDict version:
GPU:
NVIDIA driver:
CUDA/runtime information:
Git commit SHA:
Working tree clean: yes/no
Run owner:
```

可辅助采集：

```powershell
python --version
python -m pip show mindspore mindcv numpy matplotlib easydict
nvidia-smi
git rev-parse HEAD
git status --short
```

## 4. 数据目录与固定 Split

### 4.1 数据目录

仓库根目录下必须存在：

```text
flower_photos/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

当前固定数据记录为 3670 张图片：Train 2568、Validation 367、Test 735。当前 train 类别计数为：

| Label | Class | Train count |
|---:|---|---:|
| 0 | daisy | 443 |
| 1 | dandelion | 628 |
| 2 | roses | 449 |
| 3 | sunflowers | 489 |
| 4 | tulips | 559 |

这些数字用于人工核对；正式程序以 manifest、SHA-256 和当前数据文件联合校验为准。

### 4.2 Split manifests

```text
splits/
├── train.txt
├── val.txt
├── test.txt
└── split_info.json
```

正式运行必须复用已有 manifests。不要运行：

```powershell
python tools/generate_split_manifest.py --force
```

除非项目负责人明确决定整个实验项目从新 split 重新开始。一旦 manifests 被重新生成，E0～E6 必须全部重跑；新旧 split 结果不能混入同一对比表。

当前记录的 SHA-256：

```text
train.txt e3746f3898a0bf61e9082dad57e3b7da60cd0ad5d37cfef8df6c3f86a48e7aee
val.txt   cd1d81e02b8b4a8089258f48a37d692805d3cdf642ea54350964493db3daec95
test.txt  560d77f600d243e2ea1f084bebd49d8b1dbb7ce65ee2b2fa1ecce82adc4136b3
```

## 5. E4/E5/E6 Pretrained Checkpoint

E4、E5、E6 必须使用 MindCV 0.3.0 官方 ImageNet-1K ResNet18 权重。加载失败时必须停止，禁止改用随机初始化继续。

### 5.1 在线模式

MindCV 自动下载或复用其缓存：

```powershell
python ms3.py --experiment-id E4
python ms3.py --experiment-id E5
python ms3.py --experiment-id E6
```

正式训练前确认网络可访问官方权重源或本机缓存已有正确权重，并从启动输出确认 pretrained source。

### 5.2 离线模式

先由负责人准备官方 1000 类 checkpoint `resnet18-1e65cd21.ckpt`，然后传入当前 CLI 实际支持的参数：

```powershell
python ms3.py --experiment-id E4 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
python ms3.py --experiment-id E5 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
python ms3.py --experiment-id E6 --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
```

代码使用严格参数加载；缺失、多余或不匹配参数会终止。记录 checkpoint 来源、文件大小和 SHA-256：

```powershell
Get-FileHash C:\path\to\resnet18-1e65cd21.ckpt -Algorithm SHA256
```

## 6. 正式运行前 Preflight

Preflight 不生成 split、不修改数据、不下载 checkpoint、不启动训练，也不写正式结果目录。它检查：

- 当前 commit SHA 与 working tree 是否干净
- Python 和 requirements 中的运行依赖
- MindCV 是否为 0.3.0
- GPU/driver 是否可由 `nvidia-smi` 发现
- `flower_photos` 和五个类别目录
- 四个 split 文件是否存在
- manifest SHA-256、类别路径、文件存在性、无交集、union 和数据漂移
- split 总数及 train 类别计数
- 在线/离线 pretrained 条件
- checkpoint/result 目录结构是否可在临时目录创建

在线 checkpoint 模式：

```powershell
python tools/preflight_check.py --pretrained-mode online
```

离线 checkpoint 模式：

```powershell
python tools/preflight_check.py --pretrained-mode offline --pretrained-checkpoint C:\path\to\resnet18-1e65cd21.ckpt
```

只有最终 `RESULT: PASS` 才代表该脚本的硬检查全部通过。`WARN` 必须阅读并记录。MindSpore/MindCV 缺失、Git dirty、manifest 失败或 GPU 不可发现会得到 `RESULT: FAIL`；脚本仍尽量完成其余纯文件检查。即使 PASS，也不能替代 MindSpore GRAPH_MODE、checkpoint 严格加载和单 batch 梯度 smoke test。

## 7. 正式运行顺序与命令

推荐按依赖关系运行：E0 → E1 → E2 → E3 → E4 → E5 → E6。E6 为候选实验，可在决定评估类别均衡方案时运行。

```powershell
python ms3.py --experiment-id E0
python ms3.py --experiment-id E1
python ms3.py --experiment-id E2
python ms3.py --experiment-id E3
python ms3.py --experiment-id E4
python ms3.py --experiment-id E5
python ms3.py --experiment-id E6
```

E4/E5/E6 若使用离线权重，改用第 5.2 节列出的命令。一次只运行一个正式实验；等待该实验完成、核对产物并完成记录后再开始下一个。

当前代码不会自动保存控制台日志或 runtime。运行前人工记录开始时间，结束后记录结束时间和时长。若团队需要保存 PowerShell 输出，可先建立被 `.gitignore` 排除的日志目录，再使用：

```powershell
New-Item -ItemType Directory -Force results/logs | Out-Null
python ms3.py --experiment-id E0 2>&1 | Tee-Object -FilePath results/logs/E0.log
```

替换实验编号运行其他实验。使用管道时还应检查训练输出和生成产物，不能仅凭 shell 命令结束判断成功。

## 8. 每个实验的预期输出

| 实验 | 运行命令 | Best checkpoint | Metrics JSON | Training curve | Confusion matrix | 特殊输出 |
|---|---|---|---|---|---|---|
| E0 | `python ms3.py --experiment-id E0` | `checkpoints/E0/E0_best.ckpt` | `results/evaluation/E0_metrics.json` | `results/curves/E0_training_curves.png` | `results/confusion_matrix/E0_confusion_matrix.png` | 控制台配置/参数组 |
| E1 | `python ms3.py --experiment-id E1` | `checkpoints/E1/E1_best.ckpt` | `results/evaluation/E1_metrics.json` | `results/curves/E1_training_curves.png` | `results/confusion_matrix/E1_confusion_matrix.png` | 控制台确认 extra augmentation |
| E2 | `python ms3.py --experiment-id E2` | `checkpoints/E2/E2_best.ckpt` | `results/evaluation/E2_metrics.json` | `results/curves/E2_training_curves.png` | `results/confusion_matrix/E2_confusion_matrix.png` | 控制台 optimizer 参数组 |
| E3 | `python ms3.py --experiment-id E3` | `checkpoints/E3/E3_best.ckpt` | `results/evaluation/E3_metrics.json` | `results/curves/E3_training_curves.png` | `results/confusion_matrix/E3_confusion_matrix.png` | `results/curves/E3_lr_curve.png` |
| E4 | `python ms3.py --experiment-id E4` | `checkpoints/E4/E4_best.ckpt` | `results/evaluation/E4_metrics.json` | `results/curves/E4_training_curves.png` | `results/confusion_matrix/E4_confusion_matrix.png` | pretrained source、冻结/可训练参数审计 |
| E5 | `python ms3.py --experiment-id E5` | `checkpoints/E5/E5_best.ckpt` | `results/evaluation/E5_metrics.json` | `results/curves/E5_training_curves.png` | `results/confusion_matrix/E5_confusion_matrix.png` | layer4/classifier LR、WD、参数审计 |
| E6 | `python ms3.py --experiment-id E6` | `checkpoints/E6/E6_best.ckpt` | `results/evaluation/E6_metrics.json` | `results/curves/E6_training_curves.png` | `results/confusion_matrix/E6_confusion_matrix.png` | train counts、class weights、E5 参数审计 |

只有 E3 当前会额外生成 LR curve。日志不是训练脚本自动产物，必须由实验负责人保存控制台输出。checkpoint 被 `.gitignore` 排除，交付时使用团队制品存储并记录路径/引用与校验值。

每次运行结束必须确认：

1. 控制台没有异常退出，实验开关和关键参数符合目标实验。
2. best checkpoint 存在。
3. JSON 中 `experiment_id`、checkpoint 路径和数值字段完整。
4. training curve 与 confusion matrix 文件存在且可打开。
5. E3 的 LR curve 存在。
6. E4/E5/E6 的 pretrained source 正确，参数审计无报错。

## 9. 每次正式实验记录清单

- experiment_id
- date/time、负责人、开始/结束时间、runtime
- Git commit SHA 和 working tree 状态
- OS、Python、MindSpore、MindCV、CUDA/runtime、GPU、driver
- train/val/test manifest SHA-256
- seed、batch size、epochs
- best epoch
- best validation accuracy
- best-epoch train accuracy
- test accuracy
- macro precision、macro recall、macro F1
- worst-class name、worst-class recall
- checkpoint 路径/制品引用/SHA-256
- metrics JSON、training curve、confusion matrix、特殊产物和日志路径
- 异常、重跑原因、环境差异和其他 notes

复制 `docs/RESULT_HANDOFF_TEMPLATE.md` 为每次运行填写交付记录，不要直接把模板改成某个实验结果。

## 10. `results/metrics.csv` 回填规则

当前表头：

```text
experiment_id,model,augmentation,weight_decay,scheduler,fine_tune,class_weight,seed,best_epoch,train_acc,val_acc,test_acc,precision,recall,f1,worst_class_recall,train_test_gap,checkpoint,notes
```

数值只能来自正式运行产生的 `results/evaluation/E#_metrics.json`：

| CSV 字段 | 来源/填写规则 |
|---|---|
| `experiment_id` | 正式实验 ID；重复 run 使用唯一后缀并在 notes 说明 |
| `model` | E0～E3 为 custom ResNet18；E4～E6 为 MindCV 0.3.0 ImageNet pretrained ResNet18，并说明冻结范围 |
| `augmentation` | 按对应实验文档填写，不从结果图猜测 |
| `weight_decay` | 写明参数范围和数值，不只写单个模糊数字 |
| `scheduler` | E3 为 per-step cosine；其余为 none |
| `fine_tune` | E4 为 frozen backbone；E5/E6 为 layer4；E0～E3 为 none |
| `class_weight` | 只有 E6 填 normalized inverse-frequency；其余为 none |
| `seed` | 当前固定为 42 |
| `best_epoch` | JSON `best_epoch` |
| `train_acc` | JSON `best_epoch_train_accuracy` |
| `val_acc` | JSON `best_val_accuracy` |
| `test_acc` | JSON `accuracy` |
| `precision` | JSON `macro_precision` |
| `recall` | JSON `macro_recall` |
| `f1` | JSON `macro_f1` |
| `worst_class_recall` | JSON `worst_class_recall`；类别名放 notes/交付模板 |
| `train_test_gap` | JSON `train_test_gap` |
| `checkpoint` | JSON `checkpoint`，并在交付记录补充制品引用/SHA |
| `notes` | worst class、环境/运行标识、异常、重跑原因等 |

禁止从控制台估算、从曲线图片肉眼读取、使用 smoke-test 数值、填虚构值，或多次 test 后只保留最高结果。回填前保留 JSON 原件并交叉核对实验 ID。

当前代码固定 seed=42，不要求擅自开展多 seed 实验。如果项目负责人未来正式批准多 seed，每个 seed/run 必须单独一行，以唯一 run ID（例如 `E1-S7`）标识，并保留所有 run；汇总 mean/std 应另行生成，不能覆盖原始行。相同 seed 因故重跑也使用 `E1-R1`、`E1-R2` 等唯一记录并说明原因，不能挑选最好的一次。

## 11. Test 使用纪律

> Test 只允许用于：训练完成 → 载入 Validation Accuracy 选出的 best checkpoint → 最终一次 evaluation。

严禁：

- 根据 Test Accuracy 调 learning rate
- 根据 test 选择 checkpoint
- 根据 test 调 class weight
- 根据 test 改 augmentation、preprocessing 或其他超参数
- 多次尝试后只保留最好的 test
- 把 test 当作 validation 重复查看

Validation 才用于训练过程中的模型选择。如果 test 已被非正式查看或用于决策，必须在 notes 中披露，相关结果不能再被当作无偏最终评估。

## 12. 结果分析框架

### 12.1 E0～E3

- E0 vs E1：只解释 Extra Data Augmentation 的差异。
- E1 vs E2：只解释 Wider Weight Decay 的差异。
- E2 vs E3：只解释 Cosine LR Scheduler 的差异。

每组同时比较 Test Accuracy、Macro F1、Train-Test Gap、Worst-class Recall 和训练/验证曲线。检查提升是否伴随过拟合变化、类别间权衡或训练不稳定，不能只看 Accuracy，也不能提前写提升结论。

### 12.2 E4～E6

- E4：独立的 pretrained frozen-backbone baseline，不与 E3 宣称单变量因果关系。
- E5 vs E4：重点分析解冻 `layer4` 的 partial fine-tuning。
- E6 vs E5：重点分析 class-weighted loss。

E6 特别比较 Macro Recall、Macro F1、每类 recall、Worst-class Recall、Confusion Matrix 和 Overall Accuracy。即使 Overall Accuracy 未提高，也不能仅凭这一点判定无价值；反之，某一指标提高也不能忽略其他类别退化。是否有效只能由完整真实结果决定。

## 13. 项目相关 Troubleshooting

1. **Dataset shape 错误**：先确认 E0～E3 输入为 100×100，E4～E6 为 224×224；确认没有手工改 transform 顺序，检查报错批次对应图片能否解码。
2. **Label 超出 0～4**：检查 manifest 第一层目录名只能是五个固定类别，运行 preflight；不要通过截断或取模掩盖错误。
3. **MindCV pretrained 下载失败**：检查网络/缓存；无法联网时改用官方 checkpoint 的离线 CLI，不得回退随机初始化。
4. **Checkpoint 参数不匹配**：确认文件是 MindCV 0.3.0 官方 ResNet18 ImageNet 1000 类 checkpoint，而不是已替换 5 类 head 的实验 checkpoint；保留严格加载错误。
5. **E4 Backbone 意外可训练**：查看启动参数审计；E4 trainable set 必须只有 classifier，任何 Backbone 参数出现都应停止实验。
6. **E5 layer4 没更新**：先确认运行的是 E5/E6，审计中应包含 layer4 + classifier；在正式训练外做单 batch 参数前后对比，不能用 test 指标猜测。
7. **E3 LR Tensor 长度不匹配**：核对 `steps_per_epoch × 10` 与控制台 total decay steps，确认数据集和 batch size 未被改动，也未额外 repeat。
8. **Weighted loss GRAPH_MODE 报错**：仅针对 E6，用一个 batch 检查 label dtype/shape、logits `[N,5]`、gather 后权重 `[N]` 和标量 loss；不要临时改公式绕过。
9. **CUDA/MindSpore 环境错误**：记录完整异常、Python/MindSpore/CUDA/driver/GPU；按 MindSpore GPU 安装兼容要求修复环境，不改 `device_target` 后冒充同一正式实验。
10. **Out of memory**：先关闭其他 GPU 进程并确认实际 batch size 32 和输入尺寸符合实验；若必须改 batch size，创建新实验 ID，不能覆盖 E0～E6。

## 14. 结果文件交付

不移动现有项目目录。每个实验向报告负责人交付下列现有产物或制品引用：

```text
checkpoints/E#/E#_best.ckpt
results/evaluation/E#_metrics.json
results/curves/E#_training_curves.png
results/confusion_matrix/E#_confusion_matrix.png
results/curves/E3_lr_curve.png       # 仅 E3
results/logs/E#.log                  # 若按本手册保存控制台日志
docs/RESULT_HANDOFF_TEMPLATE.md 的已填写副本
对应 experiments/E#_*.md 的正式结果更新
results/metrics.csv 的对应行
```

checkpoint 和原始日志默认被 `.gitignore` 排除；通过团队制品存储交付，并提供路径/URL、文件名、SHA-256。经复核用于报告的 metrics JSON、曲线和混淆矩阵可按项目流程提交。交付前确认产物来自同一 commit、split 和 run，不要混用不同运行文件。

## 15. 完成判定

一个正式实验只有同时满足以下条件才算交付完成：

1. preflight 通过且 runtime smoke test 有记录。
2. 使用冻结配置完整运行，无异常中断。
3. Validation Accuracy 选出的 best checkpoint 已保存。
4. best checkpoint 完成一次最终 test evaluation。
5. JSON、曲线、混淆矩阵、checkpoint 和日志/关键控制台信息齐全。
6. `metrics.csv` 只用 JSON 正式值回填。
7. 实验文档和结果交付记录完整，环境、commit、split SHA、runtime 可追溯。
8. 没有虚构、挑选或未披露的重复 test 结果。
