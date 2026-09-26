# GitHub 协作规范

## 1. 分支模型

```text
main
  ↑
dev
  ↑
feature/* 或 fix/*
```

- `main`：稳定、可展示、已验证的阶段版本。禁止随意直接提交或 force push。
- `dev`：团队日常集成分支；功能分支通常向它发起 PR。
- `feature/model-xxx`：模型或迁移学习能力。
- `feature/data-xxx`：数据处理、增强和划分。
- `feature/eval-xxx`：指标、曲线、混淆矩阵和错例分析。
- `feature/experiment-xxx`：实验配置、运行脚本与实验记录。
- `fix/xxx`：缺陷修复。

建议在 GitHub 为 `main`（必要时也为 `dev`）配置 branch protection：必须通过 PR、至少一名 Reviewer 批准、解决所有 review conversation 后才能合并。课程团队较小时，可采用 squash merge 保持主线清晰。

## 2. 标准开发流程

```bash
git switch dev
git pull --ff-only
git switch -c feature/eval-confusion-matrix
# 开发、自测、记录真实结果
git add <相关文件>
git commit -m "feat: add confusion matrix evaluation"
git push -u origin feature/eval-confusion-matrix
```

随后在 GitHub 创建到 `dev` 的 Pull Request。功能完成后删除远程功能分支。阶段版本由 `dev` 向 `main` 发起单独 PR；不要绕过评审直接把本地分支推到 `main`。

## 3. Issue 规范

- 开始实现前创建对应 Issue，并选择 experiment、bug 或 feature 模板。
- 一个 Issue 描述一个清晰目标，写明验收条件、负责人和依赖。
- 实验 Issue 必须注明主要修改变量与必须保持不变的条件。
- PR 使用 `Closes #<编号>` 或 `Refs #<编号>` 关联 Issue。

## 4. Commit 规范

格式：

```text
<type>: <imperative summary>
```

允许的常用类型：

- `feat:` 新功能或模型能力
- `fix:` 缺陷修复
- `exp:` 实验配置、实验执行相关代码或真实结果
- `refactor:` 不改变对外行为的重构
- `docs:` 文档修改
- `chore:` 依赖、工具或工程维护

示例：

```text
exp: add E1 data augmentation config
feat: add pretrained backbone
fix: correct per-class recall calculation
```

要求：

- 一个 commit 尽量只解决一个问题。
- 不把格式化、模型逻辑变更和实验结果混在一个 commit 中。
- Commit message 描述真实修改，不使用 `update`、`change` 等无信息词。
- 不提交数据集、checkpoint、密钥、虚拟环境、缓存或大体积日志。
- 真实结果必须能追溯到产生它的代码 commit；不得修改或美化原始数值。

## 5. Pull Request 规范

- PR 标题遵循 Commit 类型，例如 `feat: add pretrained backbone`。
- 内容必须使用仓库 PR 模板，说明修改边界、测试、变量变化和结果。
- PR 应保持小而可审查；无关改动拆分到其他 PR。
- 提交前同步 `dev` 并解决冲突，不覆盖他人工作。
- 至少完成与修改相关的自测；未执行的测试须写明原因和风险。
- 核心模型实现、loss、数据划分或评估公式的修改，需要技术负责人 review。
- 数据与增强由数据负责人重点 review；指标实现由评估负责人重点 review；配置及结果追溯由复现负责人重点 review。
- 合并实验结果前，核对实验 Markdown、`metrics.csv`、配置、commit SHA 和制品引用一致。

## 6. Review 检查清单

- 是否符合 Issue 的目标和验收条件？
- 是否意外改变了数据划分、随机种子或多个实验变量？
- 训练集、验证集、测试集是否严格隔离？
- 指标公式和平均方式是否明确且正确？
- 是否存在硬编码本机绝对路径？
- 是否写入密钥、数据集、checkpoint 或大文件？
- 是否提供足够的运行、测试和复现说明？
- 是否同步更新实验记录和结果总表？

## 7. 冲突与紧急修复

普通冲突由分支作者在自己的功能分支解决并重新自测。紧急修复使用 `fix/<slug>`，仍通过 PR 进入目标分支。除非全组明确同意并保留原因记录，否则不使用 force push 改写共享分支历史。

