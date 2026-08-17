---
name: research-work-summary
description: Generate Chinese research work summaries from current repository evidence. Use for 工作总结、本周工作总结、实验总结、近期实验进展、组会汇报、导师汇报、论文进展、学习总结, or EnTeR-Track/PCUM/multi-UAV tracking project progress from Git changes, code, configs, experiment results, analysis reports, papers, and logs.
---

# Research Work Summary

Use this skill to produce Chinese research summaries grounded in the current repository. The default time range is the last 7 days unless the user specifies another range.

## Operating Mode

Default to read-only evidence gathering. Do not modify code, YAML, papers, experiment results, logs, or reports unless the user explicitly asks for edits. Do not start training, long tests, deletion, commits, or pushes while using this skill.

Do not request historical summaries from the user. Do not read, copy, or save old work-summary files as templates. Generate the summary from current repository evidence and the rules below.

## Evidence Workflow

Start by checking recent repository changes:

```bash
git status --short
git log --since="<start-date>" --date=short --pretty=format:'%h|%ad|%an|%s'
git log --since="<start-date>" --name-status
git diff --stat
git diff --cached --stat
```

Use Git only to locate candidate work. Do not summarize from commit messages alone. Inspect relevant current files and outputs, especially:

- model code, training actors, test trackers, evaluation scripts, analysis scripts;
- experiment YAML files and launcher scripts;
- logs, result CSV/PKL/TXT files, Markdown reports, LaTeX papers, README files;
- visualizations and `output/analysis` artifacts;
- uncommitted changes.

Prefer `rg`, `rg --files`, `sed`, `head`, and targeted reads over broad dumps. When results are large, inspect summaries first, then drill into the files that support claims.

## Project Focus

For this repository, prioritize EnTeR-Track, PCUM, multi-UAV single object tracking, multi-view fusion, prompt generation/injection, cross-view prompt consistency, `gated_add`, dynamic gating, one-way cross attention, token pruning/recovery, Adaptive Threshold Predictor, knowledge distillation, Tiny/lightweight models, long occlusion, THREEMDOT, UAV123, UAVDT, VisDrone, DTB70, LaSOT, and TrackingNet.

## Summary Type Selection

Infer the needed summary type from evidence and the user request. Combine types when appropriate.

- Experiment progress: purpose, baseline, backbone, model changes, training strategy, configs, datasets, results, deltas, analysis, problems, next experiments.
- Module design: background, existing limitation, design idea, inputs, process, outputs, difference from prior method, training, initial results, limitations, improvements.
- Paper learning: paper name, motivation, limitations of prior work, method, modules, experiments, implications for the current project, follow-up validation. Do not merely translate abstracts.
- Paper writing: changed sections, motivation/method revisions, new experiments or figures, review-response mapping, missing evidence, writing plan.

## Experiment Evidence Rules

For each experiment, extract as many of these fields as evidence allows:

- experiment name and purpose;
- baseline, backbone, config file, checkpoint, run ID, dataset;
- training strategy, frozen/unfrozen settings, distillation, pretrained weights;
- AUC, Precision, Normalized Precision, OP50, OP75;
- FLOPs, MACs, GPU FPS, CPU FPS, latency;
- delta versus baseline;
- experiment status.

Use only evidence-supported statuses:

- completed with complete results;
- completed but results incomplete;
- run, results need review;
- training only;
- testing only;
- config only;
- code implementation only;
- currently training;
- cannot confirm from repository.

Do not treat a YAML file as a completed experiment. Do not treat an output directory as proof of success without checking result completeness and metrics.

## Result Analysis Rules

Use concrete numbers when available. Say which dataset or view improves or drops, by how much, and what tradeoff is visible. Distinguish percentage points from relative percentages: if AUC goes from 63.0 to 63.3, write "提升 0.3 个百分点", not "提升 0.3%".

When explaining causes without direct proof, use cautious language such as:

- 可能原因是;
- 初步推测;
- 从当前结果来看;
- 仍需通过消融实验验证;
- 目前尚不能确认;
- 该结论仍需进一步实验支持.

Do not write vague claims like "效果较好" or "有所提升" without metrics and context.

## Fairness Checks

Before comparing methods, check whether they share comparable settings:

- same dataset, evaluation code, invalid-frame handling, input resolution, checkpoint, pretrained weights, model scale, and baseline training;
- same FPS hardware and no mixing of GPU FPS with CPU FPS;
- same evaluation protocol;
- no cherry-picking only the best run;
- random seeds or repeated experiments when needed;
- metrics from the same experimental setting.

If settings differ or cannot be confirmed, state the limitation and avoid declaring one method strictly superior.

## Writing Style

Write in Chinese Markdown, suitable for a graduate research group meeting or advisor report. Do not produce a Git commit list or a file-change inventory. Organize the narrative as:

做了什么 -> 为什么这样做 -> 如何实现 -> 做了哪些训练或实验 -> 得到什么结果 -> 如何理解结果 -> 当前问题 -> 下一步.

Keep the language formal, natural, and clear. Avoid inflating workload by listing many filenames. Use file paths only as evidence in the final evidence section.

Strictly separate:

- 已经完成;
- 正在进行;
- 仅完成代码;
- 仅完成配置;
- 计划开展;
- 推测原因;
- 已验证结论.

Never invent metrics, training status, or paper progress. Do not present a single run as a stable conclusion unless evidence supports it.

## Default Output Structure

Use this structure by default, omitting sections with no evidence.

```markdown
# 工作总结

## 1. 本阶段主要工作

用一到两段说明本阶段研究主线、目标和主要进展。不要直接罗列文件。

## 2. 方法或模块设计

说明为什么需要该模块、当前方法的问题、采用的设计、具体实现方式、与已有方法的区别，以及预期解决的问题。

## 3. 训练和实验情况

### 3.1 实验一：实验名称

实验目的：

模型与配置：

具体方法：

测试数据集：

实验结果：

结果分析：

当前状态：

## 4. 实验结果汇总

| 实验/方法 | 配置 | 数据集 | 关键指标 | 相比基线 | 结论 | 状态 |
|---|---|---|---|---|---|---|

只填写仓库中能够确认的结果。没有找到的数据写“未找到”或“无法确认”。

## 5. 当前问题与原因分析

区分已确认问题、指标异常、实验设置问题、数据集问题、泛化问题、模块设计问题、推测原因和仍需验证的问题。

## 6. 论文或学习进展

如存在论文阅读、论文修改或方法调研，说明其对当前项目的帮助。

## 7. 后续工作

列出 2-5 项最重要、具体、可执行的工作，按优先级排列。

## 8. 相关证据

列出关键代码路径、配置文件、实验结果目录、分析报告、论文文件和相关 commit hash。
```

## Final Checks Before Answering

Before returning the summary, verify:

- the requested time range is respected, defaulting to the last 7 days;
- claims are backed by inspected files or command output;
- no historical summary text was copied;
- no repository files were modified during summary generation unless explicitly requested;
- experiment completion is not inferred from config files alone;
- comparisons include fairness caveats when settings differ;
- the final answer is not a commit list and not a file inventory.
