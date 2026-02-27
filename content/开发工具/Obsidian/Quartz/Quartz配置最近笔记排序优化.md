---
title: Quartz 配置最近笔记排序优化
tags:
  - Quartz
  - Git
  - 最近笔记排序
---
本文记录 Quartz 框架中“最近笔记”组件排序异常的原因及解决方案，主要涉及 Git 提交记录作为时间元数据的优先级配置。

## 1. 问题现象描述

在 GitHub Actions 自动化构建环境下，右侧侧边栏“最近笔记”列表未按预期展示最新的编辑内容，而是显示较早期的历史笔记。

## 2. 核心原因分析

Quartz 的日期获取插件 `CreatedModifiedDate` 默认采用 `["frontmatter", "filesystem"]` 优先级。该配置在云端构建环境下存在以下缺陷：

1. **文件系统日期重置**：GitHub Actions 在执行构建时会重新 Checkout 仓库代码，导致所有文件的系统修改时间被更新为流水线运行的当前时间。
2. **逻辑回溯失效**：若笔记未在 Frontmatter 中手动指定 `date` 字段，Quartz 会回退至文件系统日期。由于所有文件在云端的系统时间一致，导致基于时间的排序逻辑失效或产生偏差。

## 3. 技术解决方案

通过调整 `quartz.config.ts` 配置文件，引入 Git 日期抓取机制并将其设为最高优先级。

### 3.1 配置调整

修改 `plugins.transformers` 中的 `CreatedModifiedDate` 配置如下：

```typescript
// quartz.config.ts
Plugin.CreatedModifiedDate({
  priority: ["git", "frontmatter", "filesystem"],
}),
```

### 3.2 优先级说明

1. **Git 记录（最高优先级）**：直接解析 Git 提交历史，获取文件真实的最后编辑时间。此配置可有效规避 GitHub Actions 环境下的文件系统日期重置问题。
2. **Frontmatter 日期**：作为补充，当 Git 记录缺失（如本地未提交文件）时读取。
3. **文件系统日期**：作为最后的兜底方案。

## 4. 实施效果

调整后，Quartz 在构建过程中会优先通过 `git log` 提取时间戳。部署后的“最近笔记”列表将严格按照文件的实际提交历史进行降序排列，确保内容的实时性与准确性。

---

*Created: 2026-02-27*
