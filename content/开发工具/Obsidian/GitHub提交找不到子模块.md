---
title: GitHub Actions 子模块 "not our ref" 报错解决方案
date: 2026-02-27
tags:
  - Git
  - GitHubActions
  - Quartz
  - 故障排除
---

# GitHub Actions 子模块 "not our ref" 报错解决方案

## 1. 现象描述

在使用 GitHub Actions 自动化部署（例如 Quartz 博客）时，由 `actions/checkout` 触发的子模块同步步骤失败，报错信息如下：

```text
Fetching submodules
  /usr/bin/git submodule sync
  /usr/bin/git -c protocol.version=2 submodule update --init --force
  Submodule 'quartz' (https://github.com/homekdio/quartz.git) registered for path 'quartz'
  Cloning into '/home/runner/work/chico-quartz/chico-quartz/quartz'...
  Error: fatal: remote error: upload-pack: not our ref 3f3963b4db9f5755397c5b6cca39a5694c37087e
  Error: fatal: Fetched in submodule path 'quartz', but it did not contain 3f3963b4db9f5755397c5b6cca39a5694c37087e. Direct fetching of that commit failed.
  Error: The process '/usr/bin/git' failed with exit code 128
```

## 2. 问题根源分析

这是一个典型的 **Submodule 指针不匹配** 问题。

1. **主仓库记录了新指针**：你在本地修改了子模块（`quartz` 目录）的代码并执行了 `commit`。此时，主仓库记录了子模块指向了一个新的提交 ID（例如 `3f3963b...`）。
2. **只有本地有该提交**：由于这笔 `commit` 发生在子模块内部，如果你**只推送了主仓库**而没有**进入子模块目录推送其自身**，那么 `3f3963b...` 这笔提交就只存在于你的本地电脑，而不在子模块的远程仓库（GitHub）里。
3. **Actions 拉取失败**：当 GitHub Actions 运行并试图拉取主仓库要求的 `3f3963b...` 时，它访问子模块的远程地址却找不到这个号，于是抛出 `not our ref` 错误。

## 3. 故障排除步骤

### 方案 A：命令行修复（推荐）

1. **进入子模块目录**：

   ```bash
   cd quartz
   ```

2. **推送子模块代码**：
   确保你处于正确的分支（如 `v4`），并将本地提交推送到服务器。

   ```bash
   git checkout v4
   git merge 3f3963b  # 如果你在 Detached HEAD 状态下做了修改，需要合并回来
   git push origin v4
   ```

3. **更新主仓库引用**：
   返回根目录，确保主仓库也记录并推送了这次更新。

   ```bash
   cd ..
   git add quartz
   git commit -m "chore: update quartz reference"
   git push
   ```

### 方案 B：VS Code 图形界面操作

如果你偏好使用 VS Code 的“源代码管理”面板，请确保：

![[Pasted image 20260227211359.png]]
1. 先找到底部的 **quartz (子模块)** 仓库，点击提交并点击**同步**图标。
2. 等子模块同步完成后，再点击顶部的 **Chico-quartz (主项目)** 仓库进行提交和同步。

## 4. 预防措施

**核心原则：先推子模块，再推主项目。**

建议在修改具有 Submodule 的项目时，遵循以下工作流：

1. `cd submodule_dir` -> `git add/commit/push`
2. `cd ..` -> `git add submodule_dir` -> `git commit/push`

只要子模块的远程仓库包含了主仓库所引用的所有 Commit ID，GitHub Actions 就不会报错。
