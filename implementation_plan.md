# 修正 Quartz 构建报错

## 发现的问题

目前项目部署遇到两个核心问题：

### 1. 模块解析错误 (Could not resolve)

GitHub Actions 构建流程会将根目录下的配置文件拷贝到 `quartz` 子目录运行，导致原有的导入路径 `./quartz/quartz/...` 失效。

### 2. 子模块同步失败 (upload-pack: not our ref)

CI 无法拉取子模块提交 `000de8e`。这说明本地对子模块进行了修改并提交了主仓库（更新了 Submodule pointer），但该提交尚未推送到远程仓库 `https://github.com/homekdio/quartz.git`。

---

## 解决方案

### [问题 1：修复导入路径]

我们将修改以下文件，移除冗余的路径前缀：

#### [MODIFY] [quartz.config.ts](file:///d:/GitHub/Chico-quartz/quartz.config.ts)

- `import { QuartzConfig } from "./quartz/cfg"`
- `import * as Plugin from "./quartz/plugins"`

#### [MODIFY] [quartz.layout.ts](file:///d:/GitHub/Chico-quartz/quartz.layout.ts)

- `import { PageLayout, SharedLayout } from "./quartz/cfg"`
- `import * as Component from "./quartz/components"`

### [问题 2：同步子模块]

由于子模块 `quartz` 指向了远程不存在的提交，您需要按以下步骤修复：

> [!IMPORTANT]
> **操作建议：**
>
> 1. 进入 `quartz` 目录。
> 2. 运行 `git push origin`（或推送到您个人的 fork 仓库，并确保 `.gitmodules` 中的 URL 顺畅）。
> 3. 只有当远程仓库存在 `000de8e` 提交时，GitHub Actions 才能通过。

---

## 验证计划

1. 修正路径后，确认主项目提交中包含这些更改。
2. 确认子模块已成功推送。
3. 观察 GitHub Actions 的 `Deploy to GitHub pages` 工作流。
