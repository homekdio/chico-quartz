# 修正 Quartz 构建模块路径引用

GitHub Actions 中的构建流程会将根目录下的 `quartz.config.ts` 和 `quartz.layout.ts` 拷贝到 `quartz` 子目录中运行。由于拷贝后的相对路径发生变化，导致构建报错：`Could not resolve "./quartz/quartz/plugins"`。

> [!IMPORTANT]
> 我们需要将相对路径中的 `./quartz/` 前缀移除。由于配置文件将被直接放在 `quartz` 目录下，因此导入路径应变为 `./quartz/plugins`。

## 方案说明

我们将修改以下文件：

1. `quartz.config.ts`
2. `quartz.layout.ts`

### [Quartz 配置修复]

#### [MODIFY] [quartz.config.ts](file:///d:/GitHub/Chico-quartz/quartz.config.ts)

- 修改 `import { QuartzConfig } from "./quartz/quartz/cfg"` 为 `import { QuartzConfig } from "./quartz/cfg"`
- 修改 `import * as Plugin from "./quartz/quartz/plugins"` 为 `import * as Plugin from "./quartz/plugins"`

#### [MODIFY] [quartz.layout.ts](file:///d:/GitHub/Chico-quartz/quartz.layout.ts)

- 修改 `import { PageLayout, SharedLayout } from "./quartz/quartz/cfg"` 为 `import { PageLayout, SharedLayout } from "./quartz/cfg"`
- 修改 `import * as Component from "./quartz/quartz/components"` 为 `import * as Component from "./quartz/components"`

## 验证计划

### 自动化测试

- 模拟拷贝操作，并在本地运行 `npx quartz build` 验证配置文件是否能被正常解析。

### 手动验证

- 提交代码到 GitHub，触发 GitHub Actions 验证构建是否通过。
