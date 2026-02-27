---
title: 配置 Frontmatter 并设置成模板
tags:
  - Obsidian
  - Quartz
---

## 什么是 Frontmatter

Frontmatter 是 Markdown 文件**最顶部**用三个短横线 `---` 包裹的一段元数据（metadata），写给 Quartz 等渲染工具看的，而非正文内容。

```yaml
---
title: 文章标题
tags:
  - 标签1
  - 标签2
---
```

经过 Quartz 网页渲染之后，`title` 会作为页面标题显示，`tags` 会变成可点击的标签链接：

![[Frontmatter图片渲染事例.png]]

> [!TIP]
> 正文中**不需要**再写 `# 标题`，否则会跟 Frontmatter 的 `title` 重复显示。

---

## 如何设置模板

每次新建 Markdown 文件都手写 Frontmatter 太麻烦，可以用 Obsidian 的**模板插件**一键插入。

### 第一步：配置忽略规则

找到项目根目录的 `quartz.config.ts`，确认 `ignorePatterns` 中包含 `_templates`：

```typescript
ignorePatterns: ["private", "_templates", ".obsidian"]
```

这样 `_templates` 文件夹下的内容**不会被 Quartz 渲染成网页**，只作为 Obsidian 的模板使用。

### 第二步：创建模板文件

在根目录下创建 `_templates/` 文件夹，在里面新建 `note.md`，写入以下内容：

```yaml
---
title: {{title}}
tags:
  - 
---
```

其中 `{{title}}` 是 Obsidian 的模板变量，插入模板时会**自动替换为当前文件名**。

### 第三步：启用模板插件

1. 打开 Obsidian **设置** → **Core plugins（核心插件）** → 找到 **Templates** → 开启
2. 点开 Templates 插件的设置 → **Template folder location（模板文件夹位置）** → 填入 `_templates`

### 第四步：使用模板

新建一个 Markdown 文件后：

1. 按 `Ctrl + P` 打开命令面板
2. 输入 `Insert template`（或中文 `插入模板`）
3. 选择 `note` → Frontmatter 自动填好，只需补充 tags 即可

> [!NOTE]
> 也可以给"插入模板"设置快捷键，在 **设置 → Hotkeys** 中搜索 `Templates: Insert template`，绑定一个顺手的快捷键（比如 `Alt + T`），以后一键插入。

