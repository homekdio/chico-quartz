---
title: 解决 Quartz 被 GitHub 解析成 XML
tags:
  - Quartz
  - GitHub-Pages
  - 踩坑记录
---

## 问题现象

部署 Quartz 到 GitHub Pages 后，访问网站根路径（如 `https://xxx.github.io/repo-name/`），浏览器不显示正常网页，而是展示**原始 RSS XML 文档**：
![[解决Quartz被github解析成XML示例图.png]]

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
    <channel>
      <title>Chico's Garden</title>
      ...
    </channel>
</rss>
```

## 原因分析

### 1. 缺少 `content/index.md`

Quartz 的路由映射规则是把 `content/` 下的 `.md` 文件按路径 1:1 编译为 HTML：

| Markdown 源文件            | 生成的 HTML 产物              |
| -------------------------- | ----------------------------- |
| `content/index.md`         | `public/index.html`           |
| `content/CV/简历.md`       | `public/CV/简历/index.html`   |

**如果 `content/` 下没有 `index.md`，Quartz 就不会生成根目录的 `index.html`。**

### 2. RSS feed 始终会生成

`quartz.config.ts` 中的 `ContentIndex` plugin 配置了 `enableRSS: true`：

```typescript
Plugin.ContentIndex({
  enableSiteMap: true,
  enableRSS: true,  // 这个选项让 Quartz 始终生成 index.xml
}),
```

所以无论有没有 `index.md`，根目录下都会生成一个 `index.xml`（RSS feed）。

### 3. GitHub Pages 的 fallback 机制

当用户访问一个目录 URL 时，GitHub Pages 会按优先级查找文件：

```
请求: https://xxx.github.io/repo-name/
  ↓
查找 index.html → ❌ 不存在（因为没有 index.md）
查找 index.xml  → ✅ 存在（ContentIndex plugin 生成的）
  ↓
返回 index.xml → 浏览器展示原始 XML
```

## 解决方案

在 `content/` 根目录下创建 `index.md` 作为网站首页：

```markdown
---
title: 网站标题
---

首页内容写在这里。
```

push 到 `main` 分支后，CI 重新构建时 Quartz 会生成 `index.html`，网站首页就能正常显示。

## 总结

### Web 服务器的 `index` 约定

`index` 这个名字来源于早期网站的"目录索引"概念，就像一本书的**目录页**。当用户访问一个目录 URL（如 `https://example.com/`）时，Web 服务器会自动查找该目录下的 `index.html` 来返回。这是一个几十年的 Web 标准约定，所有 Web 框架（Hugo、Jekyll、Next.js 等）和托管平台（GitHub Pages、Vercel、Netlify 等）都遵循这个约定。

### Quartz 的构建流程

整个问题的因果链如下：

```
content/index.md 不存在
       ↓
Quartz build 不会生成 public/index.html
       ↓
但 ContentIndex plugin（enableRSS: true）始终生成 public/index.xml
       ↓
GitHub Pages 在根路径找不到 index.html，fallback 到 index.xml
       ↓
浏览器收到 XML 内容，直接展示原始 XML 文档树
```

> [!IMPORTANT] 关键结论
> **`content/index.md` 是 Quartz 网站的首页入口**，必须存在。缺少它不会报错，但网站根路径会因为没有 `index.html` 而 fallback 到 RSS feed 的 `index.xml`。

### 排查思路

遇到类似"页面显示异常"的问题时，可以按以下步骤排查：

1. **确认 build 产物**：检查 `public/` 目录下是否生成了预期的 `index.html`
2. **检查 content 源文件**：对应路径下是否存在 `.md` 源文件
3. **检查 CI 日志**：GitHub Actions 的构建日志中是否有报错或警告
4. **检查 plugin 配置**：`quartz.config.ts` 中的 emitters 配置是否正确
