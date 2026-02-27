---
title: Quartz 主页布局自定义与组件开发记录
date: 2026-02-27
tags:
  - Quartz
  - CSS
  - 博客搭建
---

# Quartz 主页布局自定义与组件开发记录

本文记录基于 Quartz v4 修改博客默认侧边栏布局的过程：包括添加自定义个人名片、调整 Flexbox 布局以放大 Explorer (文件树) 区域、以及精简右侧边栏组件。

核心理解：

* **TypeScript (`.ts` / `.tsx`)**：负责组件结构和页面布局顺序。
* **CSS / SCSS (`.scss`)**：控制元素样式、尺寸与排版分布。

---

## 改造目标与思路

1. **左上角**：用包含头像、名字、简介和外部链接的自定义名片替换默认的 Site Title。
2. **左下角**：让由于内容限制缩在角落的 Explorer (文件树) 变高，独占侧边栏的下半部分。
3. **右侧栏**：去除默认的拓扑图 (Graph)，仅保留“最近笔记”列表并隐藏具体的 tag 标签。

---

## 具体修改步骤

### 1. 调整根布局配置 (`quartz.layout.ts`)

`quartz/quartz.layout.ts` 是全局布局的入口文件。

**精简右侧栏：**
去掉 `Component.Graph()`，并修改“最近笔记”的参数屏蔽标签。

```typescript
// quartz.layout.ts (right 区域)
right: [
    // 设置 showTags: false 来隐藏文章自带的 tag
    Component.RecentNotes({ title: "最近笔记", limit: 5, showTags: false }), 
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
],
```

**引入自定义的个人名片：**
在 `left` 数组中，将原本的 `Component.PageTitle()` 替换为将要开发的 `Component.Profile()`。

### 2. 开发 Profile 组件

在 `quartz/quartz/components/` 目录下新建 `Profile.tsx`。

基本结构：

* 使用 `<img>` 展示头像
* `<h2>` 和 `<p>` 用于文本介绍
* `<a>` 结合 `<svg>` 实现 GitHub / Bilibili 等社交链接

相应的样式则跟在下方，为 `.profile-card` 添加合适的边距(`padding`, `margin`)、限制头像的宽高(`width`, `height`) 等。通过压低 `margin-bottom` 和缩小字号把自身变得更加紧凑。

**注意：** 新增的组件必须要在 `quartz/quartz/components/index.ts` 中 `export` 出去，否则在 `layout.ts` 里面会引发报错。

### 3. 使用 Flexbox 撑起 Explorer

为了让底下的文件目录占满左侧栏剩余高度，需要解决父级到子集的 Flex 高度传递问题。

**步骤 1：让外框占据满屏高度**
编辑 `quartz/quartz/styles/base.scss`，使包围侧边栏的 `.sidebar.left` 使用 flex 垂直排列：

```scss
& .sidebar.left {
  display: flex;
  flex-direction: column;
  height: 100vh;
  /* ... */
}
```

**步骤 2：添加占位符 (Spacer) 隔离上下**
回到 `quartz.layout.ts`，在左侧栏的名片区和文件目录之间，塞入一个弹性的 `Spacer`。

```typescript
Component.DesktopOnly(Component.Spacer()),
Component.Explorer(),
```

这个 Spacer 会推开上下方的组件，强迫上部位靠顶，Explorer 靠底。

**步骤 3：解除 Explorer 本身的高度限制**
修改 `quartz/quartz/components/styles/explorer.scss`，设置 `.explorer` 的高度弹性：

```scss
.explorer {
  /* ... */
  min-height: 50vh; 
  height: 100%;
  flex: 1 1 auto;
}
```

经过以上三点联合作用，Explorer 就能完整接管左侧栏的下半块区域。

---

## 日常修改指南（给自己的备忘录）

后续如果还需要自定义样式或位置，主要遵循以下四个操作流：

1. **改布局查 `quartz.layout.ts`**：
   想换组件的上下次序，或者增加/隐藏模块，直接对 `left` 或 `right` 数组做增减。

2. **改结构找 `quartz/quartz/components/`**：
   Quartz 内置的模块基本都在此路径下。想改名字或者换个 SVG 下拉图标，直接改对应的 `.tsx` 文件。

3. **调间距/颜色调 CSS/SCSS**：
   在组件对应的 `.scss` 或者组件底部的 css 文本域修改。多用浏览器 F12 的开发者工具定位到对应 Class 名字（比如 `.folder-icon`），如果是要它消失直接加上 `display: none;`。

4. **实时预览看效果**：
   调参时启动本地服务：

   ```bash
   npx quartz build --serve
   ```

   代码保存后，本地 `localhost:8080` 或开启的端口会自动刷新。边改 CSS 数值边看结果，能最高效地找到对齐和留白的最佳甜点。
