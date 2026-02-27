---
title: Quartz 部署踩坑及子模块管理
tags:
  - Quartz
  - Git
  - 博客搭建
---

# Quartz 部署踩坑及子模块管理

最近在折腾 Quartz 博客时，因为添加自定义组件（比如 Profile）导致远程部署一直报错。折腾了一圈才发现，带子模块（Submodule）的项目坑确实不少。记录一下这次的排错过程和 Git 逻辑，省得下次再犯。

## 1. 现象：本地预览正常，云端“面目全非”

常见的两个症状：

- **报错崩溃**：`Import "Profile" will always be undefined`。原因通常是远程找不到你的组件文件。
- **样式不生效 / 头像加载失败**：本地有头像，推上去就变成了破损图片，或者干脆没有自定义布局。

## 2. 深度排错：那些容易忽略的“大坑”

核心原因主要有这四个：

1. **子模块没提交**：自定义组件是在 `quartz/` 文件夹（子模块）里新建的。在主目录执行 `git add .` 根本**管不到**子模块内部。
2. **子模块远程地址没改**：`.gitmodules` 里的 URL 还是原作者的仓库。Actions 会跑到原作者那里找你的 Commit，结果必然报错 `not our ref`。
3. **两个 Layout 文件混淆**：Quartz 项目根目录和子模块内部都有 `quartz.layout.ts`。**Quartz 真正读取的是根目录那个**。改了子模块里的没改根目录的，远程就不会生效。
4. **子路径部署导致的路径偏移**：如果博客部署在 `username.github.io/repo/` 下，绝对路径 `/static/img.jpg` 会指向域名根目录导致图片 404。

## 3. 正确的同步流程（避坑指南）

不管是改样式还是加组件，只要动了 `quartz/` 下的东西，必须严格按这个顺序：

1. **同步子模块内容**：
   - `cd quartz`
   - 提交并推送：`git add .` -> `git commit` -> `git push origin HEAD:refs/heads/v4`
2. **同步主目录配置**：
   - 确保**根目录**下的 `quartz.layout.ts` 同步了最新的布局修改。
   - 确保**根目录**下的 `static/` 文件夹里放了你要用的头像图片（主仓库也需要这份资源）。
3. **处理代码中的图片路径**：
   - 在组件里使用 `resolveRelative(fileData.slug!, opts.avatar)` 来解析路径。
4. **更新主仓库引用**：
   - `cd ..`
   - `git add quartz quartz.layout.ts static/`
   - `git commit -m "feat: sync layout and assets"`
   - `git push`

## 4. 常见报错与修复总结

- **fatal: not our ref...**：子模块代码没 push，或者地址还是官方的。
- **图片显示不出来 (Avatar)**：绝对路径在子路径下失效，改用 Quartz 提供的相对路径解析函数。
- **改了代码没效果**：重点检查根目录下的 `quartz.layout.ts` 是否已经更新。

---
*Created: 2026-02-27*
