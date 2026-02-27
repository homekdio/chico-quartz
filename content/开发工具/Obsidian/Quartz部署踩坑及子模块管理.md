---
title: Quartz 部署踩坑及子模块管理
tags:
  - Quartz
  - Git
  - 博客搭建
---

# Quartz 部署踩坑及子模块管理

最近在折腾 Quartz 博客时，因为添加自定义组件（比如 Profile）导致远程部署一直报错。折腾了一圈才发现，带子模块（Submodule）的项目坑确实不少。记录一下这次的排错过程和 Git 逻辑，省得下次再犯。

## 1. 现象：本地 Build 正常，云端报错

**报错信息：**
`Import "Profile" will always be undefined because there is no matching export in "quartz/components/index.ts"`

明明本地预览好好的，一推到 GitHub 触发 Action 就直接崩溃。

## 2. 深度排错：子模块是“独立仓库”

其实核心原因就三个：

1. **子模块没提交**：自定义的 `Profile.tsx` 是在 `quartz/` 文件夹里新建的。在主目录执行 `git add .` 根本**管不到**子模块内部。
2. **远程地址不对**：`.gitmodules` 里的 URL 还是原作者的，云端构建时会按这个地址去原作者仓库找 Commit。
3. **没 Push 到云端**：本地子模块改好了没推，云端仓库自然搜不到那个版本号。

### 总结一套复现流程（正确做法）

不管是改样式还是加组件，只要动了 `quartz/` 下的东西，必须按这个顺序：

1. **进入子模块：** `cd quartz`
2. **子模块内提交通道：** `git add .` -> `git commit -m "xxx"`
3. **一定要 Push 子模块：** `git push origin HEAD:refs/heads/v4`（注意分支名）
4. **回到主目录：** `cd ..`
5. **更新主仓库指针：** `git add quartz` -> `git commit` -> `git push`

> [!IMPORTANT]
> 主仓库只记录一个名为“子模块指针”的 Commit ID，并不包含子模块里的代码。所以必须先有子模块的远程更新，主仓库的引用才有意义。

## 3. 关于 main 和 v4 分支的区别

在 GitHub 仓库里经常看到这两个分支，容易搞混：

| 分支 | 角色 | 作用 |
| --- | --- | --- |
| **main** | 你的家 | 存放你写的 content 内容、私人配置、样式自定义。GitHub Actions 监听这个分支。 |
| **v4** | Quartz 骨架 | 官方框架的标准分支。平时不用切，但子模块的修复通常是推到这个分支。 |

**一句话总结：** 写文章、改样式认准 `main` 即可。那个“比较与拉取请求”的黄色提示，如果不打算合并原作者的新功能，完全可以忽略。

## 4. 常见的 Git 报错解释

* **fatal: not our ref...**：地址指到原作者仓库去了，里面没你的代码。
* **TypeError: (void 0) is not a function**：云端没拉到对应的 `.tsx` 文件，导致 import 成了空。

---
*Created: 2026-02-27*
