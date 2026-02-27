---
title: ContentIndex Plugin 解析
tags:
  - Quartz
  - ContentIndexPlugin
---

## ContentIndex Plugin 是干什么的？

它是 Quartz 核心的 **emitter (输出器)** 插件之一。它的主要职责是在构建流程的最后阶段，根据收集到的所有 note 内容，汇聚生成用于导航、搜索和订阅的支持性文件。

简单来说，如果把 Quartz 网站比作一本书，`ContentIndex` 就是负责印发 **目录 (Sitemap)**、**索引 (JSON)** 和 **更新摘要 (RSS)** 的部门。

## 核心功能：生成三个索引文件

``` typescript
Plugin.ContentIndex({

        enableSiteMap: true,

        enableRSS: true,

      }),
```

| 生成的文件                      | 用途                                                             | 由哪个配置控制               |
| :------------------------- | :------------------------------------------------------------- | :-------------------- |
| `sitemap.xml`              | **站点地图**。由 XML 编写，专门给 Google/Bing 等搜索引擎的爬虫使用，提升 SEO。           | `enableSiteMap: true` |
| `index.xml`                | **RSS feed**。符合 RSS 2.0 协议的摘要文件，供用户通过外部阅读器（如 NetNewsWire）订阅更新。 | `enableRSS: true`     |
| `static/contentIndex.json` | **全文搜索索引**。Quartz 网站自带的搜索框就是读取这个 JSON 文件来进行毫秒级的本地搜索。           | 始终生成                  |

