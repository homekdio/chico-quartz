// @ts-ignore
import { PageLayout, SharedLayout } from "./quartz/cfg"
// @ts-ignore
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/sosiristseng/template-quartz",
    },
  }),
}

// components for the homepage (blog style)
export const defaultHomePageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.Profile(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.DesktopOnly(Component.Spacer()),
    Component.Explorer({ title: "资源管理器" }),
  ],
  right: [
    Component.RecentNotes({ title: "📝最近笔记", limit: 5, showTags: false, showOnlyOnIndex: true }),
    Component.Backlinks(),
  ],
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.Profile(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.DesktopOnly(Component.Spacer()),
    Component.Explorer({ title: "笔记目录" }),
  ],
  right: [
    Component.DesktopOnly(Component.RecentNotes({ title: "📝最近笔记", limit: 5, showTags: false, showOnlyOnIndex: true })),
    Component.DesktopOnly(Component.TableOfContents()),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.Profile(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.DesktopOnly(Component.Spacer()),
    Component.Explorer({ title: "资源管理器" }),
  ],
  right: [],
}
