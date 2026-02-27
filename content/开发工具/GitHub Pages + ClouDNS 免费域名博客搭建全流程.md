---
title: GitHub Pages绑定自己的域GitHub Pages + ClouDNS 免费域名博客搭建全流程名
date: 2026-02-27 22:24
tags:
  -
---
## 📋 项目背景
- **目标**：将 GitHub 上的 Quartz 博客绑定到免费自定义域名 `chico.abrdns.com`。
- **平台**：
  - 代码托管：GitHub Pages (`homekdio.github.io`)
  - 域名服务：ClouDNS (免费二级域名 `.abrdns.com`)
- **最终状态**：✅ 配置成功，HTTPS 证书已激活，网站可正常访问。

---

## ⚠️ 遇到的问题与解决方案

### 1. 仓库命名与目录结构误区
- **问题描述**：
  最初使用的项目仓库名为 `chico-quartz`，导致 GitHub Pages 生成的默认地址带有子目录（`.../chico-quartz/`）。如果直接绑定根域名，会导致网页 404 或样式丢失（CSS/JS 路径错误）。
- **解决方案**：
  - 更改项目名称：把原理的项目名改成`homekdio.github.io` 

### 2. DNS 记录类型选择冲突 (CNAME vs A Record)
- **问题描述**：
  尝试为根域名 `chico.abrdns.com` 添加 `CNAME` 记录指向 `homekdio.github.io` 时，DNS 服务商报错：“根域名不能是 CNAME 记录”。
- ![[Pasted image 20260227223324.png]]
  - *原因*：DNS 协议规定根域名（@）不能设置 CNAME，否则会影响 MX（邮件）等其他记录的解析。
- **解决方案**：
  - **改用 A 记录**：放弃 CNAME，改为添加 **4 条 A 记录**，将根域名直接指向 GitHub Pages 的官方 IP 地址。
  - **IP 列表**：
    ```text
    185.199.108.153
    185.199.109.153
    185.199.110.153
    185.199.111.153
    ```
- ![[Pasted image 20260227223228.png]]

### 3. www 子域名配置缺失
- **问题描述**：
  根域名配置成功后，GitHub 提示 `www.chico.abrdns.com` 配置不正确，导致带 `www` 的访问无法跳转或报错。
- ![[Pasted image 20260227222940.png]]
- **解决方案**：
  - **补充 CNAME 记录**：专门为 `www` 子域名添加一条 `CNAME` 记录。
  - **配置详情**：
    - 主机：`www`
    - 指向到：`homekdio.github.io`

  - ![[Pasted image 20260227223114.png]]
### 4. DNS 生效延迟与 HTTPS 证书签发
- **问题描述**：
  配置完成后，GitHub 后台立即点击“检查”显示“DNS 检查失败”或“未解析至服务器”。
  - *原因*：全球 DNS 刷新需要时间（TTL），并非即时生效。
  - ![[Pasted image 20260227223459.png]]
- **解决方案**：
  - **耐心等待**：等待 5~30 分钟让 DNS 全球传播。
  - **验证工具**：使用 [dnschecker.org](https://dnschecker.org) 查询 A 记录是否已在全球大部分地区生效。
  - **HTTPS 证书**：DNS 生效后，GitHub 会自动申请 TLS 证书，过程可能需要额外 10~30 分钟。期间访问可能会提示“不安全”，刷新或等待即可自动恢复。

---

## 🛠️ 最终配置清单 (可直接抄作业)

### 1. GitHub 仓库设置
- **仓库名**：`homekdio.github.io`
- **Custom Domain**：`chico.abrdns.com` (保存后会自动生成 CNAME 文件)
- **Enforce HTTPS**：证书生效后勾选此项（强制 HTTPS）

### 2. ClouDNS DNS 解析记录

| 类型 | 主机 (Host) | 值 (Value/Points to) | 说明 |
| :--- | :--- | :--- | :--- |
| **A** | `@` (或留空) | `185.199.108.153` | 根域名解析 IP 1 |
| **A** | `@` (或留空) | `185.199.109.153` | 根域名解析 IP 2 |
| **A** | `@` (或留空) | `185.199.110.153` | 根域名解析 IP 3 |
| **A** | `@` (或留空) | `185.199.111.153` | 根域名解析 IP 4 |
| **CNAME**| `www` | `homekdio.github.io` | 子域名跳转 |

---

## 💡 关键知识点总结

1.  **用户主页 vs 项目主页**：
    - `username.github.io` 仓库 = 用户主页 (根目录访问，无子路径)。
    - `other-repo` 仓库 = 项目主页 (需带 `/repo-name/` 子路径)。
    - **结论**：绑定独立域名务必使用 `username.github.io` 格式。

2.  **根域名解析规则**：
    - 根域名 (`@`) 只能用 **A 记录** (指向 IP)。
    - 子域名 (`www`, `blog` 等) 可以用 **CNAME 记录** (指向域名)。

3.  **安全性**：
    - GitHub Pages 免费仓库必须是 **Public (公开)** 的。
    - 静态博客代码公开是安全的（不含后端密钥），且前端代码本身在浏览器端就是可见的。

---

## 🎉 成果展示
- **访问地址**：[https://chico.abrdns.com](https://chico.abrdns.com)
- **状态**：✅ 运行正常，HTTPS 安全加密，无样式丢失，URL 干净简洁。
```