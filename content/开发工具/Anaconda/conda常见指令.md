---
title: conda常见指令
date: 2026-03-29 15:15
tags:
  - conda指令
---
# 一、基础查看

``` bash
# 查看 conda 版本 
conda --version 

# 查看所有已创建的环境 
conda env list 

# 查看当前环境里装了哪些包 
conda list 

```

# 二、环境管理

``` bash
# 创建新环境（指定 Python 版本）
conda create -n 环境名 python=3.10

# 激活环境（进入环境）
conda activate 环境名

# 退出当前环境
conda deactivate

# 删除环境（谨慎用）
conda remove -n 环境名 --all

# 复制环境
conda create -n 新环境名 --clone 旧环境名
```

# 三、包安装 / 卸载（装库）

``` bash
# 安装包
conda install 包名

# 安装指定版本
conda install 包名=版本号

# 卸载包
conda remove 包名

# 更新包
conda update 包名

# 更新所有包
conda update --all

# 用 pip 安装（conda 找不到时用）
pip install 包名
```
