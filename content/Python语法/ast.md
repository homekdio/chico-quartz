---
title: ast 模块
tags:
  - Python
---

## 简介

```python
import ast         # 就像一个"翻译机"，能把长得像字典的字符串（"{'a':1}"）安全地变成真正的字典对象
```

## 使用示例

```python
import ast
str = '{"a":1}'
data = ast.literal_eval(str)         # data['a'] == 1
```
