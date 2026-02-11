# 基于生成对抗网络的图像转换系统设计与实现

**时间**：2025年02月 - 2025年05月
**角色**：核心负责人 (独立开发/后端算法/全栈)

### 项目简介 (Introduction)
针对无配对样本 (Unpaired Data) 条件下图像风格迁移困难的问题，基于 CycleGAN 模型构建了一套端到端的图像转换系统。该系统通过无监督学习算法实现了马与斑马、苹果与橙子等不同类别物体间的自然风格转换，探索并验证了 GAN 技术在计算机视觉领域的实际应用价值。

---

### 项目技术 (Core Tech Stack)
*   **深度学习框架**：Python, PyTorch, TorchOne (自研/Mock)
*   **核心模型**：CycleGAN, ResNet Generator (9-blocks), PatchGAN Discriminator
*   **训练策略**：Adam Optimizer, Instance Normalization, Cycle Consistency Loss
*   **工程化部署**：Gradio (Web UI), NumPy, PIL (Pillow)

---

### 项目描述 (Project Description)
系统采用**前后端分离**的分层架构设计：
1.  **后端算法层**：基于 PyTorch 框架封装 CycleGAN 推理引擎。针对官方源码进行了模块化重构，剔除冗余依赖，封装了模型加载、数据预处理及前向推理流程，实现了 CPU/GPU 自适应推理。
2.  **前端交互层**：利用 Gradio 开发图形化交互界面 (GUI)，集成图像上传预览、预训练模型实时切换、转换结果动态展示及一键保存等功能。
3.  **集成部署**：打通了深度学习模型与可视化应用的壁垒，实现了从“算法黑盒”到“可视应用”的转化。

---

### 核心工作 (Key Achievements)

#### 1. 模型架构复现与优化 (Replication & Optimization)
*   复现了基于 **ResNet** 结构的生成器网络，采用 **9个残差块 (Residual Blocks)** 作为核心转换单元，有效解决了深层网络的梯度消失问题，并在风格转换的同时保持了原图的内容结构。
*   使用了 **Instance Normalization (实例归一化)** 代替传统的 Batch Normalization，显著提升了单张图片风格迁移的质量和细节保留度。

#### 2. 工程化封装与解耦 (Engineering & Refactoring)
*   深入解析并重构了 CycleGAN 官方源码，将高度耦合的 `TestModel` 拆解为轻量级的推理接口。
*   实现了模型的**热拔插设计**：通过封装 `CycleGANInference` 类，支持在运行时动态加载 `horse2zebra` 或 `apple2orange` 等不同领域的权重文件，无需重启服务。
*   优化了图像预处理流水线 (`Resize` -> `ToTensor` -> `Normalize`)，确保输入数据分布与训练时一致，保证了推理精度。

#### 3. 可视化交互系统开发 (Interactive UI)
*   基于 Gradio 搭建了 Web 演示系统，设计了简洁直观的用户界面。
*   解决了 PyTorch Tensor 数据与前端图像格式 (`numpy.uint8`) 的转换问题，实现了毫秒级的推理响应与展示。
*   通过 CSS 自定义样式优化了界面视觉体验（如按钮交互效果），提升了系统的完成度和演示效果。
