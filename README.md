# 📊 ChartQA-X 三模型评测系统

> NLP 多模态大作业 — 论文图表问答评测系统

在 **AMD 云平台（ROCm）** 上对比三种技术路线在图表理解任务上的表现，
重点分析多模态信息的 **"交融机制"**。

| 模型类型 | 代表模型 | 输入形式 | 视觉→文本融合方式 | 交融发生层级 |
|----------|----------|----------|-------------------|-------------|
| **OCR+LLM 基线** | PaddleOCR + Qwen2.5-0.5B | 纯文本（OCR 提取） | ❌ 无融合 | 无 |
| **外挂式多模态** | LLaVA 1.5-7B | 图片 + 问题 | ViT → 投影层 → LLM | **投影层**（显式对齐） |
| **原生多模态** | Gemma 4 E4B | 图片 + 问题 | Patch Embedding 软Token | **输入层**（隐式统一空间） |

---

## 📁 项目结构

```
MtoNLP/
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖清单
├── install.sh                    # 一键安装（AMD ROCm 自动检测）
├── run.sh                        # 一键运行（支持 test/quick/full/tsne/gradio）
│
├── src/                          # 源代码
│   ├── __init__.py               # 包初始化 + 公共 API 导出
│   ├── __main__.py               # 支持 python -m src 启动
│   ├── config.py                 # 统一配置（路径/模型/GPU/日志）
│   ├── data_prepare.py           # 步骤1: 数据集下载与预处理
│   ├── batch_inference.py        # 步骤2: vLLM 批量推理（Gemma4 / LLaVA）
│   ├── baseline_ocr_llm.py       # 步骤2: OCR + LLM 基线推理
│   ├── evaluator.py              # 步骤3: 评估指标（EM / Contains / F1）
│   ├── error_analyzer.py         # 步骤3: 四类错误自动分类 + 错误矩阵
│   ├── visualizer.py             # 步骤3: 可视化（柱状图/雷达图/热力图）
│   ├── visualize_tsne.py         # 步骤3: t-SNE 交融机制可视化
│   ├── gradio_app.py             # 步骤4: Gradio Web 演示界面
│   └── main.py                   # 主入口（全流程编排，步骤1→5）
│
├── results/                      # 评测结果输出
│   ├── gemma4/                   # Gemma 4 推理结果 + 评估报告 + 误差分析
│   ├── llava/                    # LLaVA 推理结果 + 评估报告 + 误差分析
│   ├── baseline/                 # 基线推理结果 + 评估报告 + 误差分析
│   └── comparison/               # 多模型对比报告 + 图表 + 错误矩阵
│
├── tsne_visualization/           # t-SNE 可视化输出
├── error_figures/                # 误差分析截图
├── logs/                         # 运行日志（含时间戳）
└── models/                       # 模型文件（install.sh 自动下载）
```

---

## 🚀 快速开始

### 1. 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | AMD GPU（ROCm 7.x），单卡 ≥ 16 GB 显存 |
| 系统 | Linux（云平台 Ubuntu） |
| Python | 3.10+ |
| 存储 | ~50GB（模型 ~30GB + 数据 ~1GB） |

### 2. 一键安装

```bash
cd MtoNLP/
chmod +x install.sh run.sh
./install.sh
```

脚本自动完成：目录创建 → ROCm 版本检测 → PyTorch/vLLM 安装 → 依赖安装 → 模型下载 → 环境变量配置。

### 3. 启动 vLLM 服务（两个终端）

```bash
# 终端 1 — Gemma 4 E4B
vllm serve ./models/google/gemma-4-E4B-it/ \
    --served-model-name gemma-4-E4B-it \
    --port 8000 --max-model-len 8192

# 终端 2 — LLaVA 1.5-7B
vllm serve ./models/swift/llava-1.5-7b-hf \
    --served-model-name llava-1.5-7b-hf \
    --port 8001 --max-model-len 8192
```

看到 `Application startup complete.` 即启动成功。

### 4. 运行评测

```bash
./run.sh test              # 🔬 测试模式（基线模型，5个样本）
./run.sh quick             # ⚡ 快速评测（所有模型，10个样本）
./run.sh full              # 🚀 完整评测（所有模型，全部样本）

./run.sh gemma             # 只跑 Gemma 4
./run.sh llava             # 只跑 LLaVA
./run.sh baseline          # 只跑 OCR+LLM 基线
```

### 5. 可视化与分析

```bash
./run.sh eval              # 已有推理结果 → 只运行评测+误差分析+图表
./run.sh viz               # 只生成可视化图表
./run.sh tsne --max-samples 30              # t-SNE 交融机制分析
./run.sh tsne --max-samples 10 --skip-llava # 仅 Gemma 4 t-SNE
```

### 6. Gradio 演示

```bash
./run.sh gradio                         # 默认 http://0.0.0.0:7860
./run.sh gradio --port 8080 --share     # 自定义端口 + 公网链接
```

---

## 📊 评测流程

```
数据准备 → 模型推理 → 指标计算 → 误差分析 → 可视化
                 ↓              ↓
           t-SNE 交融分析  错误矩阵 + 典型案例
```

### 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match** | 预测与标准答案完全一致 |
| **Contains Match** | 预测包含标准答案或反之 |
| **Avg F1** | 词级 F1 分数的平均值 |

### 四类错误自动分类

| 错误类型 | 判断逻辑 | 置信度 |
|----------|----------|--------|
| **OCR_Error**（文字识别） | 编辑距离 ≤ 2 且长度相近 | 高 |
| **Visual_Localization_Failure**（视觉定位） | 预测关键词在图中 OCR 文本中存在但不在标准答案中 | 中 |
| **Reasoning_Error**（推理错误） | 共享数值但最终答案不一致 | 中 |
| **Hallucination**（幻觉） | 预测数字/实体在图中 OCR 文本中完全不存在 | 中-高 |

> **关键设计**：`error_analyzer.py` 内置 80+ 英文停用词过滤，避免 "the"/"is" 等常见词被误判为幻觉。
>
> 错误分析依赖 PaddleOCR 提取图表文字用于交叉验证；OCR 失败时自动回退到纯文本规则。

---

## 🔧 CLI 完整参考

```bash
python -m src.main [选项]

选项:
  --models MODEL [MODEL ...]   选择模型（gemma4, llava, baseline）
                               默认: 全部三个
  --max-samples N              限制样本数（快速测试用）
  --skip-data                  跳过数据准备步骤
  --verbose                    详细日志输出（DEBUG 级别）
  --no-resume                  禁用断点续传，重新推理所有样本
  --tsne                       额外运行 t-SNE 交融机制分析
  --version                    显示版本号
```

示例：
```bash
# 基线模型 + 5 样本快速验证
python -m src.main --models baseline --max-samples 5 --verbose

# 完整评测 + t-SNE
python -m src.main --tsne

# 禁用断点续传，从头重跑
python -m src.main --no-resume --max-samples 50
```

---

## 🧬 交融机制可视化（t-SNE）

**这是项目的核心创新点**，用于对比 LLaVA（外挂式）和 Gemma 4（原生多模态）中视觉 Token 与文本 Token 的融合方式。

### 分析方法

1. 从 ChartQA-X 抽取 20-50 个样本，简单/复杂问题各半
2. 对两个模型的统一 9 层（0, 4, 8, 12, 16, 20, 24, 28, 32）提取 hidden states
3. 取视觉/文本 Token 的层平均向量，t-SNE 降维到 2D
4. 计算跨模态余弦相似度随层数变化曲线
5. 自动标注"交融起始层"（相似度首次快速上升的层）

### ⚠️ 显存安全

- 两个 7B 模型**绝不会同时驻留显存**：每提取完一个模型的特征后立即 `del model` + `torch.cuda.empty_cache()`
- 峰值显存 ≈ 14GB（单模型加载），提取完成后释放
- 若显存不足 16GB：`./run.sh tsne --skip-llava --max-samples 10`

### 输出

| 文件 | 说明 |
|------|------|
| `tsne_comparison_matrix.png` | 3×2 矩阵图（LLaVA/Gemma 4 × Layer 0/12/32） |
| `cross_modal_similarity.png` | 相似度曲线 + 交融起始层标注 |
| `alignment_analysis_report.md` | 分析报告（含融合机制解读） |

---

## 🌐 Gradio 演示界面

支持：
- 📤 上传论文图表（png/jpg）
- ❓ 输入自然语言问题
- 🔧 **多选模型**同时对比（Gemma 4 / LLaVA / OCR+LLM）
- ⏱️ 每个模型独立显示回答 + 推理耗时
- 📊 自动读取并展示批量评测结果
- 💡 从数据集中随机选取示例图表

> **注**：多模型选中时推理**串行执行**，避免 GPU 资源争抢。

---

## ⚠️ 注意事项

| 场景 | 建议 |
|------|------|
| 云平台实例关闭 | 虚拟环境重置，用 `./install.sh` 重建 |
| 显存不足 | vLLM 加 `--max-model-len 4096`；t-SNE 加 `--skip-llava` |
| 端口被占用 | 检查 8000/8001，或修改 `config.py` 中 `api_base` |
| 首次运行 PaddleOCR | 会自动下载模型到 `~/.paddleocr/`，需网络畅通 |
| 断点续传 | 默认开启；结果实时保存，中断后自动跳过已完成样本 |
| 数字答案匹配 | `normalize_answer` 保留数字中的小数点（3.14 不会被错误标准化为 314） |
| GPU 并发 | `VLLM_CONCURRENT` 默认 ≤2，避免 vLLM 服务端过载 |

---

## 📋 分工对应

| 编号 | 角色 | 对应文件 |
|------|------|----------|
| 1 | 数据负责人 | `data_prepare.py` |
| 2 | 基线负责人 | `baseline_ocr_llm.py` |
| 3 | 外挂式模型 | `batch_inference.py`（LLaVA 部分） |
| 4 | 原生模型 | `batch_inference.py`（Gemma 4 部分） |
| 5 | 评测分析 | `evaluator.py` + `error_analyzer.py` |
| 6 | 交融可视化 | `visualize_tsne.py` + `visualizer.py` |
| 7 | 系统与汇报 | `gradio_app.py` + `main.py` + 报告整合 |

---

## 📖 参考资料

- [ChartQA-X 数据集](https://huggingface.co/datasets/shamanthakhegde/ChartQA-X)
- [ChartQA-X 论文](https://ar5iv.labs.arxiv.org/html/2504.13275)
- [Gemma 4 官方文档](https://ai.google.dev/gemma)
- [LLaVA 论文](https://llava-vl.github.io/)
- [vLLM 文档](https://docs.vllm.ai/)
- [AMD ROCm 入门](https://github.com/datawhalechina/hello-rocm)
- [AMD 云平台部署 Gemma 4 教程](在AMD云平台部署&运行%20Gemma4%20大模型.md)
