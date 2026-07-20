# 📊 ChartQA-X 三模型评测系统

> NLP 多模态大作业 — 论文图表问答评测系统

在 **AMD 云平台（ROCm）** 上对比三种技术路线在图表理解任务上的表现：

| 模型类型 | 代表模型 | 输入形式 | 视觉→文本融合方式 |
|----------|----------|----------|-------------------|
| **OCR+LLM 基线** | PaddleOCR + Qwen2.5-0.5B | 纯文本 | ❌ 无融合 |
| **外挂式多模态** | LLaVA 1.5-7B | 图片 + 问题 | ViT → 投影层 → LLM |
| **原生多模态** | Gemma 4 E4B | 图片 + 问题 | Patch Embedding 直接生成软 Token |

---

## 📁 项目结构

```
MtoNLP/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖清单
├── install.sh                   # 一键安装脚本（AMD ROCm）
├── run.sh                       # 一键运行脚本
├── src/                         # 源代码
│   ├── __init__.py              # 包初始化
│   ├── __main__.py              # 包入口
│   ├── config.py                # 统一配置管理
│   ├── data_prepare.py          # 数据下载与预处理
│   ├── batch_inference.py       # vLLM 批量推理（Gemma 4 + LLaVA）
│   ├── baseline_ocr_llm.py      # OCR + LLM 基线
│   ├── evaluator.py             # 评估指标计算
│   ├── error_analyzer.py        # 四类错误自动分类
│   ├── visualizer.py            # 结果可视化（柱状图/雷达图/热力图）
│   ├── visualize_tsne.py        # t-SNE 交融机制可视化
│   ├── gradio_app.py            # Gradio 演示界面
│   └── main.py                  # 主入口（全流程编排）
├── results/                     # 评测结果输出
│   ├── gemma4/                  # Gemma 4 结果
│   ├── llava/                   # LLaVA 结果
│   ├── baseline/                # 基线结果
│   └── comparison/              # 对比报告与图表
├── tsne_visualization/          # t-SNE 可视化输出
├── error_figures/               # 误差分析截图
├── logs/                        # 运行日志
└── models/                      # 模型文件（下载后）
```

---

## 🚀 快速开始

### 1. 环境要求

- **硬件**: AMD GPU（ROCm 7.2.1+），单卡即可
- **系统**: Linux（云平台 Ubuntu）
- **Python**: 3.10+
- **存储**: ~50GB（模型 ~30GB + 数据 ~1GB）

### 2. 安装

```bash
cd MtoNLP/
chmod +x install.sh run.sh
./install.sh
```

这会自动：
1. 创建目录结构
2. 安装 PyTorch ROCm 版本
3. 安装 vLLM ROCm 版本
4. 安装所有 Python 依赖
5. 下载 Gemma 4 E4B 和 LLaVA 1.5-7B 模型
6. 配置环境变量

### 3. 启动 vLLM 服务

**需要两个终端**：

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
# 快速测试（基线模型，5 个样本）
./run.sh test

# 快速评测（所有模型，10 个样本）
./run.sh quick

# 完整评测（所有模型，全部样本）
./run.sh full

# 只跑单个模型
./run.sh gemma
./run.sh llava
./run.sh baseline

# 包含 t-SNE 交融机制分析
./run.sh --tsne

# 启动 Gradio 演示界面
./run.sh gradio
```

---

## 📊 评测流程

```
数据准备 → 模型推理 → 指标计算 → 误差分析 → 可视化
                                    ↓
                          t-SNE 交融机制分析（可选）
```

### 评测指标

| 指标 | 说明 |
|------|------|
| **Exact Match** | 预测与标准答案完全一致 |
| **Contains Match** | 预测包含标准答案或反之 |
| **Avg F1** | 词级 F1 分数的平均值 |

### 四类错误分析

| 错误类型 | 定义 | 典型例子 |
|----------|------|----------|
| 文字识别错误 | OCR/模型看错了数字/文字 | "2019" → "2018" |
| 视觉定位失败 | 模型未关注到正确区域 | 问"1月"却关注"2月" |
| 推理错误 | 识别正确但逻辑出错 | 数值对但单位错 |
| 模型幻觉 | 生成图表中不存在的信息 | 编造不存在的年份 |

---

## 🌐 Gradio 演示

```bash
python -m src.gradio_app
# 或
./run.sh gradio
```

打开浏览器访问 `http://localhost:7860`，可以：
- 上传论文图表
- 输入问题
- 选择模型（Gemma 4 / LLaVA / OCR+LLM）
- 对比不同模型的回答

---

## 🧬 交融机制可视化（t-SNE）

```bash
python -m src.visualize_tsne
# 或
./run.sh tsne
```

生成：
- LLaVA 各层 t-SNE 图（视觉 Token vs 文本 Token 分布）
- Gemma 4 各层 t-SNE 图
- 跨模态相似度随层数变化曲线
- 分析报告

**关键发现**：
- **LLaVA**：低层视觉/文本 Token 分离 → 高层逐渐混合（融合发生在投影层之后）
- **Gemma 4**：输入层已经较接近 → 各层进一步缩小差距（原生统一空间）

---

## 🔧 配置说明

所有配置集中在 [src/config.py](src/config.py) 中：

```python
# 路径（可通过环境变量覆盖）
WORKSPACE = "/network-workspace"
RESULTS_DIR = f"{WORKSPACE}/results"

# 模型配置
MODELS = {
    "gemma4": {"api_base": "http://localhost:8000/v1", ...},
    "llava":  {"api_base": "http://localhost:8001/v1", ...},
    "baseline": {"model_type": "baseline", ...},
}

# GPU 自适应并发
MAX_WORKERS = get_optimal_workers()  # 根据显存自动调整
```

---

## 📋 命令行参考

```
python -m src.main [选项]

选项:
  --models MODEL [MODEL ...]   选择模型 (gemma4, llava, baseline)
  --max-samples N              限制样本数量
  --skip-data                  跳过数据准备
  --verbose                    详细日志
  --tsne                       运行 t-SNE 分析
```

---

## 📝 分工说明

| 编号 | 角色 | 对应文件 | 说明 |
|------|------|----------|------|
| 1 | 数据负责人 | `data_prepare.py` | 下载/处理 ChartQA-X |
| 2 | 基线负责人 | `baseline_ocr_llm.py` | OCR+LLM 部署与推理 |
| 3 | 外挂式模型 | `batch_inference.py` (LLaVA) | LLaVA 部署与推理 |
| 4 | 原生模型 | `batch_inference.py` (Gemma 4) | Gemma 4 部署与推理 |
| 5 | 评测分析 | `evaluator.py`, `error_analyzer.py` | 指标计算 + 误差分类 |
| 6 | 可视化 | `visualizer.py`, `visualize_tsne.py` | t-SNE + 图表生成 |
| 7 | 系统汇报 | `gradio_app.py`, `main.py` | Gradio 界面 + 全流程 |

---

## ⚠️ 注意事项

1. **云平台特性**：关闭实例后虚拟环境会重置，务必用 `install.sh` 保存环境配置。
2. **显存不足**：使用 `--max-model-len 8192` 或更小值减少 vLLM 显存占用。
3. **断点续传**：推理结果实时保存，中断后重新运行会自动跳过已完成样本。
4. **t-SNE 内存**：加载完整 7B 模型约需 14GB 显存，若不足可先只运行 baseline。
5. **端口占用**：确保 8000/8001 端口未被占用后再启动 vLLM。

---

## 📖 参考资料

- [ChartQA-X 数据集](https://huggingface.co/datasets/shamanthakhegde/ChartQA-X)
- [ChartQA-X 论文](https://ar5iv.labs.arxiv.org/html/2504.13275)
- [Gemma 4 官方文档](https://ai.google.dev/gemma)
- [LLaVA 论文](https://llava-vl.github.io/)
- [vLLM 文档](https://docs.vllm.ai/)
- [AMD ROCm 开发指南](https://github.com/datawhalechina/hello-rocm)
