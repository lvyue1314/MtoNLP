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
/workspace/repo/src/fine-tune/models/gemma4/   ← 项目根目录 (git clone 到这里)
├── README.md
├── install_deps.sh                # 安装额外依赖（4个包）
├── install_models.sh              # 下载模型到 /models
├── run.sh                         # 一键运行
│
├── src/                           # 源代码（同上）
│   ├── config.py                  # 统一配置（路径在此修改）
│   └── ...（其余 11 个 .py 文件）
│
├── results/                       # 评测结果
├── tsne_visualization/
├── error_figures/
└── logs/

/models/                           ← 模型文件（独立路径，项目盘空间不足）
├── google/gemma-4-E4B-it/
└── swift/llava-1.5-7b-hf/
```

> **路径说明**：项目本身在 `/workspace/repo/src/fine-tune/models/gemma4`，但模型文件因磁盘空间限制存放在 `/models`。

---

## 🚀 快速开始

### 1. 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | AMD GPU（ROCm 7.2.1），单卡 48 GB 显存 |
| 系统 | Linux（云平台 Ubuntu） |
| Python | 3.10+（云平台已预装） |
| 存储 | /models 分区 ≥ 40GB（两个模型约 30GB） |

### 2. 安装

```bash
# 克隆项目到云平台工作目录
cd /workspace/repo/src/fine-tune/models/gemma4
git clone <repo-url> 
cd ./MtoNLP

# 赋予执行权限
chmod +x install_deps.sh install_models.sh run.sh

# 步骤 A: 安装额外依赖（仅 4 个缺失的包）
./install_deps.sh

# 步骤 B: 下载模型文件到 /models
./install_models.sh
```

#### 依赖说明

| 来源 | 内容 |
|------|------|
| **云平台已预装** | torch 2.9.1, vllm 0.16.1, transformers 4.57.6, datasets 4.8.2, accelerate, fastapi, openai, matplotlib, seaborn, scikit-learn, pandas, numpy, Pillow, tqdm, sentencepiece, huggingface_hub 等 |
| **install_deps.sh 安装** | `paddlepaddle`, `paddleocr`, `gradio`, `modelscope`（仅 4 个） |

### 3. 启动 vLLM 服务（⚠️ 串行，不可同时运行）

> **显存限制**：48GB 无法同时跑两个 7B 模型。**先跑完一个 → 停止 → 再启动下一个**。
> **关键**：必须加 `--served-model-name`，否则 batch_inference 会报 404。

```bash
# ====== 第一步：Gemma 4 ======
# 终端 1：启动服务
vllm serve /models/google/gemma-4-E4B-it/ \
    --served-model-name gemma-4-E4B-it \
    --port 8000 \
    --max-model-len 4096

# 终端 2：等服务就绪（"Application startup complete."），运行
cd /workspace/repo/src/fine-tune/models/gemma4/MtoNLP
./run.sh --models gemma4 --no-resume

# Gemma4 跑完后 Ctrl+C 停服务

# ====== 第二步：LLaVA ======
# 终端 1：启动服务
vllm serve /models/swift/llava-1.5-7b-hf \
    --served-model-name llava-1.5-7b-hf \
    --port 8001 \
    --max-model-len 4096 \
    --trust-remote-code

# 终端 2：服务就绪后运行
./run.sh --models llava --no-resume

# LLaVA 跑完后 Ctrl+C 停服务

# ====== 基线不需要 vLLM，直接跑 ======
./run.sh --models baseline --no-resume
```

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
```

### 6. t-SNE 交融机制分析

```bash
# 全量 2500 样本（约 25 分钟）
python src/visualize_tsne.py --max-samples 2500

# 快速验证（30 样本，约 1 分钟）
python src/visualize_tsne.py --max-samples 30

# 打包
tar -czf tsne_results_v2.tar.gz tsne_visualization/
```

### 7. 结果打包下载

```bash
# 评测结果 + 环境快照
tar -czf final_results_v3.tar.gz results/ requirements_cloud_lock.txt

# t-SNE 结果
tar -czf tsne_results_v2.tar.gz tsne_visualization/

# 数据集图片
tar -czf chartqa_images.tar.gz LMUData/datasets/ChartQA/images/

# 带图片路径的逐题对比表
python3 -c "
import json, csv, re
g=json.load(open('results/gemma4/gemma4_results.json',encoding='utf-8'))
l=json.load(open('results/llava/llava_results.json',encoding='utf-8'))
b=json.load(open('results/baseline/baseline_results.json',encoding='utf-8'))
def cm(pred,gt):
    def norm(s):
        o=str(s).strip();t=o.lower()
        t=re.sub(r'(\d)\.(\d)',r'\1<DOT>\2',t)
        t=re.sub(r'[^a-z0-9\s]','',t)
        t=t.replace('<DOT>','.').strip()
        return t if t else o
    p,g=norm(pred),norm(gt)
    if not p or not g: return False
    return g in p or p in g
with open('results/comparison/all_results_with_images.csv','w',encoding='utf-8-sig') as f:
    f.write('id,image_path,question,answer,gemma4_pred,gemma4_correct,llava_pred,llava_correct,baseline_pred,baseline_correct\n')
    for i in range(2500):
        q=g[i]['question'].replace('\"','\"\"'); a=g[i]['answer'].replace('\"','\"\"')
        img=g[i]['image_path']
        gp=g[i]['predicted_answer'].replace('\"','\"\"').replace('\n',' ')
        lp=l[i]['predicted_answer'].replace('\"','\"\"').replace('\n',' ')
        bp=b[i]['predicted_answer'].replace('\"','\"\"').replace('\n',' ')
        f.write(f'{i},\"{img}\",\"{q}\",\"{a}\",\"{gp}\",{1 if cm(g[i][\"predicted_answer\"],g[i][\"answer\"]) else 0},\"{lp}\",{1 if cm(l[i][\"predicted_answer\"],l[i][\"answer\"]) else 0},\"{bp}\",{1 if cm(b[i][\"predicted_answer\"],b[i][\"answer\"]) else 0}\n')
print('Done')
"
```

### 8. Gradio 演示

#### 公网访问原理

云平台只有内网 IP，通过 frp 隧道将 Gradio 映射到公网：

```
云平台(内网)          HuggingFace中转          你的电脑(公网)
Gradio :7860 ←─frpc──→ frp服务器 ──→ xxx.gradio.live ──→ 浏览器
```

`frpc` 二进制已内置在仓库 `gradio/frpc_linux_amd64_v0.3` 中。

#### 首次部署（一次性）

```bash
mkdir -p /root/.cache/huggingface/gradio/frpc
cp gradio/frpc_linux_amd64_v0.3 /root/.cache/huggingface/gradio/frpc/
chmod +x /root/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3
uv pip install "gradio>=6.0" --no-cache
```

#### 启动

> 需要先启动至少一个 vLLM 服务（Gemma 4 或 LLaVA），基线无需服务。

```bash
python src/gradio_app.py --share
# → https://xxx.gradio.live（有效 1 周）
```

> **注意**：切换 vLLM 服务后需重启 Gradio（`Ctrl+C` 后重新 `python src/gradio_app.py --share`）。

### 9. 环境快照

```bash
pip freeze > requirements_cloud_final.txt
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

## ⚠️ 常见问题

### 模型推理

| 问题 | 原因 | 解决 |
|------|------|------|
| `model does not exist` 404 | vLLM 启动时未加 `--served-model-name` | 重启 vLLM 服务并加上该参数 |
| 推理结果全部 IMAGE_NOT_FOUND | 图片路径不对 | 检查 config.py 中 CHARTQA_DATA |
| 两个模型不能同时跑 | 48GB 显存不够 | 串行：Gemma4→停止→LLaVA |
| Gemma4 启动报 `model type gemma4 not recognized` | transformers 版本太旧 | `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/` 会自动升级 transformers |
| 推理速度慢 | 每张图 1.3 秒 | 正常速度，2500 条约 50 分钟 |

### 测评指标

| 问题 | 根因 | 解决 |
|------|------|------|
| 基线 CM=50.4% 虚高 | 中文"无法确定"被 normalize 成空串 → `"" in "3"` 返回 True | 已在 `evaluator.py:46` 修复，正确值 6.2% |
| 跨模型交叉分析不匹配 | `as_completed` 返回随机顺序 | 已在 `batch_inference.py:177` 修复，使用 indexed_results |
| 小数点被去除 (3.14→314) | `re.sub(r"[^a-z0-9\s]","",s)` | 已在 `evaluator.py:47` 修复 |

### t-SNE 分析

| 问题 | 原因 | 解决 |
|------|------|------|
| `n_iter` 参数报错 | sklearn 1.2+ 改名为 `max_iter` | 已修复 |
| 图表显示 "t-SNE Failed" | 多个原因（最常：perplexity 太高） | 已添加降级重试 + 错误日志 |
| Gemma4 图片加载失败 | torchvision ROCm 版无 PNG 支持 | 已改为 PIL 预加载 |
| LLaVA 图片加载失败 | processor API 参数顺序 | `images=` 放在 `text=` 前面 |

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
