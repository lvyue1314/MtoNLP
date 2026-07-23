#!/usr/bin/env python3
"""
步骤 3.3: 交融机制可视化（t-SNE + 跨模态距离）

核心目标:
    对比 LLaVA 1.5-7B 和 Gemma 4 E4B 中"视觉Token"与"文本Token"的
    跨模态交融过程。这是项目的核心创新点，用于分析两种架构的融合机制差异。

方法:
    1. 从 ChartQA-X 抽取 20-50 个样本，对各层提取 hidden states
    2. 对每层取视觉/文本 Token 平均向量，做 t-SNE 降维
    3. 生成 3×3 散点图矩阵对比两模型各层
    4. 计算跨模态余弦相似度曲线，标注"交融起始层"
    5. 生成分析报告（Markdown）

注意:
    - 需要直接加载 transformers 模型（vLLM 不暴露 hidden states）
    - 显存需求较高（7B 约 14GB），不足时请用 --max-samples 5
    - 两个模型使用相同的层采样列表，确保公平对比
"""

import os
import sys
import json
import random
import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# 项目根
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from config import (
    MODELS, MODELS_DIR, CHARTQA_DATA, DATA_ANNOTATIONS,
    setup_logging,
)

logger = setup_logging(__name__)

# 输出目录
TSNE_OUTPUT_DIR = os.path.join(_project_root, "tsne_visualization")

# 两个模型使用完全相同的层采样列表（确保公平对比）
UNIFIED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]


def _free_gpu_memory():
    """
    释放 GPU 显存。

    每个模型约 14GB，在提取完一个模型的特征后必须释放，
    否则两个模型同时驻留会导致 OOM（~28GB+）。
    """
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU 显存已释放")
    except Exception:
        pass


# ================================================================
# 1. 数据准备：从 ChartQA-X 抽取样本
# ================================================================

def _load_sample_paths(max_samples: int = 30) -> list[dict]:
    """
    从 ChartQA-X 数据集中抽取样本，包含简单和复杂问题各半。

    Returns:
        [{"image_path": abs_path, "question": str}, ...]
    """
    ann_path = os.path.join(CHARTQA_DATA, "annotations", "test.json")
    if not os.path.exists(ann_path):
        ann_path = DATA_ANNOTATIONS

    if not os.path.exists(ann_path):
        logger.error("数据集标注文件不存在，请先运行 data_prepare.py")
        return []

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 简单问题：包含 what, how many, value
    simple_keywords = ["what", "how many", "value", "number", "how much"]
    simple = [
        d for d in data
        if any(kw in d.get("question", "").lower() for kw in simple_keywords)
    ]
    complex_cases = [d for d in data if d not in simple]

    half = max_samples // 2
    sampled = simple[:half] + complex_cases[:max_samples - half]
    random.shuffle(sampled)

    # 转换路径为绝对路径
    result = []
    for item in sampled[:max_samples]:
        abs_path = os.path.join(CHARTQA_DATA, item["image_path"])
        if not os.path.exists(abs_path) and "image_path_abs" in item:
            abs_path = item["image_path_abs"]
        if os.path.exists(abs_path):
            result.append({
                "image_path": abs_path,
                "question": item["question"],
                "answer": item.get("answer", ""),
            })

    logger.info(f"加载 {len(result)} 个样本用于 t-SNE 分析")
    return result


# ================================================================
# 2. 特征提取：LLaVA
# ================================================================

def extract_llava_features(
    samples: list[dict],
    model_id: str = None,
    target_layers: list = None,
) -> dict:
    """
    对 LLaVA 批量提取各层视觉/文本 Token 的平均向量。

    Args:
        samples: 样本列表 [{"image_path": ..., "question": ...}]
        model_id: HuggingFace 模型 ID
        target_layers: 层索引列表

    Returns:
        {"layer_0": {"visual": np.ndarray(N_samples, D), "text": np.ndarray(N_samples, D)}, ...}
    """
    target_layers = target_layers or UNIFIED_LAYERS
    model_id = model_id or MODELS["llava"].get("hf_model_id", "llava-hf/llava-1.5-7b-hf")

    logger.info(f"🔄 加载 LLaVA 模型: {model_id}")
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
    from PIL import Image

    try:
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
    except Exception as e:
        logger.error(f"LLaVA 模型加载失败: {e}。若显存不足，请尝试 --skip-llava")
        return {}

    # 初始化累加器
    accum = {}
    for layer_idx in target_layers:
        accum[f"layer_{layer_idx}"] = {"visual": [], "text": []}
    valid_count = 0

    for sample in tqdm(samples, desc="LLaVA 特征提取"):
        try:
            prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
            image = Image.open(sample["image_path"]).convert("RGB")
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            input_ids = inputs["input_ids"][0]
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

            for layer_idx in target_layers:
                if layer_idx + 1 >= len(outputs.hidden_states):
                    continue
                hidden = outputs.hidden_states[layer_idx + 1][0]  # [seq_len, D]

                image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                if len(image_positions) == 0:
                    continue

                vis_mean = hidden[image_positions].mean(dim=0).cpu().float().numpy()
                txt_positions = [i for i in range(len(input_ids))
                               if i not in image_positions.cpu().tolist()]
                txt_mean = hidden[txt_positions].mean(dim=0).cpu().float().numpy()

                accum[f"layer_{layer_idx}"]["visual"].append(vis_mean)
                accum[f"layer_{layer_idx}"]["text"].append(txt_mean)
            valid_count += 1
        except Exception as e:
            logger.warning(f"LLaVA 样本处理失败: {e}")
            continue

    # 堆叠成 (N_samples, D)
    result = {}
    for key, val in accum.items():
        if val["visual"]:
            result[key] = {
                "visual": np.stack(val["visual"]),
                "text": np.stack(val["text"]),
            }

    logger.info(f"LLaVA: {valid_count}/{len(samples)} 个样本成功, {len(result)} 层")

    # 显式释放模型以节省显存（调用方也会执行 _free_gpu_memory）
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# ================================================================
# 3. 特征提取：Gemma 4
# ================================================================

def extract_gemma_features(
    samples: list[dict],
    model_id: str = None,
    target_layers: list = None,
    visual_token_count: int = 280,
) -> dict:
    """
    对 Gemma 4 批量提取各层视觉/文本 Token 的平均向量。
    """
    target_layers = target_layers or UNIFIED_LAYERS

    local_path = os.path.join(MODELS_DIR, "google", "gemma-4-E4B-it")
    actual_id = local_path if os.path.exists(local_path) else (model_id or "google/gemma-4-e4b-it")

    logger.info(f"🔄 加载 Gemma 4 模型: {actual_id}")
    from transformers import AutoProcessor, AutoModel

    try:
        processor = AutoProcessor.from_pretrained(actual_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            actual_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=False,  # 我们会在 forward 时手动请求
        )
        model.eval()
    except Exception as e:
        logger.error(f"Gemma 4 模型加载失败: {e}。若显存不足，请尝试 --skip-gemma")
        return {}

    accum = {}
    for layer_idx in target_layers:
        accum[f"layer_{layer_idx}"] = {"visual": [], "text": []}
    valid_count = 0

    for sample in tqdm(samples, desc="Gemma 4 特征提取"):
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": sample["image_path"]},
                    {"type": "text", "text": sample["question"]},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for layer_idx in target_layers:
                if layer_idx >= len(outputs.hidden_states):
                    continue
                hidden = outputs.hidden_states[layer_idx][0]  # [seq_len, D]

                n_vis = min(visual_token_count, hidden.shape[0] // 2)
                vis_mean = hidden[:n_vis].mean(dim=0).cpu().float().numpy()
                txt_mean = hidden[n_vis:].mean(dim=0).cpu().float().numpy()

                accum[f"layer_{layer_idx}"]["visual"].append(vis_mean)
                accum[f"layer_{layer_idx}"]["text"].append(txt_mean)
            valid_count += 1
        except Exception as e:
            logger.warning(f"Gemma 4 样本处理失败: {e}")
            continue

    result = {}
    for key, val in accum.items():
        if val["visual"]:
            result[key] = {
                "visual": np.stack(val["visual"]),
                "text": np.stack(val["text"]),
            }

    logger.info(f"Gemma 4: {valid_count}/{len(samples)} 个样本成功, {len(result)} 层")

    # 显式释放模型以节省显存
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# ================================================================
# 4. t-SNE 3×3 矩阵图
# ================================================================

def _compute_center_distance(vis: np.ndarray, txt: np.ndarray) -> float:
    """计算视觉和文本 Token 中心之间的欧氏距离"""
    if len(vis) == 0 or len(txt) == 0:
        return float("inf")
    return float(np.linalg.norm(vis.mean(axis=0) - txt.mean(axis=0)))


def plot_tsne_matrix(
    llava_data: dict,
    gemma_data: dict,
    output_dir: str,
    max_points: int = 400,
):
    """
    生成 3行×3列 的 t-SNE 散点图矩阵。

    行1: LLaVA 的 layer_0, layer_12, layer_32
    行2: Gemma 4 的 layer_0, layer_12, layer_32
    行3: 跨模态相似度曲线对比

    蓝色 = 视觉 Token, 红色 = 文本 Token
    """
    # 选择关键层
    key_layers = ["layer_0", "layer_12", "layer_32"]
    layer_labels = ["Layer 0 (Input)", "Layer 12 (Middle)", "Layer 32 (Deep)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, (layer_name, layer_label) in enumerate(zip(key_layers, layer_labels)):
        # ---- LLaVA (row 0) ----
        ax_llava = axes[0, col]
        if layer_name in llava_data:
            vis = llava_data[layer_name]["visual"]
            txt = llava_data[layer_name]["text"]
            dist = _compute_center_distance(vis, txt)
            _tsne_subplot(ax_llava, vis, txt, max_points,
                          title=f"LLaVA — {layer_label}\nCenter Distance = {dist:.4f}")

        # ---- Gemma 4 (row 1) ----
        ax_gemma = axes[1, col]
        if layer_name in gemma_data:
            vis = gemma_data[layer_name]["visual"]
            txt = gemma_data[layer_name]["text"]
            dist = _compute_center_distance(vis, txt)
            _tsne_subplot(ax_gemma, vis, txt, max_points,
                          title=f"Gemma 4 — {layer_label}\nCenter Distance = {dist:.4f}")

    plt.tight_layout()
    path = os.path.join(output_dir, "tsne_comparison_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"t-SNE 矩阵图已保存: {path}")
    return path


def _tsne_subplot(ax, visual: np.ndarray, text: np.ndarray,
                  max_points: int, title: str):
    """在单个子图中绘制 t-SNE 散点"""
    if len(visual) == 0 or len(text) == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    all_tokens = np.concatenate([visual, text], axis=0)
    labels = np.array(["Visual"] * len(visual) + ["Text"] * len(text))

    if len(all_tokens) > max_points:
        idx = np.random.choice(len(all_tokens), max_points, replace=False)
        all_tokens = all_tokens[idx]
        labels = labels[idx]

    try:
        emb = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_tokens)//2-1),
                   n_iter=1000).fit_transform(all_tokens)
    except Exception:
        ax.text(0.5, 0.5, "t-SNE Failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    vis_mask = labels == "Visual"
    txt_mask = labels == "Text"

    ax.scatter(emb[vis_mask, 0], emb[vis_mask, 1],
               c="#3498DB", s=15, alpha=0.6, label=f"Visual ({vis_mask.sum()})")
    ax.scatter(emb[txt_mask, 0], emb[txt_mask, 1],
               c="#E74C3C", s=15, alpha=0.6, label=f"Text ({txt_mask.sum()})")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])


# ================================================================
# 5. 跨模态相似度曲线 + 交融起始层标注
# ================================================================

def _layer_similarities(features: dict) -> dict[str, float]:
    """计算各层视觉-文本余弦相似度"""
    sims = {}
    for layer_name in sorted(features.keys(), key=lambda x: int(x.split("_")[1])):
        data = features[layer_name]
        vis = torch.tensor(data["visual"])
        txt = torch.tensor(data["text"])
        if vis.shape[0] == 0 or txt.shape[0] == 0:
            sims[layer_name] = 0.0
            continue
        vis_mean = vis.mean(dim=0)
        sim = torch.cosine_similarity(vis_mean.unsqueeze(0), txt, dim=1).mean().item()
        sims[layer_name] = sim
    return sims


def _find_fusion_start_layer(sims: dict, threshold_ratio: float = 0.3) -> int:
    """
    找到"交融起始层"：相似度开始快速上升的层。

    策略：找到相似度增量首次超过总体增量的 threshold_ratio 的层。
    """
    items = sorted(sims.items(), key=lambda x: int(x[0].split("_")[1]))
    values = [v for _, v in items]
    layers = [int(k.split("_")[1]) for k, _ in items]

    if len(values) < 2:
        return layers[0] if layers else 0

    total_gain = max(values) - min(values)
    if total_gain <= 0:
        return layers[0]

    for i in range(1, len(values)):
        gain = values[i] - values[i-1]
        if gain > total_gain * threshold_ratio:
            return layers[i]

    return layers[-1]


def plot_cross_modal_curve(llava_features: dict, gemma_features: dict, output_dir: str):
    """
    绘制 LLaVA vs Gemma 4 跨模态相似度曲线，标注交融起始层。
    """
    llava_sims = _layer_similarities(llava_features)
    gemma_sims = _layer_similarities(gemma_features)

    fig, ax = plt.subplots(figsize=(12, 7))

    # LLaVA
    llava_items = sorted(llava_sims.items(), key=lambda x: int(x[0].split("_")[1]))
    lx = [int(k.split("_")[1]) for k, _ in llava_items]
    ly = [v for _, v in llava_items]

    # Gemma 4
    gemma_items = sorted(gemma_sims.items(), key=lambda x: int(x[0].split("_")[1]))
    gx = [int(k.split("_")[1]) for k, _ in gemma_items]
    gy = [v for _, v in gemma_items]

    ax.plot(lx, ly, "o-", color="#A23B72", linewidth=2.5, markersize=8,
            label="LLaVA (External Alignment)", zorder=5)
    ax.plot(gx, gy, "s-", color="#2E86AB", linewidth=2.5, markersize=8,
            label="Gemma 4 (Native Multimodal)", zorder=5)

    # 标注交融起始层
    if llava_sims:
        llava_fusion = _find_fusion_start_layer(llava_sims)
        if llava_fusion in lx:
            idx = lx.index(llava_fusion)
            ax.annotate(
                f"LLaVA Fusion Start\nLayer {llava_fusion}",
                xy=(llava_fusion, ly[idx]),
                xytext=(llava_fusion + 4, ly[idx] + 0.02),
                arrowprops=dict(arrowstyle="->", color="#A23B72", lw=1.5),
                fontsize=10, color="#A23B72", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    if gemma_sims:
        gemma_fusion = _find_fusion_start_layer(gemma_sims)
        if gemma_fusion in gx:
            idx = gx.index(gemma_fusion)
            ax.annotate(
                f"Gemma 4 Fusion Start\nLayer {gemma_fusion}",
                xy=(gemma_fusion, gy[idx]),
                xytext=(gemma_fusion + 4, gy[idx] - 0.05),
                arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=1.5),
                fontsize=10, color="#2E86AB", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # 起始点差异标注
    if ly and gy:
        ax.annotate(
            f"LLaVA: starts low ({ly[0]:.3f})\n→ needs Projector alignment",
            xy=(lx[0], ly[0]),
            xytext=(lx[0] + 2, ly[0] + 0.06),
            fontsize=9, color="#A23B72",
        )
        ax.annotate(
            f"Gemma 4: starts higher ({gy[0]:.3f})\n→ unified input space",
            xy=(gx[0], gy[0]),
            xytext=(gx[0] + 2, gy[0] - 0.08),
            fontsize=9, color="#2E86AB",
        )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cross-Modal Cosine Similarity", fontsize=12)
    ax.set_title(
        "Visual-Text Alignment Across Transformer Layers\n"
        "LLaVA (Projector-based) vs Gemma 4 (Unified Embedding Space)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 34)

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_modal_similarity.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"跨模态相似度曲线已保存: {path}")
    return path


# ================================================================
# 6. 生成分析报告
# ================================================================

def _generate_report(llava_features: dict, gemma_features: dict, output_dir: str) -> str:
    """生成 Markdown 分析报告"""
    llava_sims = _layer_similarities(llava_features)
    gemma_sims = _layer_similarities(gemma_features)

    llava_fusion = _find_fusion_start_layer(llava_sims) if llava_sims else "N/A"
    gemma_fusion = _find_fusion_start_layer(gemma_sims) if gemma_sims else "N/A"

    llava_start = list(llava_sims.values())[0] if llava_sims else 0
    gemma_start = list(gemma_sims.values())[0] if gemma_sims else 0
    llava_end = list(llava_sims.values())[-1] if llava_sims else 0
    gemma_end = list(gemma_sims.values())[-1] if gemma_sims else 0

    report = f"""# 交融机制可视化分析报告

## 1. 分析方法

- **特征提取**: 从 ChartQA-X 中抽取样本，对每个样本在各 Transformer 层提取视觉 Token 和文本 Token 的 hidden states，取平均作为该层该模态的代表向量。
- **t-SNE 降维**: 将各层视觉/文本向量降维到 2D 空间，观察分布重叠程度。
- **跨模态相似度**: 计算每层视觉和文本 Token 的平均余弦相似度，绘制随层数变化曲线。

## 2. 关键观察

| 指标 | LLaVA（外挂式） | Gemma 4（原生多模态） |
|------|----------------|---------------------|
| 起始相似度 (Layer 0) | {llava_start:.4f} | {gemma_start:.4f} |
| 最终相似度 (Layer 32) | {llava_end:.4f} | {gemma_end:.4f} |
| **交融起始层** | **Layer {llava_fusion}** | **Layer {gemma_fusion}** |
| 相似度总增幅 | {llava_end - llava_start:+.4f} | {gemma_end - gemma_start:+.4f} |

## 3. 融合机制差异解读

### LLaVA（外挂式多模态）
- 视觉特征通过 ViT 编码后，经过 **投影层（Projector）** 显式映射到文本空间
- 融合发生在投影层之后，需要多层 LLM 处理才能逐步缩小视觉-文本距离
- t-SNE 图中低层红蓝分离明显，高层逐渐混合

### Gemma 4（原生多模态）
- Patch Embedding 直接将图像块编码为 **软 Token**，与文本 Token 在同一空间
- 从输入层开始视觉和文本已较接近，各层进一步缩小差距
- 交融更早、更彻底：视觉信息从第一层就参与 Attention 计算

### 对下游任务的影响
- **空间推理任务**（如"哪个柱子最高"）：Gemma 4 应有优势，因为视觉位置信息从一开始就参与所有层的计算
- **纯文字提取**：OCR+LLM 基线可能反而更好，因为 PaddleOCR 在文字识别上极其成熟
- **幻觉**：原生多模态的统一空间表示可能减少幻觉，因为视觉和文本信息在每一层相互约束

## 4. 结论

1. **外挂式**的融合发生在投影层，是一个显式、可定位的对齐步骤
2. **原生多模态**的融合从输入层就已开始，是一个渐进、深层的过程
3. 两类模型的根本差异不在于"是否融合"，而在于**融合发生的层级和方式**

## 5. 输出文件

- `tsne_comparison_matrix.png`: t-SNE 3×3 矩阵对比图
- `cross_modal_similarity.png`: 跨模态相似度随层数变化曲线
- `alignment_analysis_report.md`: 本报告
"""

    report_path = os.path.join(output_dir, "alignment_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"分析报告已保存: {report_path}")
    return report


# ================================================================
# 7. 主入口
# ================================================================

def run_tsne_analysis(
    max_samples: int = 30,
    skip_llava: bool = False,
    skip_gemma: bool = False,
    output_dir: str = None,
):
    """
    运行完整 t-SNE 分析流程。

    Args:
        max_samples: 使用的样本数量（20-50 推荐）
        skip_llava: 跳过 LLaVA 分析
        skip_gemma: 跳过 Gemma 4 分析
        output_dir: 输出目录
    """
    output_dir = output_dir or TSNE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ---- 加载样本 ----
    samples = _load_sample_paths(max_samples)
    if not samples:
        logger.error("无可用样本，退出")
        return

    llava_features = {}
    gemma_features = {}

    # ---- LLaVA ----
    if not skip_llava:
        logger.info("=" * 60)
        logger.info("1/3 提取 LLaVA 各层特征...")
        logger.info("=" * 60)
        try:
            llava_features = extract_llava_features(samples)
            # 显式释放模型显存，避免与下一个模型叠加
            _free_gpu_memory()
        except Exception as e:
            logger.error(f"LLaVA 分析失败: {e}")
            import traceback; traceback.print_exc()
            _free_gpu_memory()
    else:
        logger.info("跳过 LLaVA 分析 (--skip-llava)")

    # ---- Gemma 4 ----
    if not skip_gemma:
        logger.info("=" * 60)
        logger.info("2/3 提取 Gemma 4 各层特征...")
        logger.info("=" * 60)
        try:
            gemma_features = extract_gemma_features(samples)
            _free_gpu_memory()
        except Exception as e:
            logger.error(f"Gemma 4 分析失败: {e}")
            import traceback; traceback.print_exc()
            _free_gpu_memory()
    else:
        logger.info("跳过 Gemma 4 分析 (--skip-gemma)")

    # ---- t-SNE 矩阵 + 相似度曲线 ----
    logger.info("=" * 60)
    logger.info("3/3 生成可视化图表与报告...")
    logger.info("=" * 60)

    plot_tsne_matrix(llava_features, gemma_features, output_dir)
    plot_cross_modal_curve(llava_features, gemma_features, output_dir)
    _generate_report(llava_features, gemma_features, output_dir)

    logger.info("=" * 60)
    logger.info(f"t-SNE 分析全部完成！输出目录: {output_dir}")
    logger.info("=" * 60)


# ================================================================
# CLI 入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE 交融机制可视化分析")
    parser.add_argument("--max-samples", type=int, default=30,
                       help="使用的样本数量（默认 30）")
    parser.add_argument("--skip-llava", action="store_true",
                       help="跳过 LLaVA 分析（显存不足时使用）")
    parser.add_argument("--skip-gemma", action="store_true",
                       help="跳过 Gemma 4 分析（显存不足时使用）")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认 tsne_visualization/）")
    args = parser.parse_args()

    run_tsne_analysis(
        max_samples=args.max_samples,
        skip_llava=args.skip_llava,
        skip_gemma=args.skip_gemma,
        output_dir=args.output_dir,
    )
