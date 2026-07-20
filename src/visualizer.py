#!/usr/bin/env python3
"""
步骤 3.1 & 3.2: 结果可视化模块

生成:
    - 模型对比柱状图（三指标并列）
    - 成功/失败数量对比
    - 雷达图
    - 失败案例分析图
    - 错误矩阵热力图
"""

import json
import os
import platform
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 无头模式，不需要 GUI
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

from config import RESULTS_DIR, setup_logging

logger = setup_logging(__name__)


# ================================================================
# 中文字体配置
# ================================================================

def _setup_chinese_font() -> Optional[str]:
    """配置中文字体（Linux / macOS / Windows 兼容），结果自动缓存"""
    # 如果已检测过，直接返回缓存结果
    if hasattr(_setup_chinese_font, "_cached"):
        return _setup_chinese_font._cached

    system = platform.system()

    if system == "Linux":
        candidates = [
            "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
            "Noto Sans CJK SC", "Noto Sans CJK TC",
            "Source Han Sans SC", "AR PL UMing CN", "DejaVu Sans",
        ]
    elif system == "Darwin":
        candidates = [
            "PingFang SC", "Heiti SC", "STHeiti",
            "Arial Unicode MS", "DejaVu Sans",
        ]
    else:  # Windows
        candidates = [
            "Microsoft YaHei", "SimHei", "SimSun",
            "Arial Unicode MS", "DejaVu Sans",
        ]

    available = {f.name for f in fm.fontManager.ttflist}
    found = None
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            logger.info(f"使用字体: {font}")
            found = font
            break

    if found is None:
        logger.warning("未找到中文字体，使用默认字体")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    _setup_chinese_font._cached = found
    return found


# ================================================================
# 主对比图
# ================================================================

def plot_model_comparison(comparison_path: str, output_dir: str = None) -> Optional[str]:
    """
    生成模型对比柱状图（三指标并列）。

    Args:
        comparison_path: model_comparison.json 路径
        output_dir: 输出目录

    Returns:
        输出 png 路径
    """
    if not os.path.exists(comparison_path):
        logger.warning(f"对比文件不存在: {comparison_path}")
        return None

    with open(comparison_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.warning("对比数据为空")
        return None

    output_dir = output_dir or os.path.dirname(comparison_path)
    os.makedirs(output_dir, exist_ok=True)
    _setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = [item["model"] for item in data]
    metrics = ["exact_match", "contains_match", "avg_f1"]
    titles = ["Exact Match", "Contains Match", "Avg F1 Score"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        scores = [item.get(metric, 0) for item in data]
        bars = axes[i].bar(models, scores, color=color, alpha=0.85, edgecolor="white")
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_ylabel("Score (%)")
        axes[i].set_ylim(0, max(105, max(scores) * 1.2))
        axes[i].grid(axis="y", alpha=0.3)
        axes[i].tick_params(axis="x", rotation=15)

        for bar, score in zip(bars, scores):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    plt.tight_layout()

    for fmt in ["png"]:  # 默认只输出 PNG（PDF/SVG 较大且服务器通常不需要）
        path = os.path.join(output_dir, f"model_comparison.{fmt}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"图表已保存: {path}")

    plt.close()
    return os.path.join(output_dir, "model_comparison.png")


# ================================================================
# 成功/失败对比图
# ================================================================

def plot_success_failed(comparison_path: str, output_dir: str = None) -> Optional[str]:
    """生成成功/失败数量对比图"""
    if not os.path.exists(comparison_path):
        return None

    with open(comparison_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = output_dir or os.path.dirname(comparison_path)
    _setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = [d["model"] for d in data]
    success = [d.get("success_count", 0) for d in data]
    failed = [d.get("failed_count", 0) for d in data]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, success, width, label="Success", color="#27AE60", alpha=0.85)
    ax.bar(x + width / 2, failed, width, label="Failed", color="#E74C3C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Success vs Failed Counts by Model", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 标注数字
    for i, (s, f) in enumerate(zip(success, failed)):
        ax.text(i - width / 2, s + 0.5, str(s), ha="center", fontweight="bold")
        ax.text(i + width / 2, f + 0.5, str(f), ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "success_failed_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"成功/失败图已保存: {path}")
    return path


# ================================================================
# 雷达图
# ================================================================

def plot_radar(comparison_path: str, output_dir: str = None) -> Optional[str]:
    """生成模型能力雷达图"""
    if not os.path.exists(comparison_path):
        return None

    with open(comparison_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = output_dir or os.path.dirname(comparison_path)
    _setup_chinese_font()

    metrics = ["exact_match", "contains_match", "avg_f1"]
    metric_labels = ["Exact Match", "Contains Match", "F1 Score"]
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for item, color in zip(data, colors):
        values = [item["metrics"][m] if "metrics" in item else item.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=item["model"], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Model Capability Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"雷达图已保存: {path}")
    return path


# ================================================================
# 错误矩阵热力图
# ================================================================

def plot_error_matrix(matrix_path: str, output_dir: str = None) -> Optional[str]:
    """生成错误矩阵热力图"""
    if not os.path.exists(matrix_path):
        logger.warning(f"错误矩阵文件不存在: {matrix_path}")
        return None

    with open(matrix_path, "r", encoding="utf-8") as f:
        matrix = json.load(f)

    output_dir = output_dir or os.path.dirname(matrix_path)
    _setup_chinese_font()

    models = list(matrix.keys())
    error_labels = ["Text Recognition", "Visual Localization", "Reasoning", "Hallucination"]
    error_keys = ["text_recognition", "visual_localization", "reasoning", "hallucination"]

    # 构建数据矩阵
    data_matrix = []
    for model in models:
        row = [matrix[model].get(ek, 0) for ek in error_keys]
        data_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt="d",
        xticklabels=error_labels,
        yticklabels=models,
        cmap="YlOrRd",
        ax=ax,
        linewidths=1,
        cbar_kws={"label": "Error Count"},
    )
    ax.set_title("Error Type Distribution Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Model")
    plt.tight_layout()

    path = os.path.join(output_dir, "error_matrix_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"错误矩阵热力图已保存: {path}")
    return path


# ================================================================
# 失败案例长度差异分析
# ================================================================

def plot_failed_cases_analysis(
    report_path: str, output_dir: str = None, top_n: int = 10
) -> Optional[str]:
    """绘制失败案例的预测长度 vs 标准长度差异"""
    if not os.path.exists(report_path):
        logger.warning(f"报告文件不存在: {report_path}")
        return None

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    failed = report.get("failed_cases", [])
    if not failed:
        logger.info("没有失败案例，跳过绘图")
        return None

    output_dir = output_dir or os.path.dirname(report_path)
    _setup_chinese_font()

    # 截取前 N 个
    failed = failed[:top_n]

    questions = []
    diffs = []
    for case in failed:
        q = case.get("question", "")[:40]
        if len(case.get("question", "")) > 40:
            q += "..."
        questions.append(q)
        pred_len = len(str(case.get("predicted_answer", "")))
        gt_len = len(str(case.get("answer", "")))
        diffs.append(pred_len - gt_len)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in diffs]
    ax.barh(range(len(questions)), diffs, color=colors, alpha=0.8)

    # 修复: 先设置 yticks，再设置 yticklabels
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions, fontsize=9)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Pred Length - GT Length")
    ax.set_title(f"Failed Cases: Length Difference (Top {top_n})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "failed_cases_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"失败案例分析图已保存: {path}")
    return path


# ================================================================
# 一键生成所有图表
# ================================================================

def generate_all_plots(results_dir: str = None):
    """生成所有可视化图表"""
    results_dir = results_dir or RESULTS_DIR
    _setup_chinese_font()

    comp_path = os.path.join(results_dir, "comparison", "model_comparison.json")
    matrix_path = os.path.join(results_dir, "comparison", "error_matrix.json")

    if not os.path.exists(comp_path):
        logger.warning("未找到 model_comparison.json，请先运行评测")
        return

    logger.info("=" * 60)
    logger.info("📊 生成所有可视化图表...")
    logger.info("=" * 60)

    plot_model_comparison(comp_path)
    plot_success_failed(comp_path)
    plot_radar(comp_path)

    if os.path.exists(matrix_path):
        plot_error_matrix(matrix_path)

    # 各模型的失败案例分析
    for model_name in ["gemma4", "llava", "baseline"]:
        report_path = os.path.join(
            results_dir, model_name,
            f"{model_name}_evaluation_report.json",
        )
        if os.path.exists(report_path):
            plot_failed_cases_analysis(report_path)

    logger.info("所有图表生成完成！")
