#!/usr/bin/env python3
"""
ChartQA-X 三模型评测系统

在 AMD 云平台上对 ChartQA-X 数据集进行多模态模型评测，
对比 OCR+LLM 基线、LLaVA（外挂式多模态）、Gemma 4（原生多模态）
三种技术路线在图表理解任务上的表现。

使用方式:
    python -m src.main --models gemma4 llava baseline
    python -m src.main --models baseline --max-samples 5

功能:
    - 自动数据下载与预处理
    - vLLM 批量推理（Gemma 4 / LLaVA）
    - OCR+LLM 基线推理
    - 准确率 / F1 / 包含匹配 等指标计算
    - 四类误差自动分类
    - t-SNE 交融机制可视化
    - 结果对比图表生成
    - Gradio 演示界面
"""

__version__ = "1.0.0"
__author__ = "NLP Course Team"

from .config import *
from .data_prepare import prepare_chartqa_data
from .batch_inference import VLLMInference
from .baseline_ocr_llm import OCRLLMBaseline
from .evaluator import Evaluator
from .error_analyzer import ErrorAnalyzer
from .visualizer import (
    plot_model_comparison,
    plot_success_failed,
    plot_radar,
    plot_error_matrix,
    plot_failed_cases_analysis,
    generate_all_plots,
)

__all__ = [
    "prepare_chartqa_data",
    "VLLMInference",
    "OCRLLMBaseline",
    "Evaluator",
    "ErrorAnalyzer",
    "plot_model_comparison",
    "plot_success_failed",
    "plot_radar",
    "plot_error_matrix",
    "plot_failed_cases_analysis",
    "generate_all_plots",
]
