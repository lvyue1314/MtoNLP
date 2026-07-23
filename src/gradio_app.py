#!/usr/bin/env python3
"""
步骤 4.1: Gradio 演示系统

提供 Web 界面:
    - 上传论文图表（支持 png/jpg/jpeg）
    - 输入问题
    - 多选模型（Gemma 4 / LLaVA / OCR+LLM），同时对比
    - 每个模型独立显示回答 + 推理耗时
    - 从数据集中选取 2-3 张示例图表供快速体验

启动方式:
    python gradio_app.py
    python gradio_app.py --port 7860 --share
    ./run.sh gradio
"""

import os
import sys
import time
import base64
import json
import argparse
import random
from pathlib import Path
from typing import Optional

import gradio as gr

# 确保 src 可导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from config import (
    MODELS, WORKSPACE, RESULTS_DIR, CHARTQA_DATA, DATA_IMAGES,
    setup_logging,
)

logger = setup_logging(__name__)

# ================================================================
# 懒加载的模型实例（避免启动时占用资源）
# ================================================================
_baseline_model: Optional["OCRLLMBaseline"] = None


def _get_baseline():
    """懒加载 OCR+LLM 基线模型（线程安全）"""
    global _baseline_model
    if _baseline_model is None:
        from baseline_ocr_llm import OCRLLMBaseline
        _baseline_model = OCRLLMBaseline()
    return _baseline_model


# ================================================================
# 图片编码（复用 batch_inference 的 base64 方式）
# ================================================================

def _encode_image_to_data_url(image_path: str) -> str:
    """将本地图片编码为 base64 data URL（OpenAI 兼容格式）"""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ================================================================
# 单模型推理
# ================================================================

def _infer_baseline(image_path: str, question: str) -> tuple[str, float]:
    """OCR + LLM 基线推理，返回 (answer, elapsed_seconds)"""
    model = _get_baseline()
    start = time.time()
    try:
        answer = model.answer_question(image_path, question)
    except Exception as e:
        answer = f"❌ 推理失败: {e}"
    elapsed = time.time() - start
    return answer, elapsed


def _infer_vllm(image_path: str, question: str, model_key: str) -> tuple[str, float]:
    """通过 vLLM OpenAI 兼容 API 推理，返回 (answer, elapsed_seconds)"""
    import openai

    cfg = MODELS[model_key]
    image_url = _encode_image_to_data_url(image_path)

    start = time.time()
    try:
        client = openai.OpenAI(
            base_url=cfg["api_base"],
            api_key="EMPTY",
            timeout=120,
        )
        response = client.chat.completions.create(
            model=cfg["model_name"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }],
            max_tokens=256,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = (
            f"❌ vLLM 服务不可用\n"
            f"   端口: {cfg['api_base']}\n"
            f"   错误: {e}\n\n"
            f"请确认已启动 vLLM 服务"
        )
    elapsed = time.time() - start
    return answer, elapsed


# ================================================================
# 统一回调：对选中的模型分别推理
# ================================================================

def run_multi_inference(image, question: str, models_selected: list) -> dict:
    """
    对用户选中的所有模型分别推理。

    Args:
        image: 上传的图片（filepath 或 PIL Image）
        question: 问题文本
        models_selected: 选中的模型标签列表，如 ["Gemma 4", "LLaVA"]

    Returns:
        字典，键名对应各 gradio 输出组件
    """
    # ---- 输入校验 ----
    if image is None:
        empty = {"__all_empty__": "⚠️ 请先上传一张图片"}
        return empty

    if not question or not question.strip():
        empty = {"__all_empty__": "⚠️ 请输入问题"}
        return empty

    if not models_selected:
        empty = {"__all_empty__": "⚠️ 请至少选择一个模型"}
        return empty

    # 保存上传图片到临时路径（使用网络盘路径，避免跨机器丢失）
    tmp_dir = os.path.join(CHARTQA_DATA, "gradio_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "upload_temp.png")

    if hasattr(image, "save"):
        image.save(tmp_path)
    else:
        tmp_path = image  # 已是文件路径

    question = question.strip()

    # ---- 所有 3 个模型默认空值，选中才推理 ----
    ALL_MODELS = ["Gemma 4", "LLaVA", "OCR+LLM Baseline"]
    results = {}
    for m in ALL_MODELS:
        results[f"answer_{m}"] = ""
        results[f"time_{m}"] = ""

    for model_label in models_selected:
        try:
            if model_label == "OCR+LLM Baseline":
                answer, elapsed = _infer_baseline(tmp_path, question)
            elif model_label == "LLaVA":
                answer, elapsed = _infer_vllm(tmp_path, question, "llava")
            elif model_label == "Gemma 4":
                answer, elapsed = _infer_vllm(tmp_path, question, "gemma4")
            else:
                answer, elapsed = "未知模型", 0.0
        except Exception as e:
            answer = f"❌ 未预期错误: {e}"
            elapsed = 0.0

        results[f"answer_{model_label}"] = answer
        results[f"time_{model_label}"] = f"{elapsed:.2f} 秒"

    return results


# ================================================================
# 示例图片加载
# ================================================================

def _find_demo_images(max_count: int = 3) -> list[str]:
    """从 ChartQA-X 数据集中随机选取示例图片"""
    candidates = []
    # 优先从 CHARTQA_DATA 的 images 目录找
    img_dir = os.path.join(CHARTQA_DATA, "images")
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(os.path.join(img_dir, f))
    # 回退：遍历本地 data 目录
    data_dir = os.path.join(_project_root, "data")
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    candidates.append(os.path.join(root, f))

    # 如果还没有图片，尝试从 CHARTQA_DATA 读取标注信息
    if not candidates:
        ann_path = os.path.join(CHARTQA_DATA, "annotations", "test.json")
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data[:10]:
                abs_path = os.path.join(CHARTQA_DATA, item["image_path"])
                if os.path.exists(abs_path):
                    candidates.append(abs_path)

    if candidates:
        return random.sample(candidates, min(max_count, len(candidates)))
    return []


# ================================================================
# 界面构建
# ================================================================

def create_demo() -> gr.Blocks:
    """构建 Gradio Blocks 界面"""

    # ---- 评测结果（如有） ----
    comparison_text = "（请先运行评测 → 结果将自动显示）"
    comp_path = os.path.join(RESULTS_DIR, "comparison", "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        lines = [
            "| 排名 | 模型 | 精确匹配 | 包含匹配 | F1分数 |",
            "|------|------|----------|----------|--------|",
        ]
        for i, item in enumerate(comp_data, 1):
            lines.append(
                f"| {i} | {item['model']} | {item['exact_match']}% | "
                f"{item['contains_match']}% | {item['avg_f1']}% |"
            )
        comparison_text = "\n".join(lines)

    # ---- 示例图片 ----
    demo_images = _find_demo_images(3)

    with gr.Blocks(title="📊 论文图表问答系统") as demo:
        # ============================================================
        # 标题
        # ============================================================
        gr.Markdown("""
        # 📊 论文图表问答系统
        ### 对比评测：OCR+LLM 基线 / LLaVA（外挂式）/ Gemma 4（原生多模态）

        上传论文图表 → 输入问题 → 勾选模型 → 一键对比三个模型的回答效果。
        """)

        # ============================================================
        # 输入区
        # ============================================================
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### 📤 输入")
                image_input = gr.Image(
                    type="filepath",
                    label="上传论文图表（png / jpg）",
                    height=280,
                )
                question_input = gr.Textbox(
                    label="❓ 问题",
                    placeholder="例如：2020年的销售额是多少？哪个柱子最高？",
                    lines=2,
                )

                model_checkbox = gr.CheckboxGroup(
                    choices=["Gemma 4", "LLaVA", "OCR+LLM Baseline"],
                    label="🔧 选择模型（可多选对比）",
                    value=["Gemma 4", "LLaVA", "OCR+LLM Baseline"],
                    info="需先启动 vLLM 服务才能使用 Gemma 4 / LLaVA",
                )
                submit_btn = gr.Button(
                    "🚀 提交推理",
                    variant="primary",
                    size="lg",
                )

        # ============================================================
        # 输出区（三列）
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("### 💬 模型回答对比")

        outputs_list = []
        with gr.Row():
            for model_label in ["Gemma 4", "LLaVA", "OCR+LLM Baseline"]:
                with gr.Column(scale=1):
                    gr.Markdown(f"**{model_label}**")
                    answer_box = gr.Textbox(
                        label="回答",
                        lines=6,
                        interactive=False,
                        elem_classes=["result-box"],
                    )
                    time_box = gr.Textbox(
                        label="⏱️ 耗时",
                        lines=1,
                        interactive=False,
                    )
                    outputs_list.extend([answer_box, time_box])

        # ============================================================
        # 架构对比信息
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("### 📌 三类模型架构对比")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                **OCR + LLM 基线**
                - 📥 输入: 纯文本（OCR 提取）
                - 🔗 视觉→文本: ❌ 无融合
                - ✅ 优点: 文字提取准确
                - ❌ 缺点: 无法理解空间关系
                """)
            with gr.Column():
                gr.Markdown("""
                **LLaVA（外挂式多模态）**
                - 📥 输入: 图片 + 问题
                - 🔗 视觉→文本: 投影层对齐
                - ✅ 优点: 能理解视觉内容
                - ❌ 缺点: 融合有明确分界
                """)
            with gr.Column():
                gr.Markdown("""
                **Gemma 4（原生多模态）**
                - 📥 输入: 图片 + 问题
                - 🔗 视觉→文本: 统一空间（输入层）
                - ✅ 优点: 交融更早更彻底
                - ❌ 缺点: 纯文字提取不如 OCR
                """)

        # ============================================================
        # 评测结果
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("### 📈 批量评测结果")
        gr.Markdown(comparison_text)

        # ============================================================
        # 示例
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("### 💡 示例体验")
        gr.Examples(
            examples=[
                [img, q] for img in demo_images
                for q in [
                    "What is the maximum value in the chart?",
                    "How many categories are shown?",
                ]
            ][:6],
            inputs=[image_input, question_input],
            label="点击示例自动填充（需要已有数据）",
        )

        # ============================================================
        # 绑定事件
        # ============================================================
        # 构建输出映射
        output_keys = []
        for model_label in ["Gemma 4", "LLaVA", "OCR+LLM Baseline"]:
            output_keys.append(f"answer_{model_label}")
            output_keys.append(f"time_{model_label}")

        submit_btn.click(
            fn=run_multi_inference,
            inputs=[image_input, question_input, model_checkbox],
            outputs=outputs_list,
        )

    return demo


# ================================================================
# 启动入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChartQA-X Gradio 演示系统")
    parser.add_argument("--port", type=int, default=7860, help="服务端口（默认 7860）")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    args = parser.parse_args()

    demo = create_demo()
    logger.info(f"启动 Gradio 服务于 http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
