#!/usr/bin/env python3
"""
步骤 4.1: Gradio 演示系统

提供 Web 界面:
    - 上传论文图表
    - 输入问题
    - 选择模型（OCR+LLM / LLaVA / Gemma 4）
    - 返回答案 + 推理时间
    - 展示三类模型对比信息

启动方式:
    python gradio_app.py
    # 或
    python -m src.gradio_app
"""

import os
import sys
import time
import base64
import json
from pathlib import Path

import gradio as gr

# 确保 src 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS, MODELS_DIR, CHARTQA_DATA, setup_logging

logger = setup_logging(__name__)


# ================================================================
# 推理函数
# ================================================================

def inference_baseline(image_path: str, question: str) -> tuple[str, float]:
    """OCR + LLM 基线推理"""
    from baseline_ocr_llm import OCRLLMBaseline

    # 懒加载
    if not hasattr(inference_baseline, "_model"):
        inference_baseline._model = OCRLLMBaseline()

    model = inference_baseline._model
    start = time.time()
    answer = model.answer_question(image_path, question)
    elapsed = time.time() - start
    return answer, elapsed


def inference_vllm(image_path: str, question: str, model_key: str) -> tuple[str, float]:
    """通过 vLLM API 推理（Gemma 4 / LLaVA）"""
    import openai

    cfg = MODELS[model_key]
    client = openai.OpenAI(base_url=cfg["api_base"], api_key="EMPTY", timeout=120)

    # 读取图片并编码
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{encoded}"

    start = time.time()
    try:
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
        answer = f"[错误] vLLM 服务不可用: {e}\n请确认已启动 vLLM 服务（端口 {cfg['api_base']}）"

    elapsed = time.time() - start
    return answer, elapsed


def run_inference(image, question: str, model_choice: str) -> tuple[str, str]:
    """Gradio 回调函数"""
    if image is None:
        return "请先上传一张图片", "N/A"

    if not question or not question.strip():
        return "请输入问题", "N/A"

    # 保存上传图片到临时路径
    tmp_path = os.path.join(CHARTQA_DATA, "gradio_upload.png")
    if hasattr(image, "save"):
        image.save(tmp_path)
    else:
        # 已是文件路径
        tmp_path = image

    if model_choice == "OCR+LLM Baseline":
        answer, elapsed = inference_baseline(tmp_path, question.strip())
    elif model_choice == "LLaVA":
        answer, elapsed = inference_vllm(tmp_path, question.strip(), "llava")
    elif model_choice == "Gemma 4":
        answer, elapsed = inference_vllm(tmp_path, question.strip(), "gemma4")
    else:
        answer, elapsed = "未知模型", 0

    time_str = f"{elapsed:.2f} 秒"
    return answer, time_str


# ================================================================
# Gradio 界面
# ================================================================

def create_demo() -> gr.Blocks:
    """创建 Gradio 界面"""

    # 读取已有评测结果用于展示
    comparison_text = "（请先运行评测 -> 结果将自动显示在这里）"
    comp_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "comparison", "model_comparison.json",
    )
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        lines = ["| 排名 | 模型 | 精确匹配 | 包含匹配 | F1分数 |",
                 "|------|------|----------|----------|--------|"]
        for i, item in enumerate(comp_data, 1):
            lines.append(
                f"| {i} | {item['model']} | {item['exact_match']}% | "
                f"{item['contains_match']}% | {item['avg_f1']}% |"
            )
        comparison_text = "\n".join(lines)

    with gr.Blocks(
        title="📊 论文图表问答系统",
        theme=gr.themes.Soft(),
    ) as demo:
        # 标题
        gr.Markdown("""
        # 📊 论文图表问答系统
        ### 对比评测三类模型：OCR+LLM 基线 / LLaVA（外挂式多模态）/ Gemma 4（原生多模态）

        上传一张论文中的图表，输入问题，选择模型，查看不同模型的回答效果。
        """)

        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输入")
                image_input = gr.Image(
                    type="filepath",
                    label="上传论文图表",
                )
                question_input = gr.Textbox(
                    label="❓ 请输入问题",
                    placeholder="例如：2019年的销售额是多少？哪个柱子的值最高？图表中有几条折线？",
                    lines=3,
                )
                model_choice = gr.Radio(
                    choices=[
                        "Gemma 4",
                        "LLaVA",
                        "OCR+LLM Baseline",
                    ],
                    label="🔧 选择模型",
                    value="Gemma 4",
                    info="Gemma 4 和 LLaVA 需要先启动 vLLM 服务",
                )
                submit_btn = gr.Button("🚀 提交推理", variant="primary", size="lg")

            # 右侧：输出区
            with gr.Column(scale=1):
                gr.Markdown("### 💬 输出")
                answer_output = gr.Textbox(
                    label="模型回答",
                    lines=6,
                    interactive=False,
                )
                time_output = gr.Textbox(
                    label="⏱️ 推理时间",
                    lines=1,
                    interactive=False,
                )

        # ---- 模型对比信息 ----
        gr.Markdown("---")
        gr.Markdown("### 📌 三类模型架构对比")

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                **OCR + LLM 基线**
                - 输入: 纯文本（OCR 提取）
                - 视觉→文本融合: ❌ 无融合
                - 优点: 文字提取准确
                - 缺点: 无法理解空间关系
                """)
            with gr.Column():
                gr.Markdown("""
                **LLaVA（外挂式多模态）**
                - 输入: 图片 + 问题
                - 视觉→文本融合: 投影层（Projector）对齐
                - 优点: 能理解视觉内容
                - 缺点: 融合有明确分界
                """)
            with gr.Column():
                gr.Markdown("""
                **Gemma 4（原生多模态）**
                - 输入: 图片 + 问题
                - 视觉→文本融合: 统一空间（输入层）
                - 优点: 交融更早更彻底
                - 缺点: 文字提取不如 OCR
                """)

        # ---- 评测结果 ----
        gr.Markdown("---")
        gr.Markdown("### 📈 评测结果（自动读取 results/comparison/model_comparison.json）")
        gr.Markdown(comparison_text)

        # 绑定事件
        submit_btn.click(
            fn=run_inference,
            inputs=[image_input, question_input, model_choice],
            outputs=[answer_output, time_output],
        )

        # 示例
        gr.Markdown("---")
        gr.Markdown("### 💡 示例问题")
        gr.Examples(
            examples=[
                ["", "What is the highest value in the chart?"],
                ["", "How many categories are shown?"],
                ["", "What trend does the line show from 2018 to 2020?"],
                ["", "Which bar has the largest value?"],
            ],
            inputs=[image_input, question_input],
        )

    return demo


# ================================================================
# 启动入口
# ================================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 生成公网链接
    )
