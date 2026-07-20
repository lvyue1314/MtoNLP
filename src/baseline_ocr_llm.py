#!/usr/bin/env python3
"""
步骤 2.1: OCR + LLM 基线模型

流水线: PaddleOCR 提取图表文字 → Qwen2.5-0.5B 根据文字回答问题

关键特点:
    - 模型不看图片，只看到 OCR 提取的纯文本 → 无法回答空间推理问题
    - 文字提取准确率最高（PaddleOCR 成熟），但缺乏视觉理解
"""

import json
import os
import threading
from typing import Optional

import torch
from tqdm import tqdm

from config import CHARTQA_DATA, setup_logging

logger = setup_logging(__name__)


class OCRLLMBaseline:
    """PaddleOCR + Qwen2.5-0.5B 基线"""

    def __init__(self, llm_model_name: str = "Qwen/Qwen2.5-0.5B"):
        """
        Args:
            llm_model_name: HuggingFace 模型 ID 或本地路径
        """
        # ---- 初始化 PaddleOCR ----
        logger.info("🔍 初始化 PaddleOCR...")
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=torch.cuda.is_available(),
            )
            logger.info("PaddleOCR 初始化完成")
        except ImportError:
            logger.error("PaddleOCR 未安装! 请运行: pip install paddlepaddle paddleocr")
            raise

        # ---- 初始化 LLM ----
        logger.info(f"📦 加载 LLM: {llm_model_name} ...")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"LLM 加载完成 (device={self.device})")

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # OCR 文字提取
    # ------------------------------------------------------------------

    def extract_text(self, image_path: str) -> str:
        """
        用 PaddleOCR 提取图表中所有文字。

        Args:
            image_path: 图表图片路径

        Returns:
            空格分隔的文字字符串（按检测框排列）
        """
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if not result or not result[0]:
                logger.warning(f"OCR 未检测到文字: {image_path}")
                return ""
            # 提取每个检测框的文字
            texts = []
            for line in result[0]:
                if line and len(line) >= 2 and line[1][0]:
                    texts.append(line[1][0])
            return " ".join(texts)
        except Exception as e:
            logger.error(f"OCR 错误 ({image_path}): {e}")
            return ""

    # ------------------------------------------------------------------
    # 单样本推理
    # ------------------------------------------------------------------

    def answer_question(self, image_path: str, question: str) -> str:
        """
        OCR 提取图表文字 → LLM 回答问题。

        核心设计: 因为 LLM 看不到图片，prompt 中只包含 OCR 文字
        """
        chart_text = self.extract_text(image_path)

        if not chart_text.strip():
            return "[无法从图片中提取文字]"

        prompt = (
            f"你是一个数据分析助手。以下是图表中通过OCR识别到的文字信息：\n\n"
            f"{chart_text}\n\n"
            f"问题：{question}\n\n"
            f"请根据上面的文字信息回答问题。只给出答案，不要解释。\n"
            f"答案："
        )

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 去掉 prompt 部分，只保留"答案："之后的内容
            if "答案：" in answer:
                answer = answer.split("答案：")[-1].strip()
            return answer

        except Exception as e:
            logger.error(f"LLM 推理错误: {e}")
            return f"[推理错误: {e}]"

    # ------------------------------------------------------------------
    # 批量推理（支持断点续传）
    # ------------------------------------------------------------------

    def run_inference(
        self,
        data_path: str,
        output_dir: str,
        max_samples: int = None,
        resume: bool = True,
    ) -> list[dict]:
        """
        批量运行 OCR+LLM 推理。

        Args:
            data_path: 标注文件路径
            output_dir: 输出目录
            max_samples: 限制样本数
            resume: 是否断点续传

        Returns:
            推理结果列表
        """
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "baseline_results.json")

        # ---- 断点续传 ----
        existing = []
        processed_keys = set()
        if resume and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for r in existing:
                key = f"{r.get('question','')}|{r.get('answer','')}|{r.get('image_path','')}"
                processed_keys.add(key)
            logger.info(f"断点续传: 已有 {len(existing)} 条结果")

        remaining = []
        for item in data:
            key = f"{item['question']}|{item['answer']}|{item['image_path']}"
            if key not in processed_keys:
                remaining.append(item)

        if not remaining:
            logger.info("所有样本已处理完成")
            return existing

        logger.info(f"待处理: {len(remaining)} 个样本")

        # ---- 串行推理（基线模型较轻量，串行即可） ----
        results = list(existing)
        data_dir = CHARTQA_DATA

        for item in tqdm(remaining, desc="OCR+LLM 基线推理"):
            image_rel = item["image_path"]
            image_abs = os.path.join(data_dir, image_rel)
            if not os.path.exists(image_abs) and "image_path_abs" in item:
                image_abs = item["image_path_abs"]

            pred = self.answer_question(image_abs, item["question"])

            result = {
                "id": item.get("id"),
                "question": item["question"],
                "answer": item["answer"],
                "predicted_answer": pred,
                "image_path": image_rel,
            }

            with self._lock:
                results.append(result)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"基线推理完成: {len(results)} 条, 输出: {output_path}")
        return results
