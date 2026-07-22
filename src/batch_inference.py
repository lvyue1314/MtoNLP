#!/usr/bin/env python3
"""
步骤 2.2 & 2.3: 使用 vLLM OpenAI 兼容 API 进行批量推理

支持:
    - Gemma 4 E4B (端口 8000)
    - LLaVA 1.5-7B (端口 8001)
    - 多线程并发请求
    - 断点续传（中断后自动跳过已处理样本）
"""

import json
import os
import base64
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import openai
from tqdm import tqdm

from config import CHARTQA_DATA, VLLM_CONCURRENT, setup_logging

logger = setup_logging(__name__)


class VLLMInference:
    """通过 vLLM 的 OpenAI 兼容 API 调用多模态大模型"""

    def __init__(
        self,
        api_base: str,
        model_name: str,
        output_dir: str,
        output_name: str = None,
        max_workers: int = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
        timeout: int = 300,
    ):
        self.api_base = api_base
        self.model_name = model_name                       # vLLM served-model-name（API 调用用）
        self.output_name = output_name or model_name       # 输出文件名前缀（如 "gemma4"）
        self.output_dir = output_dir
        self.max_workers = max_workers or VLLM_CONCURRENT
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.client = openai.OpenAI(
            base_url=api_base,
            api_key="EMPTY",  # vLLM 不需要真实 API key
            timeout=timeout,
        )
        os.makedirs(output_dir, exist_ok=True)
        self._lock = threading.Lock()

        logger.info(f"初始化 VLLMInference: {model_name} @ {api_base}, 并发={self.max_workers}")

    # ------------------------------------------------------------------
    # 单样本处理
    # ------------------------------------------------------------------

    def _process_single(self, idx: int, item: dict, data_dir: str) -> dict:
        """处理单个样本（线程安全）"""
        image_rel = item["image_path"]
        image_abs = os.path.join(data_dir, image_rel)

        # 也支持绝对路径
        if not os.path.exists(image_abs) and "image_path_abs" in item:
            image_abs = item["image_path_abs"]

        base_result = {
            "id": item.get("id", idx),
            "question": item["question"],
            "answer": item["answer"],
            "image_path": image_rel,
        }

        if not os.path.exists(image_abs):
            logger.warning(f"图片不存在: {image_abs}")
            base_result["predicted_answer"] = "IMAGE_NOT_FOUND"
            base_result["status"] = "error"
            return base_result

        try:
            # 读取图片并编码为 base64 data URL
            with open(image_abs, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:image/png;base64,{encoded}"

            # 调用 vLLM API（OpenAI 协议）
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": item["question"]},
                    ],
                }],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            pred = response.choices[0].message.content.strip()
            base_result["predicted_answer"] = pred
            base_result["status"] = "success"
            return base_result

        except Exception as e:
            logger.error(f"样本 {idx} 推理失败: {e}")
            base_result["predicted_answer"] = f"ERROR: {str(e)}"
            base_result["status"] = "error"
            return base_result

    # ------------------------------------------------------------------
    # 批量推理（支持断点续传）
    # ------------------------------------------------------------------

    def run_inference(
        self,
        data_path: str,
        max_samples: int = None,
        resume: bool = True,
    ) -> list[dict]:
        """
        执行批量推理。

        Args:
            data_path: 标注文件路径 (JSON)
            max_samples: 限制样本数量（None = 全部）
            resume: 是否启用断点续传

        Returns:
            推理结果列表
        """
        # 读取数据
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]
            logger.info(f"限制样本数: {max_samples}")

        output_path = os.path.join(self.output_dir, f"{self.output_name}_results.json")

        # ---- 断点续传：加载已有结果 ----
        existing_results = []
        processed_keys = set()

        if resume and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            for r in existing_results:
                key = f"{r.get('question','')}|{r.get('answer','')}|{r.get('image_path','')}"
                processed_keys.add(key)
            logger.info(f"断点续传: 已有 {len(existing_results)} 条结果")

        # 筛选未处理的样本
        remaining = []
        for item in data:
            key = f"{item['question']}|{item['answer']}|{item['image_path']}"
            if key not in processed_keys:
                remaining.append(item)

        if not remaining:
            logger.info("所有样本已处理完成，无需重复推理")
            return existing_results

        logger.info(f"待处理: {len(remaining)} 个新样本")

        # ---- 并发推理 ----
        results = list(existing_results)
        data_dir = CHARTQA_DATA
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, item in enumerate(remaining):
                future = executor.submit(self._process_single, i, item, data_dir)
                futures[future] = item

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{self.output_name} 推理",
            )
            for future in pbar:
                result = future.result()
                with self._lock:
                    results.append(result)
                    # 实时保存（断点续传的关键）
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        logger.info(f"推理完成: {len(results)} 条, 耗时 {elapsed:.1f}s, 输出: {output_path}")
        return results
