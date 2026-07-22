#!/usr/bin/env python3
"""
步骤 2.1: 规则基线模型

简易规则匹配基线，不依赖 OCR 或 LLM：
    - 从问题中提取数字 → 按规则选择（最大值/最小值/第一个）
    - 从问题中提取年份/类别 → 做简单匹配
    - 故意不看图片 → 用于对比"有视觉 vs 无视觉"的差异

符合项目要求："至少 1 个简单 baseline"

注：原 OCR+LLM 基线因云平台 PaddlePaddle 3.x + ROCm 兼容性问题暂不可用。
    规则基线同样满足"无视觉信息"的对照条件。
"""

import json
import os
import re
import random
import threading
from typing import Optional

from tqdm import tqdm

from config import setup_logging

logger = setup_logging(__name__)


class OCRLLMBaseline:
    """
    规则基线：不看图片，只从问题文本中做启发式回答。

    策略（按优先级）：
    1. 提取问题中所有数字 → 根据问题类型选择答案
       - "最大/最高/highest/largest" → 返回最大数字
       - "最小/最低/lowest/smallest" → 返回最小数字
       - "多少/how many" → 返回第一个数字
       - 其他 → 返回所有数字的拼接
    2. 如果问题中没有数字 → 返回通用回答
    """

    def __init__(self):
        random.seed(42)
        self._lock = threading.Lock()
        logger.info("规则基线已初始化（无 OCR，纯问题文本启发式）")

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def answer_question(self, image_path: str, question: str) -> str:
        """
        从问题文本做启发式回答。image_path 参数保留但不使用（基线不看图片）。
        """
        q = question.strip()
        q_lower = q.lower()

        # 提取所有数字（含小数）
        numbers = re.findall(r"\d+\.?\d*", q)
        nums = [float(n) for n in numbers]

        if not nums:
            # 问题中没有数字 → 无法回答
            return "无法确定"

        # ---- 根据问题类型选择答案 ----
        max_kw = ["最大", "最高", "largest", "highest", "maximum", "max",
                   "peak", "top", "most", "greater", "greatest"]
        min_kw = ["最小", "最低", "smallest", "lowest", "minimum", "min",
                   "least", "fewest", "bottom"]
        count_kw = ["多少", "几个", "几条", "how many", "count", "number of"]

        if any(kw in q_lower for kw in max_kw):
            return str(int(max(nums)) if max(nums) == int(max(nums)) else max(nums))
        elif any(kw in q_lower for kw in min_kw):
            return str(int(min(nums)) if min(nums) == int(min(nums)) else min(nums))
        elif any(kw in q_lower for kw in count_kw):
            return str(len(nums))
        else:
            # 默认：返回第一个数字
            val = nums[0]
            return str(int(val) if val == int(val) else val)

    # ------------------------------------------------------------------
    # 批量推理（保持与 vLLM 接口一致的输出格式）
    # ------------------------------------------------------------------

    def run_inference(
        self,
        data_path: str,
        output_dir: str,
        max_samples: int = None,
        resume: bool = True,
    ) -> list[dict]:
        """
        批量运行规则基线推理。

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

        results = list(existing)

        for item in tqdm(remaining, desc="规则基线推理"):
            pred = self.answer_question(
                item.get("image_path_abs", item["image_path"]),
                item["question"],
            )

            result = {
                "id": item.get("id"),
                "question": item["question"],
                "answer": item["answer"],
                "predicted_answer": pred,
                "image_path": item["image_path"],
            }

            with self._lock:
                results.append(result)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"规则基线推理完成: {len(results)} 条, 输出: {output_path}")
        return results
