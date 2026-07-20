#!/usr/bin/env python3
"""
步骤 3.2: 误差分析模块

对四类错误进行分类:
    1. 文字识别错误 - OCR 或模型看错了图表中的数字/文字
    2. 视觉定位失败 - 模型没有关注到正确的图表区域（仅多模态模型）
    3. 推理错误 - 正确识别了信息但逻辑推理出错
    4. 模型幻觉 - 生成了图表中不存在的信息

输出:
    - error_analysis.md / error_analysis.json (详细分析)
    - 错误矩阵 (三类模型 × 四类错误)
"""

import json
import os
import re
from typing import Optional

from config import RESULTS_DIR, setup_logging

logger = setup_logging(__name__)


# ================================================================
# 错误类型定义
# ================================================================

ERROR_TYPES = {
    "text_recognition": {
        "name": "文字识别错误",
        "description": "OCR 或模型看错了图表中的数字/文字",
        "typical_example": '"2019" 识别为 "2018"',
    },
    "visual_localization": {
        "name": "视觉定位失败",
        "description": "模型没有关注到正确的图表区域（仅多模态模型适用）",
        "typical_example": '问"1月销量"却关注了"2月"柱子',
    },
    "reasoning": {
        "name": "推理错误",
        "description": "正确识别了信息但逻辑推理出错",
        "typical_example": "数值提取正确但单位转换或比较逻辑错误",
    },
    "hallucination": {
        "name": "模型幻觉",
        "description": "生成了图表中不存在的信息",
        "typical_example": "编造了一个不存在的年份或数据点",
    },
}


class ErrorAnalyzer:
    """自动 + 人工辅助的误差分析器"""

    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir or RESULTS_DIR

    # ------------------------------------------------------------------
    # 自动错误分类（启发式规则 → 人工复核）
    # ------------------------------------------------------------------

    def classify_errors(self, results_path: str, model_type: str) -> list[dict]:
        """
        自动分类失败案例的错误类型。

        使用启发式规则初步分类，标记 confidence，
        需人工复核 low-confidence 的分类。

        Args:
            results_path: 推理结果 JSON
            model_type: "baseline" | "llava" | "gemma4"

        Returns:
            带 error_type 标注的失败案例列表
        """
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 筛选失败的案例
        evaluator_norm = self._normalize
        failed = []
        for r in results:
            if r.get("status") == "error":
                failed.append({**r, "error_type": "system_error", "confidence": "auto"})
            else:
                pred_norm = evaluator_norm(r.get("predicted_answer", ""))
                gt_norm = evaluator_norm(r.get("answer", ""))
                if pred_norm != gt_norm:
                    failed.append(r)

        # 对每个失败案例进行启发式分类
        classified = []
        for case in failed:
            error_type, confidence, reason = self._heuristic_classify(case, model_type)
            classified.append({
                **case,
                "error_type": error_type,
                "error_type_name": ERROR_TYPES.get(error_type, {}).get("name", error_type),
                "confidence": confidence,
                "classification_reason": reason,
            })

        # 统计
        type_counts = {}
        for c in classified:
            t = c.get("error_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        logger.info(f"自动分类完成 ({model_type}): {type_counts}")
        return classified

    def _heuristic_classify(self, case: dict, model_type: str) -> tuple[str, str, str]:
        """
        启发式分类规则。

        Returns:
            (error_type, confidence, reason)
        """
        pred = self._normalize(case.get("predicted_answer", ""))
        gt = self._normalize(case.get("answer", ""))
        pred_raw = str(case.get("predicted_answer", ""))
        question = case.get("question", "")

        # 规则1: 系统错误
        if case.get("error_type") == "system_error":
            return ("text_recognition", "high", "系统运行错误，无法获取预测结果")

        # 规则2: 预测答案包含图表中不存在的数字/实体 → 幻觉
        gt_nums = set(re.findall(r"\d+\.?\d*", gt))
        pred_nums = set(re.findall(r"\d+\.?\d*", pred_raw))
        if pred_nums - gt_nums and len(pred_nums) > 0:
            return ("hallucination", "medium", f"预测包含答案外的数字: {pred_nums - gt_nums}")

        # 规则3: 预测为空或无法提取文字（baseline 模型典型问题）
        if model_type == "baseline":
            if "无法" in pred_raw or "不能" in pred_raw:
                return ("text_recognition", "high", "OCR 无法从图片中提取文字")
            if not pred.strip():
                return ("text_recognition", "high", "预测为空，OCR 可能未识别到文字")

        # 规则4: 预测答案与正确答案数值不同但结构相似 → 推理错误
        if pred_nums and gt_nums and pred_nums.intersection(gt_nums):
            return ("reasoning", "medium", "部分数值正确但整体答案不一致，可能是推理错误")

        # 规则5: 问题涉及空间/位置关键词（只有多模态模型适用）
        spatial_keywords = ["最高", "最低", "最大", "最小", "哪个", "哪个柱子", "排名",
                            "largest", "smallest", "highest", "lowest", "which", "where"]
        if model_type != "baseline" and any(kw in question.lower() for kw in spatial_keywords):
            return ("visual_localization", "low", "问题涉及空间/位置关系，可能视觉定位失败")

        # 规则6: 预测包含明显不相关内容 → 幻觉
        if len(pred_raw) > len(gt) * 3 and not any(n in pred_raw for n in gt_nums):
            return ("hallucination", "low", "预测答案远超标准答案长度且无重叠数值")

        # 默认
        return ("reasoning", "low", "无法确定具体错误类型，建议人工复核")

    @staticmethod
    def _normalize(text: str) -> str:
        """标准化"""
        text = str(text).strip().lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # 生成错误矩阵
    # ------------------------------------------------------------------

    def generate_error_matrix(
        self, classified_results: dict[str, list[dict]]
    ) -> dict:
        """
        生成错误矩阵（三类模型 × 四类错误）。

        Args:
            classified_results: {"gemma4": [...], "llava": [...], "baseline": [...]}

        Returns:
            错误矩阵数据
        """
        models = list(classified_results.keys())
        error_keys = list(ERROR_TYPES.keys())

        matrix = {}
        for model in models:
            cases = classified_results[model]
            matrix[model] = {}
            for ek in error_keys:
                count = sum(1 for c in cases if c.get("error_type") == ek)
                matrix[model][ek] = count
            matrix[model]["total_failed"] = len(cases)

        # 保存
        matrix_path = os.path.join(self.results_dir, "comparison", "error_matrix.json")
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        with open(matrix_path, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2, ensure_ascii=False)

        logger.info(f"错误矩阵已保存: {matrix_path}")
        return matrix

    # ------------------------------------------------------------------
    # 生成 markdown 报告
    # ------------------------------------------------------------------

    def generate_report(
        self,
        classified_results: dict[str, list[dict]],
        output_path: str = None,
    ) -> str:
        """
        生成 error_analysis.md 报告。

        Args:
            classified_results: 分类后的失败案例
            output_path: 输出路径

        Returns:
            markdown 文本
        """
        output_path = output_path or os.path.join(
            self.results_dir, "comparison", "error_analysis.md"
        )

        lines = []
        lines.append("# 误差分析报告")
        lines.append("")
        lines.append("## 1. 错误分类标准")
        lines.append("")
        lines.append("| 错误类型 | 定义 | 典型例子 |")
        lines.append("|----------|------|----------|")
        for ek, ev in ERROR_TYPES.items():
            lines.append(f"| **{ev['name']}** | {ev['description']} | {ev['typical_example']} |")
        lines.append("")
        lines.append(f"> 注：OCR+LLM 基线的「视觉定位失败」不适用，因为该模型不接收图片输入。")
        lines.append("")

        # 错误矩阵
        lines.append("## 2. 错误矩阵（三类模型 × 四类错误）")
        lines.append("")
        headers = ["错误类型"] + [f"{m}" for m in classified_results.keys()]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["------"] * len(headers)) + "|")
        for ek, ev in ERROR_TYPES.items():
            row = [ev["name"]]
            for model_name in classified_results:
                count = sum(1 for c in classified_results[model_name]
                           if c.get("error_type") == ek)
                row.append(str(count))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # 典型案例
        lines.append("## 3. 典型案例分析")
        lines.append("")
        for model_name, cases in classified_results.items():
            lines.append(f"### 3.{list(classified_results.keys()).index(model_name)+1} {model_name}")
            lines.append("")
            for ek, ev in ERROR_TYPES.items():
                examples = [c for c in cases if c.get("error_type") == ek][:2]
                if examples:
                    lines.append(f"#### {ev['name']}")
                    lines.append("")
                    for ex in examples:
                        lines.append(f"- **问题**: {ex.get('question', 'N/A')}")
                        lines.append(f"- **标准答案**: {ex.get('answer', 'N/A')}")
                        lines.append(f"- **模型预测**: {ex.get('predicted_answer', 'N/A')}")
                        lines.append(f"- **分类依据**: {ex.get('classification_reason', 'N/A')}")
                        lines.append("")
            lines.append("")

        # 结论
        lines.append("## 4. 分析结论")
        lines.append("")
        lines.append("（请在此处补充人工分析结论）")
        lines.append("")

        md_content = "\n".join(lines)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"误差分析报告已保存: {output_path}")
        return md_content
