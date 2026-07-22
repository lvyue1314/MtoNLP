#!/usr/bin/env python3
"""
步骤 3.2: 误差分析模块

对模型预测错误进行自动分类，识别根本原因。该模块被 src.main 调用，
输出与评估报告放在同一目录。

四类错误定义：
    1. OCR_Error（文字识别错误）：
       预测与真实答案编辑距离小（≤2）且长度相近
    2. Visual_Localization_Failure（视觉定位失败）：
       预测答案与图中其他OCR文本匹配，但与正确答案不匹配
    3. Reasoning_Error（推理错误）：
       预测包含正确答案关键词，但组合/计算错误
    4. Hallucination（模型幻觉）：
       预测中的关键实体/数值在图中OCR结果中完全不存在

输出格式：
    results/{model_name}/error_analysis.json
"""

import json
import os
import re
import sys
from typing import Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from config import RESULTS_DIR, CHARTQA_DATA, setup_logging

logger = setup_logging(__name__)


# ================================================================
# 错误类型定义
# ================================================================

ERROR_TYPES = {
    "OCR_Error": {
        "name": "文字识别错误",
        "description": "预测与真实答案编辑距离小（≤2）且长度相近，OCR或模型看错了图表中的数字/文字",
        "typical_example": '"2019" 识别为 "2018"',
    },
    "Visual_Localization_Failure": {
        "name": "视觉定位失败",
        "description": "模型预测的答案与图中其他区域的OCR文本匹配，但与正确答案不匹配（仅多模态模型适用）",
        "typical_example": '问"1月销量"却关注了"2月"柱子的数值',
    },
    "Reasoning_Error": {
        "name": "推理错误",
        "description": "预测包含正确答案的关键词/数值，但组合方式或计算过程出错",
        "typical_example": "数值提取正确但单位转换或比较逻辑错误",
    },
    "Hallucination": {
        "name": "模型幻觉",
        "description": "预测中的关键实体/数值在图中OCR结果中完全不存在",
        "typical_example": "编造了一个图表中不存在的年份或数据点",
    },
}


# ================================================================
# 辅助函数
# ================================================================

def levenshtein_distance(a: str, b: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein distance）"""
    if not a:
        return len(b)
    if not b:
        return len(a)

    m, n = len(a), len(b)
    # 使用两行优化空间
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 删除
                curr[j - 1] + 1,   # 插入
                prev[j - 1] + cost # 替换
            )
        prev, curr = curr, prev

    return prev[n]


# 英文停用词表（避免在幻觉检测中误判常见词汇）
_ENGLISH_STOP_WORDS = {
    "the", "is", "at", "in", "of", "and", "to", "a", "an", "it", "on",
    "for", "with", "as", "by", "be", "this", "that", "from", "are", "was",
    "were", "has", "have", "had", "not", "but", "or", "can", "will", "would",
    "could", "should", "may", "might", "about", "than", "then", "also", "just",
    "only", "all", "some", "any", "each", "every", "both", "few", "more",
    "most", "other", "same", "such", "no", "new", "one", "two", "first",
    "last", "long", "great", "little", "own", "old", "big", "high",
    "different", "small", "large", "next", "early", "important", "public",
    "bad", "good", "able", "what", "which", "who", "whom", "when", "where",
    "why", "how", "there", "here", "their", "they", "its", "theirs",
}


def extract_keywords(text: str) -> set[str]:
    """
    从文本中提取关键数字和名词/实体（已过滤停用词）。

    Returns:
        关键字符串集合（数字 + 长度≥2的非停用英文单词）
    """
    text = str(text).strip()
    # 提取数字（含小数）
    numbers = set(re.findall(r"\d+\.?\d*", text))
    # 提取英文单词（≥2字符），过滤停用词
    words = {
        w.lower() for w in re.findall(r"[a-zA-Z]{2,}", text)
        if w.lower() not in _ENGLISH_STOP_WORDS
    }
    return numbers | words


def normalize_for_compare(text: str) -> str:
    """标准化文本用于比较：去标点（保留数字小数点）、小写、合并空格"""
    text = str(text).strip().lower()
    # 保护数字中的小数点
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.replace("<DOT>", ".")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================================================================
# 懒加载 OCR（复用 baseline_ocr_llm，避免重复初始化）
# ================================================================

_ocr_instance = None


def _get_ocr():
    """获取全局 PaddleOCR 实例"""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR()
            logger.info("PaddleOCR 实例已创建（误差分析用）")
        except Exception:
            # PaddleOCR 不可用（未安装/版本不兼容/平台限制）
            # 误差分析自动回退到纯文本规则，不影响主流程
            logger.info("PaddleOCR 不可用，误差分析将使用纯文本规则")
            _ocr_instance = False   # 用 False 标记"已尝试但不可用"，避免重复尝试
    if _ocr_instance is False:
        return None
    return _ocr_instance


def get_ocr_text(image_path: str) -> str:
    """
    获取图片中所有 OCR 文本（空格分隔）。

    Args:
        image_path: 图片绝对路径

    Returns:
        OCR 提取的文字，失败则返回空字符串
    """
    ocr = _get_ocr()
    if ocr is None:
        return ""

    try:
        result = ocr.ocr(image_path)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and len(line) >= 2 and line[1][0]]
        return " ".join(texts)
    except Exception as e:
        logger.warning(f"OCR 提取失败 ({image_path}): {e}")
        return ""


# ================================================================
# 错误分类器
# ================================================================

class ErrorAnalyzer:
    """自动 + 人工辅助的误差分析器"""

    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir or RESULTS_DIR

    # ------------------------------------------------------------------
    # 单样本分类
    # ------------------------------------------------------------------

    def classify_error(
        self,
        pred: str,
        gt: str,
        image_path: str = None,
        question: str = "",
        model_type: str = "gemma4",
    ) -> Tuple[str, float, str]:
        """
        对单个预测错误进行分类。

        Args:
            pred: 模型预测答案
            gt: 标准答案
            image_path: 图片路径（用于 OCR 提取）
            question: 问题文本
            model_type: 模型类型 "gemma4" | "llava" | "baseline"

        Returns:
            (error_type, confidence, reason)
                error_type: "OCR_Error" | "Visual_Localization_Failure" |
                            "Reasoning_Error" | "Hallucination"
                confidence: 0.0 ~ 1.0
                reason: 分类依据的文字说明
        """
        pred_norm = normalize_for_compare(pred)
        gt_norm = normalize_for_compare(gt)
        pred_raw = str(pred)
        question_lower = question.lower()

        # ---- 规则 0: 空预测 / 系统错误 ----
        if not pred_norm:
            return ("OCR_Error", 0.9, "预测为空，模型未能生成有效输出")

        if pred_norm.startswith("error") or pred_norm.startswith("image_not_found"):
            return ("OCR_Error", 0.95, "系统运行错误或无图片")

        # ---- 规则 1: 编辑距离分类（OCR_Error 的首选指标） ----
        edit_dist = levenshtein_distance(pred_norm, gt_norm)
        len_ratio = len(pred_norm) / max(len(gt_norm), 1)

        if edit_dist <= 2 and 0.6 <= len_ratio <= 1.6:
            return (
                "OCR_Error",
                0.85,
                f"编辑距离={edit_dist}（≤2），长度比={len_ratio:.2f}（接近1），"
                f"很可能是模型看错了图表中的文字/数字",
            )

        # ---- 规则 2: OCR-aware 视觉定位失败检测 ----
        if image_path and model_type != "baseline":
            ocr_text = get_ocr_text(image_path)
            if ocr_text:
                ocr_keywords = extract_keywords(ocr_text)
                pred_keywords = extract_keywords(pred_raw)
                gt_keywords = extract_keywords(gt)

                # 预测的关键词在 OCR 文本中存在，但不在标准答案中
                # → 模型看了图，但关注了错误的区域
                matched_in_ocr = pred_keywords & ocr_keywords
                unmatched_in_gt = pred_keywords - gt_keywords

                if matched_in_ocr and unmatched_in_gt:
                    return (
                        "Visual_Localization_Failure",
                        0.72,
                        f"预测中的关键词 {unmatched_in_gt} 出现在图片OCR文本中"
                        f"但不在标准答案中，模型可能关注了图中错误区域",
                    )

                # 预测的关键词完全不在 OCR 文本中 → 幻觉
                if pred_keywords and not (pred_keywords & ocr_keywords):
                    return (
                        "Hallucination",
                        0.80,
                        f"预测中的关键词 {pred_keywords} 均未在图片OCR文本中出现，"
                        f"模型可能编造了图表中不存在的信息",
                    )

        # ---- 规则 3: 推理错误检测 ----
        pred_nums = set(re.findall(r"\d+\.?\d*", pred_raw))
        gt_nums = set(re.findall(r"\d+\.?\d*", gt))

        # 预测包含正确答案的部分数值但整体不同 → 推理错误
        if pred_nums and gt_nums and pred_nums & gt_nums:
            return (
                "Reasoning_Error",
                0.70,
                f"预测与标准答案共享数值 {pred_nums & gt_nums}，"
                f"但最终答案不一致，可能是计算/比较逻辑出错",
            )

        # 预测数值都在答案数值附近但不完全一致 → 推理错误
        if pred_nums and gt_nums:
            for pn in pred_nums:
                for gn in gt_nums:
                    try:
                        diff = abs(float(pn) - float(gn))
                        if diff > 0 and diff < 5:
                            return (
                                "Reasoning_Error",
                                0.60,
                                f"预测数值 {pn} 与正确答案 {gn} 接近（差{diff:.1f}），"
                                f"可能是单位或精度转换错误",
                            )
                    except ValueError:
                        continue

        # ---- 规则 4: 空间定位问题（多模态模型） ----
        spatial_keywords = [
            "highest", "lowest", "largest", "smallest", "maximum", "minimum",
            "which bar", "which column", "peak", "top", "bottom",
            "最高", "最低", "最大", "最小", "哪个",
        ]
        if model_type != "baseline" and any(kw in question_lower for kw in spatial_keywords):
            return (
                "Visual_Localization_Failure",
                0.55,
                f"问题涉及空间/位置关系，预测错误可能是视觉定位不准所致",
            )

        # ---- 规则 5: 幻觉兜底检测 ----
        # 预测中提取的数字完全不在标准答案的数字集合中
        if pred_nums and gt_nums and not (pred_nums & gt_nums):
            return (
                "Hallucination",
                0.55,
                f"预测数值 {pred_nums} 与标准答案数值 {gt_nums} 完全不重叠",
            )

        # 预测远长于标准答案且无共享关键词
        shared = extract_keywords(pred_norm) & extract_keywords(gt_norm)
        if len(pred_norm) > len(gt_norm) * 2 and not shared:
            return (
                "Hallucination",
                0.50,
                "预测答案远超标准答案长度且无共享关键词",
            )

        # ---- 默认 ----
        return (
            "Reasoning_Error",
            0.30,
            f"编辑距离={edit_dist}，无法确定具体类型，建议人工复核",
        )

    # ------------------------------------------------------------------
    # 批量分类
    # ------------------------------------------------------------------

    def classify_errors(self, results_path: str, model_type: str) -> list[dict]:
        """
        对结果文件中的所有失败案例进行分类。

        Args:
            results_path: 推理结果 JSON 路径
            model_type: "baseline" | "llava" | "gemma4"

        Returns:
            带 error_type 标注的案例列表
        """
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 筛选失败案例
        failed = []
        for r in results:
            if r.get("status") == "error":
                failed.append(r)
            else:
                pred_norm = normalize_for_compare(r.get("predicted_answer", ""))
                gt_norm = normalize_for_compare(r.get("answer", ""))
                if pred_norm != gt_norm:
                    failed.append(r)

        logger.info(f"开始分类 {len(failed)} 个失败案例 ({model_type}) ...")

        classified = []
        for case in failed:
            # 构建图片绝对路径
            image_rel = case.get("image_path", "")
            image_abs = os.path.join(CHARTQA_DATA, image_rel)
            if not os.path.exists(image_abs) and "image_path_abs" in case:
                image_abs = case["image_path_abs"]
            if not os.path.exists(image_abs):
                image_abs = None  # 无法进行 OCR-aware 分类

            error_type, confidence, reason = self.classify_error(
                pred=case.get("predicted_answer", ""),
                gt=case.get("answer", ""),
                image_path=image_abs,
                question=case.get("question", ""),
                model_type=model_type,
            )

            classified.append({
                **case,
                "error_type": error_type,
                "error_type_name": ERROR_TYPES.get(error_type, {}).get("name", error_type),
                "confidence": confidence,
                "classification_reason": reason,
            })

        # 统计并打印
        counts = {}
        for c in classified:
            t = c["error_type"]
            counts[t] = counts.get(t, 0) + 1

        logger.info(f"分类完成 ({model_type}): {counts}")
        return classified

    # ------------------------------------------------------------------
    # 生成结构化 JSON 报告
    # ------------------------------------------------------------------

    def generate_error_analysis(self, classified: list[dict], model_name: str) -> dict:
        """
        生成结构化 error_analysis.json。

        Args:
            classified: classify_errors() 的返回结果
            model_name: 模型名称

        Returns:
            符合输出格式的字典
        """
        total = len(classified)

        # 统计分布
        distribution = {}
        for ek, ev in ERROR_TYPES.items():
            cases = [c for c in classified if c["error_type"] == ek]
            if cases:
                avg_conf = sum(c.get("confidence", 0) for c in cases) / len(cases)
                distribution[ek] = {
                    "count": len(cases),
                    "percentage": round(len(cases) / total * 100, 1) if total > 0 else 0.0,
                    "avg_confidence": round(avg_conf, 2),
                }

        # 典型案例（每类 3-5 个）
        typical_cases = {}
        for ek, ev in ERROR_TYPES.items():
            cases = [c for c in classified if c["error_type"] == ek]
            # 优先选高置信度的案例
            cases.sort(key=lambda c: c.get("confidence", 0), reverse=True)
            typical_cases[ek] = [
                {
                    "question": c.get("question", ""),
                    "predicted_answer": c.get("predicted_answer", ""),
                    "answer": c.get("answer", ""),
                    "image_path": c.get("image_path", ""),
                    "confidence": c.get("confidence", 0),
                    "reason": c.get("classification_reason", ""),
                }
                for c in cases[:5]
            ]

        report = {
            "model_name": model_name,
            "total_errors": total,
            "error_distribution": distribution,
            "typical_cases": typical_cases,
        }

        # 保存
        output_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}_error_analysis.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"误差分析报告已保存: {output_path}")
        return report

    # ------------------------------------------------------------------
    # 一键入口
    # ------------------------------------------------------------------

    def run_error_analysis(self, model_name: str) -> dict:
        """
        一键运行误差分析（供 main.py 调用）。

        Args:
            model_name: 模型名称 "gemma4" | "llava" | "baseline"

        Returns:
            结构化分析报告
        """
        # 确定 model_type
        if model_name in ("gemma4", "llava"):
            model_type = model_name
        else:
            model_type = "baseline"

        results_path = os.path.join(
            self.results_dir, model_name,
            f"{model_name}_results.json",
        )

        if not os.path.exists(results_path):
            logger.warning(f"结果文件不存在: {results_path}")
            return {}

        classified = self.classify_errors(results_path, model_type)

        # 保存分类详情
        detail_path = os.path.join(
            self.results_dir, model_name,
            f"{model_name}_error_classification.json",
        )
        os.makedirs(os.path.dirname(detail_path), exist_ok=True)
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(classified, f, indent=2, ensure_ascii=False)

        return self.generate_error_analysis(classified, model_name)

    # ------------------------------------------------------------------
    # 错误矩阵（三个模型对比）
    # ------------------------------------------------------------------

    def generate_error_matrix(self, classified_results: dict[str, list[dict]]) -> dict:
        """
        生成错误矩阵。

        Args:
            classified_results: {"gemma4": [...], "llava": [...], "baseline": [...]}

        Returns:
            矩阵数据
        """
        models = list(classified_results.keys())
        error_keys = list(ERROR_TYPES.keys())

        matrix = {}
        for model_name in models:
            cases = classified_results[model_name]
            matrix[model_name] = {"total_failed": len(cases)}
            for ek in error_keys:
                count = sum(1 for c in cases if c.get("error_type") == ek)
                matrix[model_name][ek] = count

        # 保存
        comp_dir = os.path.join(self.results_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)
        matrix_path = os.path.join(comp_dir, "error_matrix.json")
        with open(matrix_path, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2, ensure_ascii=False)

        logger.info(f"错误矩阵已保存: {matrix_path}")
        return matrix

    # ------------------------------------------------------------------
    # Markdown 报告
    # ------------------------------------------------------------------

    def generate_report(
        self,
        classified_results: dict[str, list[dict]],
        output_path: str = None,
    ) -> str:
        """生成 error_analysis.md"""
        output_path = output_path or os.path.join(
            self.results_dir, "comparison", "error_analysis.md"
        )

        lines = [
            "# 误差分析报告",
            "",
            "## 1. 错误分类标准",
            "",
            "| 错误类型 | 定义 | 典型例子 |",
            "|----------|------|----------|",
        ]
        for ek, ev in ERROR_TYPES.items():
            lines.append(f"| **{ev['name']}** | {ev['description']} | {ev['typical_example']} |")
        lines.append("")
        lines.append("> 注：OCR+LLM 基线的「视觉定位失败」不适用（该模型不接收图片输入）。")

        # 错误矩阵
        lines.append("")
        lines.append("## 2. 错误矩阵")
        lines.append("")
        headers = ["错误类型"] + list(classified_results.keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["------"] * len(headers)) + "|")
        for ek, ev in ERROR_TYPES.items():
            row = [ev["name"]]
            for model_name in classified_results:
                count = sum(1 for c in classified_results[model_name]
                          if c.get("error_type") == ek)
                row.append(str(count))
            lines.append("| " + " | ".join(row) + " |")

        # 典型案例
        lines.append("")
        lines.append("## 3. 典型案例")
        for model_name, cases in classified_results.items():
            lines.append(f"### {model_name}")
            for ek, ev in ERROR_TYPES.items():
                examples = [c for c in cases if c.get("error_type") == ek][:3]
                if examples:
                    lines.append(f"#### {ev['name']}")
                    for ex in examples:
                        lines.append(f"- **Q**: {ex.get('question', 'N/A')}")
                        lines.append(f"- **GT**: {ex.get('answer', 'N/A')}")
                        lines.append(f"- **Pred**: {ex.get('predicted_answer', 'N/A')}")
                        lines.append(f"- **Reason**: {ex.get('classification_reason', 'N/A')}")
                        lines.append("")

        # 结论
        lines.append("## 4. 分析结论")
        lines.append("")
        lines.append("（请在人工复核后补充）")

        md_content = "\n".join(lines)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"误差分析 Markdown 报告已保存: {output_path}")
        return md_content
