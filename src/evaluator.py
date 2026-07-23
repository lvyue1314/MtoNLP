#!/usr/bin/env python3
"""
步骤 3.1: 自动化评测模块

计算指标:
    - Exact Match (精确匹配)
    - Contains Match (包含匹配)
    - F1 Score (词级 F1)
    - 成功/失败数量

生成:
    - {model}_evaluation_report.json（单模型评测报告）
    - model_comparison.json（多模型对比）
"""

import json
import os
import re
from typing import Optional

import numpy as np

from config import RESULTS_DIR, setup_logging

logger = setup_logging(__name__)


class Evaluator:
    """模型评测器"""

    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir or RESULTS_DIR

    # ------------------------------------------------------------------
    # 答案标准化
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_answer(answer) -> str:
        """
        标准化答案：去标点、小写、合并空格。

        只保留字母、数字和空格，移除其他符号。
        """
        original = str(answer).strip()
        answer = original.lower()
        # 保护数字中的小数点
        answer = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", answer)
        answer = re.sub(r"[^a-z0-9\s]", "", answer)
        answer = answer.replace("<DOT>", ".")
        answer = re.sub(r"\s+", " ", answer).strip()
        # 修复: 纯中文文本(如"无法确定")被strip后变空串 → 保留原文
        if not answer:
            answer = original
        return answer

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def compute_metrics(self, predictions: list[str], ground_truths: list[str]) -> dict:
        """
        计算评估指标。

        Args:
            predictions: 模型预测答案列表
            ground_truths: 标准答案列表

        Returns:
            {
                "exact_match": float,     # 精确匹配率 (%)
                "contains_match": float,  # 包含匹配率 (%)
                "avg_f1": float,          # 平均词级 F1 (%)
                "total_samples": int,
            }
        """
        n = len(predictions)
        if n == 0:
            return {"exact_match": 0.0, "contains_match": 0.0, "avg_f1": 0.0, "total_samples": 0}

        norm_preds = [self.normalize_answer(p) for p in predictions]
        norm_gts = [self.normalize_answer(g) for g in ground_truths]

        # Exact Match
        exact = sum(1 for p, g in zip(norm_preds, norm_gts) if p == g)

        # Contains Match
        contains = sum(1 for p, g in zip(norm_preds, norm_gts) if g in p or p in g)

        # Word-level F1
        f1_scores = []
        for p, g in zip(norm_preds, norm_gts):
            p_words = set(p.split())
            g_words = set(g.split())
            if not p_words and not g_words:
                f1_scores.append(1.0)
            elif not p_words or not g_words:
                f1_scores.append(0.0)
            else:
                inter = p_words & g_words
                precision = len(inter) / len(p_words) if p_words else 0.0
                recall = len(inter) / len(g_words) if g_words else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_scores.append(f1)

        avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        return {
            "exact_match": round(exact / n * 100, 2),
            "contains_match": round(contains / n * 100, 2),
            "avg_f1": round(avg_f1 * 100, 2),
            "total_samples": n,
        }

    # ------------------------------------------------------------------
    # 单模型评测
    # ------------------------------------------------------------------

    def evaluate_results(self, results_path: str, model_output_name: str) -> Optional[dict]:
        """
        评测单个模型的结果文件。

        Args:
            results_path: 推理结果 JSON 路径
            model_output_name: 模型输出名（如 "gemma4"）

        Returns:
            评测报告 dict，失败则返回 None
        """
        if not os.path.exists(results_path):
            logger.warning(f"结果文件不存在: {results_path}")
            return None

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 分离有效结果和错误结果
        valid = [r for r in results if r.get("status") != "error"]
        errors = len(results) - len(valid)
        if errors > 0:
            logger.warning(f"{model_output_name}: {errors}/{len(results)} 个样本出错")

        predictions = [r.get("predicted_answer", "").strip() for r in valid]
        ground_truths = [r.get("answer", "").strip() for r in valid]

        metrics = self.compute_metrics(predictions, ground_truths)

        # 分成功/失败
        success_cases = []
        failed_cases = []
        for r in valid:
            pred_norm = self.normalize_answer(r.get("predicted_answer", ""))
            gt_norm = self.normalize_answer(r.get("answer", ""))
            if pred_norm == gt_norm:
                success_cases.append(r)
            else:
                failed_cases.append(r)

        report = {
            "model_name": model_output_name,
            "metrics": metrics,
            "success_count": len(success_cases),
            "failed_count": len(failed_cases),
            "error_count": errors,
            "success_cases": success_cases,
            "failed_cases": failed_cases,
        }

        # 保存报告
        report_dir = os.path.join(self.results_dir, model_output_name)
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{model_output_name}_evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(
            f"评测完成: {model_output_name} "
            f"(EM={metrics['exact_match']}%, "
            f"F1={metrics['avg_f1']}%, "
            f"成功={len(success_cases)}, 失败={len(failed_cases)})"
        )
        return report

    # ------------------------------------------------------------------
    # 多模型对比
    # ------------------------------------------------------------------

    def generate_comparison(self, model_names: list[str]) -> list[dict]:
        """
        生成多模型对比报告。

        Args:
            model_names: 模型输出名列表，如 ["gemma4", "llava", "baseline"]

        Returns:
            对比结果列表，按 Exact Match 降序排列
        """
        comparison = []
        for name in model_names:
            report_path = os.path.join(
                self.results_dir, name,
                f"{name}_evaluation_report.json",
            )
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                comparison.append({
                    "model": name,
                    **report["metrics"],
                    "success_count": report.get("success_count", 0),
                    "failed_count": report.get("failed_count", 0),
                })
            else:
                logger.warning(f"未找到 {name} 的评测报告")

        if not comparison:
            logger.warning("没有任何评测报告可对比")
            return []

        # 按精确匹配率降序
        comparison.sort(key=lambda x: x["exact_match"], reverse=True)

        # 保存对比文件
        comp_dir = os.path.join(self.results_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)
        comp_path = os.path.join(comp_dir, "model_comparison.json")
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        # 打印对比表
        self._print_comparison_table(comparison)

        logger.info(f"对比报告已保存: {comp_path}")
        return comparison

    @staticmethod
    def _print_comparison_table(comparison: list[dict]):
        """打印格式化的对比表"""
        print()
        print("=" * 70)
        print("📊 模型对比结果")
        print("=" * 70)
        print(f"{'排名':<5} {'模型':<20} {'精确匹配':<12} {'包含匹配':<12} {'F1分数':<10}")
        print("-" * 70)
        for i, c in enumerate(comparison, 1):
            print(
                f" {i:<4} {c['model']:<20} "
                f"{c['exact_match']:<10.2f}% "
                f"{c['contains_match']:<10.2f}% "
                f"{c['avg_f1']:<8.2f}%"
            )
        print("=" * 70)
