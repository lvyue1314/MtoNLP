#!/usr/bin/env python3
"""
三模型对比分析报告生成器

读取 gemma4 / llava / baseline 三个模型的推理结果，
输出：
    1. 汇总对比表 (Markdown)
    2. 视觉依赖型问题子集分析
    3. 典型案例（Gemma 4 对基线错、基线对 Gemma 4 错）
    4. CSV 导出（方便 Excel 绘图）
"""

import json
import os
import re
import sys
from collections import defaultdict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

RESULTS_DIR = os.path.join(_project_root, "results")


def load_results(model_name: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, model_name, f"{model_name}_results.json")
    if not os.path.exists(path):
        print(f"⚠️ 未找到 {model_name} 的结果文件，跳过")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = s.replace("<DOT>", ".")
    return s


def is_contains_match(pred: str, gt: str) -> bool:
    p, g = norm(pred), norm(gt)
    return g in p or p in g


def is_visual_question(q: str) -> bool:
    """判断问题是否依赖于视觉理解（无法从问题文本中获取答案）"""
    visual_kw = [
        "color", "colour", "which bar", "which line", "which chart",
        "describe", "trend", "shape", "graph", "chart show",
        "based on the chart", "based on the graph", "based on the table",
        "ratio", "percent of", "compare", "difference between",
        "highest", "lowest", "increase", "decrease", "more than",
        "less than", "fewer", "sum", "average", "total",
        "颜色", "柱状", "折线", "趋势", "哪个", "最高", "最低",
    ]
    q_lower = q.lower()
    return any(kw in q_lower for kw in visual_kw)


def generate_report():
    gemma = load_results("gemma4")
    llava = load_results("llava")
    baseline = load_results("baseline")

    results = {"gemma4": gemma, "llava": llava, "baseline": baseline}
    available = [k for k, v in results.items() if v]
    if len(available) < 2:
        print("至少需要两个模型的结果才能对比")
        return

    n = min(len(results[m]) for m in available)
    print(f"样本对齐: {n} 条 (模型: {', '.join(available)})")

    # ================================================================
    # 1. 总体指标
    # ================================================================
    print()
    print("=" * 70)
    print("一、总体评测指标")
    print("=" * 70)

    metrics = {}
    for m in available:
        data = results[m][:n]
        em = sum(1 for d in data if norm(d["predicted_answer"]) == norm(d["answer"])) / n * 100
        cm = sum(1 for d in data if is_contains_match(d["predicted_answer"], d["answer"])) / n * 100
        metrics[m] = {"EM": em, "CM": cm, "N": n}

    print()
    print("| 模型 | Exact Match | Contains Match | 样本数 |")
    print("|------|------------|---------------|--------|")
    for m in available:
        print(f"| {m:12s} | {metrics[m]['EM']:5.1f}%     | {metrics[m]['CM']:5.1f}%       | {n:6d} |")
    print()

    # ================================================================
    # 2. 视觉依赖型问题子集
    # ================================================================
    print()
    print("=" * 70)
    print("二、视觉依赖型问题子集分析")
    print("=" * 70)

    visual_indices = [i for i in range(n) if is_visual_question(gemma[i]["question"])]
    n_vis = len(visual_indices)
    print(f"视觉依赖型问题: {n_vis}/{n} ({n_vis/n*100:.0f}%)")

    print()
    print("| 模型 | EM (视觉题) | CM (视觉题) |")
    print("|------|-----------|-----------|")
    for m in available:
        data = [results[m][i] for i in visual_indices]
        em = sum(1 for d in data if norm(d["predicted_answer"]) == norm(d["answer"])) / n_vis * 100
        cm = sum(1 for d in data if is_contains_match(d["predicted_answer"], d["answer"])) / n_vis * 100
        print(f"| {m:12s} | {em:5.1f}%      | {cm:5.1f}%      |")
    print()

    # ================================================================
    # 3. 模型间差异分析
    # ================================================================
    if "gemma4" in available and "baseline" in available:
        print()
        print("=" * 70)
        print("三、Gemma 4 vs 规则基线 案例分类")
        print("=" * 70)

        gemma_only = []     # Gemma 对，基线错
        baseline_only = []  # 基线对，Gemma 错
        both_right = []     # 两个都对
        both_wrong = []     # 两个都错

        for i in range(n):
            g = is_contains_match(gemma[i]["predicted_answer"], gemma[i]["answer"])
            b = is_contains_match(baseline[i]["predicted_answer"], baseline[i]["answer"])
            if g and not b:
                gemma_only.append(i)
            elif b and not g:
                baseline_only.append(i)
            elif g and b:
                both_right.append(i)
            else:
                both_wrong.append(i)

        print(f"Gemma 对 + 基线错: {len(gemma_only)} 条  ← 多模态价值")
        print(f"基线对 + Gemma 错: {len(baseline_only)} 条")
        print(f"两个都对 (撞库命中): {len(both_right)} 条")
        print(f"两个都错:           {len(both_wrong)} 条")

        print()
        print("### Gemma 4 独对的典型案例")
        for idx in gemma_only[:8]:
            d = gemma[idx]
            print(f"- **Q**: {d['question'][:100]}")
            print(f"  GT: `{d['answer']}`")
            print(f"  Gemma4: `{d['predicted_answer'][:80]}`")
            print(f"  基线:   `{baseline[idx]['predicted_answer'][:80]}`")
            print()

    # ================================================================
    # 4. CSV 导出
    # ================================================================
    csv_path = os.path.join(RESULTS_DIR, "comparison", "all_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        headers = ["id", "question", "answer"]
        for m in available:
            headers.append(f"{m}_pred")
            headers.append(f"{m}_correct")
        f.write(",".join(headers) + "\n")

        for i in range(n):
            row = [str(i), f'"{gemma[i]["question"]}"', f'"{gemma[i]["answer"]}"']
            for m in available:
                pred = results[m][i]["predicted_answer"].replace('"', "'")
                correct = "1" if is_contains_match(pred, results[m][i]["answer"]) else "0"
                row.append(f'"{pred}"')
                row.append(correct)
            f.write(",".join(row) + "\n")

    print(f"CSV 已导出: {csv_path}")

    # ================================================================
    # 5. 汇总 Markdown
    # ================================================================
    md_path = os.path.join(RESULTS_DIR, "comparison", "analysis_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 三模型对比分析报告\n\n")
        f.write("## 总体指标\n\n")
        f.write("| 模型 | Exact Match | Contains Match |\n")
        f.write("|------|------------|---------------|\n")
        for m in available:
            f.write(f"| {m} | {metrics[m]['EM']:.1f}% | {metrics[m]['CM']:.1f}% |\n")

        if n_vis > 0:
            f.write(f"\n## 视觉依赖型问题子集 ({n_vis} 题)\n\n")
            f.write("| 模型 | EM | CM |\n")
            f.write("|------|----|----|\n")
            for m in available:
                data = [results[m][i] for i in visual_indices]
                em = sum(1 for d in data if norm(d["predicted_answer"]) == norm(d["answer"])) / n_vis * 100
                cm = sum(1 for d in data if is_contains_match(d["predicted_answer"], d["answer"])) / n_vis * 100
                f.write(f"| {m} | {em:.1f}% | {cm:.1f}% |\n")

        f.write("\n> 生成时间: 见文件修改时间\n")

    print(f"Markdown 报告已导出: {md_path}")


if __name__ == "__main__":
    generate_report()
