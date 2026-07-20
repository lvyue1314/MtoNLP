#!/usr/bin/env python3
"""
步骤 1.2: 下载 ChartQA-X 数据集并转换为标准格式

输出:
    - data/chartqa_x/raw/             原始数据缓存
    - data/chartqa_x/processed/eval_set.json
    - data/chartqa_x/processed/success_cases.json
    - data/chartqa_x/processed/failure_cases.json
"""

import os
import json
import random
from PIL import Image
from tqdm import tqdm

from config import CHARTQA_DATA, LMU_DATA, DATA_CACHE, setup_logging

logger = setup_logging(__name__)


def prepare_chartqa_data(
    max_samples: int = None,
    success_count: int = 10,
    failure_count: int = 10,
) -> list[dict]:
    """
    下载并准备 ChartQA-X 数据集。

    Args:
        max_samples: 最多保留多少样本（None = 全部）
        success_count: 成功案例子集数量
        failure_count: 失败案例子集数量

    Returns:
        标注列表 [{"image_path": ..., "question": ..., "answer": ...}, ...]
    """
    # ================================================================
    # 0. 检查是否已存在
    # ================================================================
    ann_path = os.path.join(CHARTQA_DATA, "annotations", "test.json")
    eval_set_path = os.path.join(CHARTQA_DATA, "annotations", "eval_set.json")
    success_path = os.path.join(CHARTQA_DATA, "annotations", "success_cases.json")
    failure_path = os.path.join(CHARTQA_DATA, "annotations", "failure_cases.json")

    if os.path.exists(eval_set_path):
        with open(eval_set_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(f"数据集已存在: {len(existing)} 个样本, 跳过下载")
        return existing

    # ================================================================
    # 1. 下载数据集
    # ================================================================
    logger.info("📥 从 HuggingFace 下载 ChartQA-X 数据集...")
    from datasets import load_dataset

    try:
        dataset = load_dataset(
            "shamanthakhegde/ChartQA-X",
            split="test",
            cache_dir=DATA_CACHE,
        )
    except Exception as e:
        logger.warning(f"HuggingFace 下载失败 ({e})，尝试使用镜像...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        dataset = load_dataset(
            "shamanthakhegde/ChartQA-X",
            split="test",
            cache_dir=DATA_CACHE,
        )

    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

    # ================================================================
    # 2. 保存图片和标注
    # ================================================================
    img_dir = os.path.join(CHARTQA_DATA, "images")
    os.makedirs(img_dir, exist_ok=True)

    annotations = []
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for idx in tqdm(range(total), desc="保存图片与标注"):
        sample = dataset[idx]

        # 保存图片
        img = sample["image"]
        img_filename = f"chart_{idx:06d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img.save(img_path)

        annotations.append({
            "id": idx,
            "image_path": os.path.join("images", img_filename),
            "image_path_abs": img_path,
            "question": sample["question"],
            "answer": sample["answer"],
        })

    # ================================================================
    # 3. 保存标注文件
    # ================================================================
    # 完整评测集
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # 也是 eval_set（兼容两种命名）
    with open(eval_set_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # 随机选取成功案例（简单样本，这里随机选取，后续可人工筛选）
    success_cases = random.sample(annotations, min(success_count, len(annotations)))
    with open(success_path, "w", encoding="utf-8") as f:
        json.dump(success_cases, f, indent=2, ensure_ascii=False)

    # 随机选取失败案例（复杂样本，后续可人工筛选替换为真正容易出错的）
    remaining = [a for a in annotations if a not in success_cases]
    failure_cases = random.sample(remaining, min(failure_count, len(remaining)))
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)

    # ================================================================
    # 4. 打印统计
    # ================================================================
    logger.info("=" * 60)
    logger.info("数据准备完成！")
    logger.info(f"  图片目录:    {img_dir} ({total} 张)")
    logger.info(f"  评测集:      {eval_set_path} ({len(annotations)} 条)")
    logger.info(f"  成功案例子集: {success_path} ({len(success_cases)} 条)")
    logger.info(f"  失败案例子集: {failure_path} ({len(failure_cases)} 条)")
    logger.info("=" * 60)

    # 设置环境变量（兼容 VLMEvalKit）
    os.environ["LMUData"] = LMU_DATA

    return annotations


if __name__ == "__main__":
    prepare_chartqa_data(max_samples=150)
