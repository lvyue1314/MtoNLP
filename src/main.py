#!/usr/bin/env python3
"""
ChartQA-X 三模型评测系统 — 主入口

运行示例:
    # 测试模式（基线模型，5 个样本）
    python -m src.main --models baseline --max-samples 5

    # 快速评测（所有模型，10 个样本）
    python -m src.main --max-samples 10

    # 完整评测（所有模型，全部样本）
    python -m src.main

    # 只跑 Gemma 4
    python -m src.main --models gemma4

    # 详细日志
    python -m src.main --verbose --max-samples 20

    # 跳过数据准备
    python -m src.main --skip-data

    # 包含 t-SNE 分析
    python -m src.main --tsne
"""

import os
import sys
import json
import socket
import argparse
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional

# 确保 src 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    WORKSPACE, RESULTS_DIR, DATA_ANNOTATIONS,
    MODELS, IS_ROCM, parse_port_from_url,
    setup_logging, ensure_directories,
)
from data_prepare import prepare_chartqa_data
from batch_inference import VLLMInference
from baseline_ocr_llm import OCRLLMBaseline
from evaluator import Evaluator
from error_analyzer import ErrorAnalyzer
from visualizer import generate_all_plots

logger = setup_logging(__name__)


class ChartQAEvaluation:
    """ChartQA-X 全流程评测系统"""

    def __init__(self, verbose: bool = False):
        self.workspace = WORKSPACE
        self.data_path = DATA_ANNOTATIONS
        self.results_dir = RESULTS_DIR
        self.verbose = verbose

        if verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("详细日志模式已启用")
            logger.debug(f"工作目录: {WORKSPACE}")
            logger.debug(f"ROCm 环境: {IS_ROCM}")
            logger.debug(f"并发数: {os.environ.get('MAX_WORKERS', 'auto')}")

        ensure_directories()

    # ------------------------------------------------------------------
    # 服务检查
    # ------------------------------------------------------------------

    def check_services(self) -> bool:
        """检查 vLLM 服务是否在运行"""
        vllm_models = {
            name: cfg for name, cfg in MODELS.items()
            if cfg["model_type"] == "vllm" and cfg.get("api_base")
        }

        all_running = True
        for name, cfg in vllm_models.items():
            try:
                parsed = urlparse(cfg["api_base"])
                host = parsed.hostname or "localhost"
                port = parsed.port or 80
            except Exception:
                host = "localhost"
                port = parse_port_from_url(cfg["api_base"])

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                logger.info(f"✅ {name} 服务运行中 ({host}:{port})")
            else:
                logger.error(f"❌ {name} 服务未运行 ({host}:{port})")
                all_running = False

        return all_running

    # ------------------------------------------------------------------
    # 步骤 1: 数据准备
    # ------------------------------------------------------------------

    def step1_data_preparation(self, max_samples: int = None) -> list[dict]:
        """步骤 1: 下载并准备 ChartQA-X 数据"""
        logger.info("=" * 60)
        logger.info("【步骤 1/5】数据准备")
        logger.info("=" * 60)
        return prepare_chartqa_data(max_samples=max_samples)

    # ------------------------------------------------------------------
    # 步骤 2: 模型推理
    # ------------------------------------------------------------------

    def step2_inference(
        self,
        model_type: str,
        model_name: str,
        api_base: Optional[str],
        output_name: str,
        max_samples: int = None,
    ) -> list[dict]:
        """步骤 2: 运行单个模型推理"""
        logger.info("=" * 60)
        logger.info(f"【步骤 2/5】{model_name} 推理")
        logger.info("=" * 60)

        # 检查是否已完成
        result_path = os.path.join(self.results_dir, output_name, f"{output_name}_results.json")

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            with open(self.data_path, "r", encoding="utf-8") as f:
                total_data = json.load(f)
            if max_samples:
                total_data = total_data[:max_samples]

            if len(existing) >= len(total_data):
                logger.info(f"{model_name} 已全部完成 ({len(existing)}/{len(total_data)})，跳过推理")
                return existing

        # 执行推理
        if model_type == "vllm":
            inferencer = VLLMInference(
                api_base=api_base,
                model_name=model_name,
                output_dir=os.path.join(self.results_dir, output_name),
            )
            results = inferencer.run_inference(
                self.data_path,
                max_samples=max_samples,
                resume=True,
            )
        elif model_type == "baseline":
            baseline = OCRLLMBaseline()
            results = baseline.run_inference(
                self.data_path,
                os.path.join(self.results_dir, output_name),
                max_samples=max_samples,
                resume=True,
            )
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        return results

    # ------------------------------------------------------------------
    # 步骤 3: 评测
    # ------------------------------------------------------------------

    def step3_evaluation(self, model_names: list[str]) -> list[dict]:
        """步骤 3: 计算指标并生成对比报告"""
        logger.info("=" * 60)
        logger.info("【步骤 3/5】模型评测")
        logger.info("=" * 60)

        evaluator = Evaluator(self.results_dir)

        for name in model_names:
            results_path = os.path.join(
                self.results_dir, name,
                f"{name}_results.json",
            )
            if os.path.exists(results_path):
                evaluator.evaluate_results(results_path, name)
            else:
                logger.warning(f"未找到 {name} 的结果文件: {results_path}")

        comparison = evaluator.generate_comparison(model_names)
        return comparison

    # ------------------------------------------------------------------
    # 步骤 4: 误差分析
    # ------------------------------------------------------------------

    def step4_error_analysis(self, model_names: list[str]):
        """步骤 4: 四类错误分类 + 错误矩阵"""
        logger.info("=" * 60)
        logger.info("【步骤 4/5】误差分析")
        logger.info("=" * 60)

        analyzer = ErrorAnalyzer(self.results_dir)
        classified = {}

        for name in model_names:
            results_path = os.path.join(
                self.results_dir, name,
                f"{name}_results.json",
            )
            if os.path.exists(results_path):
                # 确定模型类型
                model_type = name  # "gemma4" / "llava" / "baseline"
                cases = analyzer.classify_errors(results_path, model_type)
                classified[name] = cases

                # 保存分类结果
                save_path = os.path.join(
                    self.results_dir, name,
                    f"{name}_error_classification.json",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(cases, f, indent=2, ensure_ascii=False)
            else:
                logger.warning(f"未找到 {name} 的推理结果，跳过误差分析")

        if classified:
            # 生成错误矩阵
            analyzer.generate_error_matrix(classified)
            # 生成 markdown 报告
            analyzer.generate_report(classified)

        return classified

    # ------------------------------------------------------------------
    # 步骤 5: 可视化
    # ------------------------------------------------------------------

    def step5_visualization(self):
        """步骤 5: 生成所有可视化图表"""
        logger.info("=" * 60)
        logger.info("【步骤 5/5】生成可视化图表")
        logger.info("=" * 60)
        generate_all_plots(self.results_dir)

    # ------------------------------------------------------------------
    # 全流程
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        model_names: list[str],
        max_samples: int = None,
        skip_data: bool = False,
        run_tsne: bool = False,
    ):
        """运行完整评测流水线"""
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("🚀 ChartQA-X 三模型评测系统")
        logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"目标模型: {', '.join(model_names)}")
        if max_samples:
            logger.info(f"样本限制: {max_samples}")
        if IS_ROCM:
            logger.info("ROCm 环境已检测")
        logger.info("=" * 60)

        # ---- 数据准备 ----
        if not skip_data:
            if not os.path.exists(self.data_path):
                self.step1_data_preparation(max_samples=max_samples)
            else:
                logger.info("数据已存在，跳过下载（使用 --skip-data 可完全跳过此检查）")
                if self.verbose:
                    with open(self.data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.debug(f"数据集大小: {len(data)} 个样本")

        # ---- 推理 ----
        for name in model_names:
            if name not in MODELS:
                logger.warning(f"未知模型: {name}，跳过")
                continue

            cfg = MODELS[name]

            # 如果模型是 vLLM 类型，检查服务
            if cfg["model_type"] == "vllm" and not self._quick_port_check(cfg["api_base"]):
                logger.error(f"vLLM 服务未运行 ({cfg['api_base']})，跳过 {name}")
                continue

            self.step2_inference(
                model_type=cfg["model_type"],
                model_name=cfg["model_name"],
                api_base=cfg.get("api_base"),
                output_name=cfg["output_name"],
                max_samples=max_samples,
            )

        # ---- 评测 ----
        output_names = [MODELS[n]["output_name"] for n in model_names if n in MODELS]
        comparison = self.step3_evaluation(output_names)

        # ---- 误差分析 ----
        self.step4_error_analysis(output_names)

        # ---- 可视化 ----
        self.step5_visualization()

        # ---- t-SNE（可选，耗时且吃显存） ----
        if run_tsne:
            logger.info("=" * 60)
            logger.info("【额外】t-SNE 交融机制可视化")
            logger.info("=" * 60)
            from visualize_tsne import run_tsne_analysis
            run_tsne_analysis()

        # ---- 完成 ----
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info("✅ 全流程评测完成！")
        logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总耗时: {duration}")
        logger.info(f"结果目录: {self.results_dir}")
        logger.info("=" * 60)

        # 最终摘要
        if comparison:
            print()
            print("📊 最终结果摘要:")
            print("-" * 60)
            for item in comparison:
                print(f"  {item['model']}: EM={item['exact_match']}%, F1={item['avg_f1']}%")
            print("-" * 60)
            print(f"📁 所有结果: {self.results_dir}")
            print(f"📊 对比图表: {self.results_dir}/comparison/")

    @staticmethod
    def _quick_port_check(api_base: str) -> bool:
        """快速检查端口是否可达"""
        try:
            parsed = urlparse(api_base)
            host = parsed.hostname or "localhost"
            port = parsed.port or 80
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False


# ================================================================
# CLI 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ChartQA-X 三模型评测系统 —— 对比 OCR+LLM / LLaVA / Gemma 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行示例:
  python -m src.main --models baseline --max-samples 5    # 测试基线
  python -m src.main --models gemma4 llava baseline       # 完整评测
  python -m src.main --max-samples 10 --verbose           # 详细模式 + 限制样本
  python -m src.main --tsne                               # 包含 t-SNE 分析
        """,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["gemma4", "llava", "baseline"],
        help="要评测的模型: gemma4, llava, baseline",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="限制样本数量（用于快速测试）",
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="跳过数据准备步骤",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细日志输出",
    )
    parser.add_argument(
        "--tsne", action="store_true",
        help="运行 t-SNE 交融机制可视化（耗时较长）",
    )
    parser.add_argument(
        "--version", action="version",
        version="ChartQA-X Evaluation System v1.0.0",
    )

    args = parser.parse_args()

    # 校验模型名称
    valid_models = list(MODELS.keys())
    for m in args.models:
        if m not in valid_models:
            print(f"❌ 未知模型: {m}，可用选项: {valid_models}")
            sys.exit(1)

    # 运行
    system = ChartQAEvaluation(verbose=args.verbose)

    # 检查 vLLM 服务（如果需要）
    need_vllm = any(
        MODELS[m]["model_type"] == "vllm" for m in args.models
    )
    if need_vllm and not system.check_services():
        print()
        print("⚠️  部分 vLLM 服务未运行！请先启动模型服务：")
        print()
        print("  # 终端 1 — Gemma 4 E4B")
        print("  vllm serve ./models/google/gemma-4-E4B-it/ \\")
        print("      --served-model-name gemma-4-E4B-it \\")
        print("      --port 8000 --max-model-len 8192")
        print()
        print("  # 终端 2 — LLaVA 1.5-7B")
        print("  vllm serve ./models/swift/llava-1.5-7b-hf \\")
        print("      --served-model-name llava-1.5-7b-hf \\")
        print("      --port 8001 --max-model-len 8192")
        print()
        # 不退出，允许只跑 baseline 时忽略
        if all(MODELS[m]["model_type"] == "vllm" for m in args.models):
            sys.exit(1)

    system.run_full_pipeline(
        model_names=args.models,
        max_samples=args.max_samples,
        skip_data=args.skip_data,
        run_tsne=args.tsne,
    )


if __name__ == "__main__":
    main()
