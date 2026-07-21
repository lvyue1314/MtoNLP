#!/usr/bin/env python3
"""
统一配置管理

所有路径、模型配置、环境变量集中在这里。
修改配置只需改这个文件，其他模块自动同步。
"""

import os
import sys
import logging
import platform
from datetime import datetime
from urllib.parse import urlparse


# ============================================================
# 环境变量（可通过 export 覆盖）
# ============================================================
WORKSPACE = os.environ.get("WORKSPACE", "/MtoNLP")
LMU_DATA = os.environ.get("LMUData", os.path.join(WORKSPACE, "LMUData"))
CHARTQA_DATA = os.environ.get("CHARTQA_DATA", os.path.join(LMU_DATA, "datasets", "ChartQA"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(WORKSPACE, "models"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(WORKSPACE, "results"))
LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join(WORKSPACE, "logs"))
DATA_CACHE = os.environ.get("DATA_CACHE", os.path.join(WORKSPACE, "data_cache"))

# ============================================================
# 路径常量
# ============================================================
DATA_ANNOTATIONS = os.path.join(CHARTQA_DATA, "annotations", "test.json")
DATA_IMAGES = os.path.join(CHARTQA_DATA, "images")

# 如果需要镜像加速 HuggingFace
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ============================================================
# 模型配置
# ============================================================
MODELS = {
    "gemma4": {
        "model_type": "vllm",                # 通过 vLLM API 调用
        "model_name": "gemma-4-E4B-it",      # vLLM served-model-name
        "api_base": "http://localhost:8000/v1",
        "output_name": "gemma4",
        "model_path": os.path.join(MODELS_DIR, "google", "gemma-4-E4B-it"),
        "description": "Gemma 4 E4B — 原生多模态模型",
        "visual_token_count": 280,            # Gemma 4 默认视觉 Token 数量
    },
    "llava": {
        "model_type": "vllm",
        "model_name": "llava-1.5-7b-hf",
        "api_base": "http://localhost:8001/v1",
        "output_name": "llava",
        "model_path": os.path.join(MODELS_DIR, "swift", "llava-1.5-7b-hf"),
        "description": "LLaVA 1.5 7B — 外挂式多模态模型",
        "hf_model_id": "llava-hf/llava-1.5-7b-hf",
    },
    "baseline": {
        "model_type": "baseline",
        "model_name": "OCR+LLM Baseline",
        "api_base": None,
        "output_name": "baseline",
        "model_path": None,
        "description": "PaddleOCR + Qwen2.5-0.5B — 纯文本基线",
        "hf_model_id": "Qwen/Qwen2.5-0.5B",
    },
}


# ============================================================
# 辅助函数
# ============================================================

def parse_port_from_url(url: str) -> int:
    """从 URL 中解析端口号"""
    if not url:
        return 80
    try:
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80
    except Exception:
        # 回退：在字符串中查找数字端口
        import re
        match = re.search(r":(\d+)", url)
        return int(match.group(1)) if match else 80


def get_optimal_workers() -> int:
    """
    根据 GPU 显存自动调整并发数。

    注意: vLLM 服务本身已经占用大量显存（通常 8-16GB），
    并发请求主要消耗推理时的 KV cache。保守估计每 worker 需要 ~6GB 可用显存，
    且总并发数不超过 4（避免 vLLM 服务端过载排队）。
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory

            # 保守估计：假设 vLLM 已占用约 60% 显存用于模型权重
            # 剩余可用显存 // 6GB per worker = 安全并发数
            available_for_inference = gpu_memory * 0.4  # 40% 可用于并发请求
            workers = max(1, min(4, int(available_for_inference // (6 * 1024 ** 3))))

            print(f"[INFO] GPU 显存: {gpu_memory // 1024 ** 3} GB, "
                  f"估计可用: {available_for_inference // 1024 ** 3:.0f} GB, "
                  f"安全并发数: {workers}")
            return workers
    except Exception as e:
        print(f"[WARN] GPU 检测失败: {e}, 使用默认并发数 2")
    print("[INFO] 未检测到 GPU, 使用 CPU 模式")
    return 1


def detect_rocm() -> bool:
    """检测是否为 ROCm 环境"""
    try:
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip:
                print(f"[INFO] 检测到 ROCm 环境: HIP {torch.version.hip}")
                return True
            print(f"[INFO] GPU: {torch.cuda.get_device_properties(0).name}")
            return True
    except Exception as e:
        print(f"[WARN] ROCm 检测失败: {e}")
    return False


def setup_logging(name: str = None, verbose: bool = False) -> logging.Logger:
    """配置日志系统（同时输出到文件和控制台）"""
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(
        LOGS_DIR,
        f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件 handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)

        # 控制台 handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

    return logger


def resolve_gemma_path() -> str:
    """
    解析 Gemma 4 模型的实际磁盘路径。

    HuggingFace/modelscope 下载的目录名可能与 config 中的不完全一致
    （大小写差异），此函数尝试所有可能的路径变体。
    """
    candidates = [
        os.path.join(MODELS_DIR, "google", "gemma-4-E4B-it"),
        os.path.join(MODELS_DIR, "google", "gemma-4-e4b-it"),
        os.path.join(MODELS_DIR, "google", "gemma-4-E4B-it"),
        os.path.join(MODELS_DIR, "gemma-4-E4B-it"),
        os.path.join(MODELS_DIR, "google", "gemma-4-e4b-it"),
        "/workspace/models/google/gemma-4-E4B-it",   # 另一个常见挂载点
        "/workspace/models/google/gemma-4-e4b-it",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    # 回退到默认路径（即使不存在，后续模块会提示下载）
    return candidates[0]


def ensure_directories():
    """确保所有必要目录存在"""
    dirs = [
        WORKSPACE,
        LMU_DATA,
        CHARTQA_DATA,
        DATA_IMAGES,
        os.path.join(CHARTQA_DATA, "annotations"),
        RESULTS_DIR,
        os.path.join(RESULTS_DIR, "gemma4"),
        os.path.join(RESULTS_DIR, "llava"),
        os.path.join(RESULTS_DIR, "baseline"),
        os.path.join(RESULTS_DIR, "comparison"),
        LOGS_DIR,
        MODELS_DIR,
        DATA_CACHE,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 运行时自动检测
# ============================================================
IS_ROCM = detect_rocm()
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", get_optimal_workers()))
# vLLM并发默认更保守（vLLM服务端有自身排队机制，客户端并发过高反而导致超时）
VLLM_CONCURRENT = int(os.environ.get("VLLM_CONCURRENT", min(MAX_WORKERS, 2)))

# 启动时创建目录
ensure_directories()
