#!/bin/bash
# ============================================================
# install_deps.sh — 检查并安装项目依赖
# 策略：云平台已预装大部分包，此脚本逐一检查，
#       已安装的跳过，缺失的才安装。
# ============================================================
set -e

echo "=========================================="
echo "📦 ChartQA-X 评测系统 - 环境诊断与依赖安装"
echo "=========================================="

# ---- 环境变量 ----
export WORKSPACE=${WORKSPACE:-/workspace/repo/src/fine-tune/models/gemma4/MtoNLP}

# ============================================================
# 步骤 0: 系统环境诊断
# ============================================================
echo ""
echo "--- 操作系统 & 内核 ---"
if [ -f /etc/os-release ]; then grep -E "^NAME=|^VERSION=" /etc/os-release; fi
echo "内核: $(uname -r)"
echo "架构: $(uname -m)"

echo ""
echo "--- CPU ---"
if command -v lscpu &>/dev/null; then lscpu | grep -E "Model name|Core|CPU\(s\)" | head -3
else cat /proc/cpuinfo | grep "model name" | head -1; fi

echo ""
echo "--- 系统内存 ---"
free -h || cat /proc/meminfo | grep -E "MemTotal|MemAvailable"

echo ""
echo "--- 磁盘空间 ---"
df -h . | tail -1
if [ -d "$WORKSPACE" ]; then echo "WORKSPACE: $(df -h "$WORKSPACE" | tail -1 | awk '{print $4}')"; fi
if [ -d /models ]; then echo "/models:    $(df -h /models | tail -1 | awk '{print $4}')"; fi

echo ""
echo "--- GPU 信息 ---"
if command -v rocm-smi &>/dev/null; then rocm-smi --showproductname 2>/dev/null; rocm-smi --showmeminfo vram 2>/dev/null
elif command -v amd-smi &>/dev/null; then amd-smi 2>/dev/null; fi

echo ""
echo "--- ROCm 版本 ---"
if [ -f /opt/rocm/.info/version ]; then cat /opt/rocm/.info/version; fi

echo ""
echo "--- Python 环境 ---"
python3 --version 2>/dev/null || python --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch 未安装"
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || echo "vLLM 未安装"

echo ""
echo "✅ 系统诊断完成"
echo ""

# ---- pip 镜像 ----
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/ 2>/dev/null || true

# ---- 逐个检查，只装缺失的 ----
install_if_missing() {
    local pkg=$1
    local import_name=${2:-$pkg}
    if python3 -c "import $import_name" 2>/dev/null; then
        local ver=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'ok'))" 2>/dev/null)
        echo "  ✅ $pkg 已安装 ($ver)"
        return 0
    else
        echo "  ❌ $pkg 缺失 → 安装中..."
        MISSING_PKGS="$MISSING_PKGS $pkg"
        return 1
    fi
}

MISSING_PKGS=""

echo ""
echo "🔍 检查核心推理依赖..."
install_if_missing torch
install_if_missing vllm
install_if_missing transformers
install_if_missing accelerate
install_if_missing sentencepiece

echo ""
echo "🔍 检查数据处理依赖..."
install_if_missing datasets
install_if_missing pandas
install_if_missing numpy
install_if_missing "Pillow" "PIL"

echo ""
echo "🔍 检查 OCR 依赖（可选，不可用时误差分析自动回退）..."
install_if_missing paddlepaddle || true
install_if_missing paddleocr || true

echo ""
echo "🔍 检查 API 与界面依赖..."
install_if_missing openai
install_if_missing fastapi
install_if_missing gradio

echo ""
echo "🔍 检查可视化与工具..."
install_if_missing matplotlib
install_if_missing seaborn
install_if_missing "scikit-learn" "sklearn"
install_if_missing "scikit-image" "skimage"
install_if_missing tqdm

echo ""
echo "🔍 检查模型下载依赖..."
install_if_missing modelscope
install_if_missing "huggingface_hub"

# ---- 安装缺失的包 ----
if [ -n "$MISSING_PKGS" ]; then
    echo ""
    echo "📦 安装缺失的包:$MISSING_PKGS"
    uv pip install $MISSING_PKGS --no-cache \
        --extra-index-url https://wheels.vllm.ai/rocm/ 2>/dev/null || \
    uv pip install $MISSING_PKGS --no-cache
else
    echo ""
    echo "✅ 所有依赖已就绪，无需安装。"
fi

# ---- 修复 gradio-client 版本冲突（如存在） ----
if python3 -c "import gradio" 2>/dev/null; then
    uv pip check 2>/dev/null | grep -q "gradio-client" && {
        echo ""
        echo "🔧 修复 gradio 依赖冲突..."
        uv pip install gradio --no-cache --reinstall 2>/dev/null || true
    }
fi

echo ""
echo "=========================================="
echo "✅ 依赖检查完成！"
echo ""

# ---- 导出当前环境快照（云平台销毁后复现用） ----
LOCK_FILE="$WORKSPACE/requirements_cloud_lock.txt"
pip freeze > "$LOCK_FILE"
echo "📋 当前环境快照已保存: $LOCK_FILE"
echo "   云平台销毁后，在新实例上执行:"
echo "   uv pip install -r $LOCK_FILE"
echo "=========================================="
