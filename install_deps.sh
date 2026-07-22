#!/bin/bash
# ============================================================
# install_deps.sh — 检查并安装项目依赖
# 策略：云平台已预装大部分包，此脚本逐一检查，
#       已安装的跳过，缺失的才安装。
# ============================================================
set -e

echo "=========================================="
echo "📦 ChartQA-X 评测系统 - 依赖安装"
echo "=========================================="

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
echo "=========================================="
