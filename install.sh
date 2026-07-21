#!/bin/bash
# ============================================================
# install.sh — ChartQA-X 评测系统一键安装脚本
# 适用: AMD ROCm 云平台 (ROCm 7.2.1)
# ============================================================
set -e

echo "=========================================="
echo "📦 ChartQA-X 评测系统 - 安装脚本 (ROCm)"
echo "=========================================="

# ---- 环境变量 ----
export WORKSPACE=${WORKSPACE:-/network-workspace}
export LMUData=$WORKSPACE/LMUData

# ---- 创建目录 ----
echo ""
echo "📁 创建目录结构..."
mkdir -p $WORKSPACE/{src,logs,results,models,data_cache}
mkdir -p $WORKSPACE/LMUData/datasets/ChartQA/{images,annotations}
mkdir -p $WORKSPACE/results/{gemma4,llava,baseline,comparison}
mkdir -p $WORKSPACE/tsne_visualization
mkdir -p $WORKSPACE/error_figures
echo "✅ 目录已创建"

# ---- 检查 Python ----
echo ""
echo "🐍 Python 版本: $(python3 --version 2>/dev/null || python --version 2>/dev/null)"

# ---- 检查 ROCm ----
echo ""
echo "🔍 检查 ROCm 环境..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm 已安装"
    rocm-smi --showproductname 2>/dev/null || echo "  (rocm-smi 详情省略)"
elif command -v amd-smi &> /dev/null; then
    echo "✅ AMD GPU 工具已安装"
else
    echo "⚠️  未检测到 ROCm 工具，可能使用 CPU 模式"
fi

# ---- 检测 ROCm 版本 ----
echo ""
echo "🔍 检测 ROCm 版本..."
ROCM_VERSION=""
if [ -f /opt/rocm/.info/version ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d. -f1,2)
elif command -v rocm-smi &> /dev/null; then
    ROCM_VERSION=$(rocm-smi --showversion 2>/dev/null | grep -oP 'ROCm\s+\K[0-9]+\.[0-9]+' | head -1)
fi
echo "   检测到 ROCm 版本: ${ROCM_VERSION:-未知}"

# ---- 配置 pip 镜像（腾讯云加速） ----
echo ""
echo "⚙️  配置 pip 镜像源..."
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/
echo "✅ pip 镜像已设置为腾讯云"

# ---- 安装 modelscope（用于下载模型） ----
echo ""
echo "📦 安装 modelscope..."
pip install modelscope -q

# ---- 卸载旧的 torchvision/torchaudio（云环境兼容性要求） ----
echo ""
echo "🧹 卸载旧版 torchvision / torchaudio..."
uv pip uninstall -y torchvision torchaudio 2>/dev/null || echo "  (未安装或已卸载，跳过)"

# ---- 安装 vLLM + PyTorch ROCm 全家桶 ----
echo ""
echo "📦 安装 vLLM + torchvision + torchaudio + fastapi (ROCm 版本)..."
# vLLM 0.23.0+rocm723 匹配 ROCm 7.2.3
VLLM_VERSION="0.23.0+rocm723"
echo "   vLLM 版本: $VLLM_VERSION"

uv pip install "vllm==${VLLM_VERSION}" torchvision torchaudio 'fastapi[standard]==0.136.0' \
    --no-cache \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://wheels.vllm.ai/rocm/ \
    -U 2>/dev/null || \
    echo "⚠️  vLLM 安装可能失败，请检查 ROCm 版本兼容性"

# ---- 安装其他 Python 依赖 ----
echo ""
echo "📦 安装其他 Python 依赖..."
uv pip install -r requirements.txt \
    --no-cache \
    -i https://mirrors.cloud.tencent.com/pypi/simple/ 2>/dev/null || \
    uv pip install -r requirements.txt --no-cache

# ---- 下载模型（可选） ----
echo ""
echo "📥 下载模型文件..."
echo ""

# Gemma 4 E4B
if [ ! -d "$WORKSPACE/models/google/gemma-4-E4B-it" ]; then
    echo "下载 Gemma 4 E4B..."
    modelscope download --model google/gemma-4-E4B-it --cache_dir "$WORKSPACE/models"
    echo "✅ Gemma 4 下载完成"
else
    echo "✅ Gemma 4 模型已存在"
fi

# LLaVA 1.5-7B
if [ ! -d "$WORKSPACE/models/swift/llava-1.5-7b-hf" ]; then
    echo "下载 LLaVA 1.5-7B（通过 huggingface-cli）..."
    pip install huggingface_hub -q
    huggingface-cli download llava-hf/llava-1.5-7b-hf \
        --local-dir "$WORKSPACE/models/swift/llava-1.5-7b-hf" \
        --resume-download 2>/dev/null || \
    echo "⚠️  LLaVA 模型下载失败，请手动下载"
else
    echo "✅ LLaVA 模型已存在"
fi

# ---- 标记安装完成 ----
touch $WORKSPACE/.deps_installed

# ---- 设置环境变量（写入 ~/.bashrc） ----
echo ""
echo "⚙️  配置环境变量..."
if ! grep -q "WORKSPACE=" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# === ChartQA-X 评测系统 ===
export WORKSPACE=/network-workspace
export LMUData=$WORKSPACE/LMUData
export CHARTQA_DATA=$LMUData/datasets/ChartQA
export RESULTS_DIR=$WORKSPACE/results
export LOGS_DIR=$WORKSPACE/logs
export MODELS_DIR=$WORKSPACE/models
export PYTHONPATH=$WORKSPACE/src:$PYTHONPATH
export MAX_WORKERS=4
EOF
    echo "✅ 环境变量已写入 ~/.bashrc"
    echo "   （请运行 source ~/.bashrc 或重新登录使其生效）"
else
    echo "ℹ️  环境变量已存在"
fi

# ---- 完成 ----
echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo ""
echo "📋 下一步:"
echo "  1. source ~/.bashrc             # 加载环境变量"
echo "  2. 启动 vLLM 服务（见下方命令）"
echo "  3. ./run.sh test                # 快速测试"
echo "  4. ./run.sh full                # 完整评测"
echo ""
echo "🚀 启动 vLLM 服务:"
echo "  # 终端 1 — Gemma 4"
echo "  vllm serve $WORKSPACE/models/google/gemma-4-E4B-it/ \\"
echo "      --served-model-name gemma-4-E4B-it \\"
echo "      --port 8000 --max-model-len 8192"
echo ""
echo "  # 终端 2 — LLaVA"
echo "  vllm serve $WORKSPACE/models/swift/llava-1.5-7b-hf \\"
echo "      --served-model-name llava-1.5-7b-hf \\"
echo "      --port 8001 --max-model-len 8192"
echo ""
echo "🌐 启动 Gradio 演示:"
echo "  python -m src.gradio_app"
echo "=========================================="
