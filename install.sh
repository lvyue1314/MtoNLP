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

# ============================================================
# 步骤 0: 系统环境诊断
# ============================================================
echo ""
echo "=========================================="
echo "🔍 步骤 0: 系统环境诊断"
echo "=========================================="

# 0.1 操作系统与内核
echo ""
echo "--- 操作系统 & 内核 ---"
if [ -f /etc/os-release ]; then
    grep -E "^NAME=|^VERSION=" /etc/os-release 2>/dev/null
fi
echo "内核: $(uname -r)"
echo "架构: $(uname -m)"

# 0.2 CPU
echo ""
echo "--- CPU ---"
if command -v lscpu &> /dev/null; then
    lscpu 2>/dev/null | grep -E "Model name|Socket|Thread|Core|CPU\(s\)" | head -5
else
    cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1
fi

# 0.3 系统内存
echo ""
echo "--- 系统内存 ---"
free -h 2>/dev/null || cat /proc/meminfo 2>/dev/null | grep -E "MemTotal|MemAvailable"

# 0.4 磁盘空间
echo ""
echo "--- 磁盘空间 ---"
echo "当前目录: $(pwd)"
df -h . 2>/dev/null
echo ""
echo "/network-workspace (如果存在):"
df -h /network-workspace 2>/dev/null || echo "  (未挂载)"

# 磁盘空间预警：模型 ~30GB + 数据 ~1GB + 缓存 ~20GB ≈ 需要 60GB
echo ""
echo "--- 磁盘空间检查 ---"
REQUIRED_GB=60
# 优先检查 /network-workspace，其次检查当前目录所在分区
if [ -d /network-workspace ]; then
    AVAIL_KB=$(df -k /network-workspace 2>/dev/null | tail -1 | awk '{print $4}')
else
    AVAIL_KB=$(df -k . 2>/dev/null | tail -1 | awk '{print $4}')
fi

if [ -n "$AVAIL_KB" ]; then
    AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
    echo "可用空间: ${AVAIL_GB} GB  |  需求: ${REQUIRED_GB} GB"

    if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  ⚠️  磁盘空间不足警告                                        ║"
        echo "║                                                            ║"
        echo "║  可用空间: ${AVAIL_GB} GB                                   ║"
        echo "║  建议空间: ${REQUIRED_GB} GB（模型 30G + 数据 1G + 缓存）     ║"
        echo "║  缺口:     $((REQUIRED_GB - AVAIL_GB)) GB                                    ║"
        echo "║                                                            ║"
        echo "║  安装可能因磁盘满而失败，建议先清理空间。                      ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo ""
        echo "按 Ctrl+C 取消安装，或等待 10 秒后自动继续..."
        sleep 10
        echo "⚠️  继续安装（磁盘空间不足，可能出现下载失败）"
    else
        echo "✅ 磁盘空间充足"
    fi
else
    echo "⚠️  无法检测可用空间，跳过空间检查"
fi

# 0.5 GPU 信息（最重要）
echo ""
echo "--- GPU 信息 ---"
if command -v rocm-smi &> /dev/null; then
    echo "GPU 列表:"
    rocm-smi --showproductname 2>/dev/null
    echo ""
    echo "显存使用:"
    rocm-smi --showmeminfo vram 2>/dev/null
elif command -v amd-smi &> /dev/null; then
    amd-smi 2>/dev/null
elif command -v lspci &> /dev/null; then
    lspci 2>/dev/null | grep -iE "vga|3d|display|amd"
else
    echo "⚠️  无法检测 GPU（缺少 rocm-smi / amd-smi / lspci）"
fi

# 0.6 ROCm 版本
echo ""
echo "--- ROCm 版本 ---"
if [ -f /opt/rocm/.info/version ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1)
    echo "ROCm 版本: $ROCM_VERSION"
elif command -v rocm-smi &> /dev/null; then
    ROCM_VERSION=$(rocm-smi --showversion 2>/dev/null | grep -oP 'ROCm\s+\K[0-9.]+' | head -1)
    echo "ROCm 版本: ${ROCM_VERSION:-未知}"
else
    echo "⚠️  未检测到 ROCm"
fi

# 0.7 Python 环境
echo ""
echo "--- Python 环境 ---"
echo "Python: $(python3 --version 2>/dev/null || python --version 2>/dev/null)"
echo "pip:    $(pip3 --version 2>/dev/null || pip --version 2>/dev/null)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "CUDA 可用: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "GPU 名称: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
    echo "GPU 显存: $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")' 2>/dev/null || echo 'N/A')"
fi

echo ""
echo "=========================================="
echo "✅ 系统诊断完成"
echo "=========================================="

# ---- 创建目录 ----
echo ""
echo "📁 创建目录结构..."
mkdir -p $WORKSPACE/{src,logs,results,models,data_cache}
mkdir -p $WORKSPACE/LMUData/datasets/ChartQA/{images,annotations}
mkdir -p $WORKSPACE/results/{gemma4,llava,baseline,comparison}
mkdir -p $WORKSPACE/tsne_visualization
mkdir -p $WORKSPACE/error_figures
echo "✅ 目录已创建"

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
