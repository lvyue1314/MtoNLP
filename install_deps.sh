#!/bin/bash
# ============================================================
# install_deps.sh — 安装项目额外依赖
# 云平台已预装 torch, vllm, transformers, datasets 等，
# 此处仅安装缺失的 4 个包。
# ============================================================
set -e

echo "=========================================="
echo "📦 ChartQA-X 评测系统 - 依赖安装"
echo "=========================================="

# ---- 环境变量 ----
export WORKSPACE=${WORKSPACE:-/workspace/repo/src/fine-tune/models/gemma4}

# ============================================================
# 步骤 0: 系统环境诊断
# ============================================================
echo ""
echo "=========================================="
echo "🔍 步骤 0: 系统环境诊断"
echo "=========================================="

echo ""
echo "--- 操作系统 & 内核 ---"
if [ -f /etc/os-release ]; then
    grep -E "^NAME=|^VERSION=" /etc/os-release
fi
echo "内核: $(uname -r)"
echo "架构: $(uname -m)"

echo ""
echo "--- CPU ---"
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "Model name|Socket|Thread|Core|CPU\(s\)" | head -5
else
    cat /proc/cpuinfo | grep "model name" | head -1
fi

echo ""
echo "--- 系统内存 ---"
free -h || cat /proc/meminfo | grep -E "MemTotal|MemAvailable"

echo ""
echo "--- 磁盘空间 ---"
echo "当前目录: $(pwd)"
df -h .
echo ""
echo "/models (模型存放目录):"
df -h /models 2>/dev/null || echo "  (尚未创建)"

# 磁盘空间预警
echo ""
echo "--- 磁盘空间检查 ---"
REQUIRED_GB=40
if [ -d "$WORKSPACE" ]; then
    AVAIL_KB=$(df -k "$WORKSPACE" | tail -1 | awk '{print $4}')
else
    AVAIL_KB=$(df -k . | tail -1 | awk '{print $4}')
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
        echo "║  建议空间: ${REQUIRED_GB} GB（模型 30G + 数据 + 缓存）        ║"
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

echo ""
echo "--- GPU 信息 ---"
if command -v rocm-smi &> /dev/null; then
    echo "GPU 列表:"
    rocm-smi --showproductname
    echo ""
    echo "显存使用:"
    rocm-smi --showmeminfo vram
elif command -v amd-smi &> /dev/null; then
    amd-smi
elif command -v lspci &> /dev/null; then
    lspci | grep -iE "vga|3d|display|amd"
else
    echo "⚠️  无法检测 GPU"
fi

echo ""
echo "--- ROCm 版本 ---"
if [ -f /opt/rocm/.info/version ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version | head -1)
    echo "ROCm 版本: $ROCM_VERSION"
elif command -v rocm-smi &> /dev/null; then
    ROCM_VERSION=$(rocm-smi --showversion | grep -oP 'ROCm\s+\K[0-9.]+' | head -1)
    echo "ROCm 版本: ${ROCM_VERSION:-未知}"
else
    echo "⚠️  未检测到 ROCm"
fi

echo ""
echo "--- Python 环境 ---"
echo "Python: $(python3 --version || python --version)"
echo "pip:    $(pip3 --version || pip --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "CUDA 可用: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "GPU 名称: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    echo "GPU 显存: $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")' 2>/dev/null)"
fi

# 检查关键包是否已预装
echo ""
echo "--- 关键依赖检查 ---"
for pkg in vllm transformers datasets accelerate fastapi openai; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null)
        echo "  ✅ $pkg ($ver)"
    else
        echo "  ❌ $pkg 未安装！"
    fi
done

echo ""
echo "=========================================="
echo "✅ 系统诊断完成"
echo "=========================================="

# ============================================================
# 步骤 1: 配置 pip 镜像 + 安装缺失依赖
# ============================================================
echo ""
echo "=========================================="
echo "📦 步骤 1: 安装缺失的 Python 依赖"
echo "=========================================="

pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/
echo "✅ pip 镜像已设置为腾讯云"

echo ""
echo "📦 安装 paddlepaddle + paddleocr + gradio + modelscope ..."
uv pip install \
    paddlepaddle \
    paddleocr \
    gradio \
    modelscope \
    --no-cache

echo ""
echo "=========================================="
echo "✅ 依赖安装完成！"
echo ""
echo "已安装:"
echo "  - paddlepaddle  (OCR 引擎)"
echo "  - paddleocr     (文字识别)"
echo "  - gradio        (Web 演示界面)"
echo "  - modelscope    (模型下载工具)"
echo ""
echo "以下包已由云平台预装，无需重复安装:"
echo "  torch, vllm, transformers, datasets, accelerate"
echo "  fastapi, openai, matplotlib, seaborn, scikit-learn"
echo "  pandas, numpy, Pillow, tqdm, sentencepiece, huggingface_hub"
echo ""
echo "📋 下一步:"
echo "  ./install_models.sh   # 下载模型文件到 /models"
echo "=========================================="
