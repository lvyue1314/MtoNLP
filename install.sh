#!/bin/bash
# ============================================================
# install.sh — ChartQA-X 评测系统一键安装脚本
# 适用: AMD ROCm 云平台 (ROCm 7.14.0)
# ============================================================
set -e

echo "=========================================="
echo "📦 ChartQA-X 评测系统 - 安装脚本 (ROCm)"
echo "=========================================="

# ---- 环境变量 ----
export WORKSPACE=${WORKSPACE:-/MtoNLP}
export LMUData=$WORKSPACE/LMUData

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
echo "\$WORKSPACE ($WORKSPACE) 所在分区:"
df -h "$WORKSPACE" || df -h .

# 磁盘空间预警
echo ""
echo "--- 磁盘空间检查 ---"
REQUIRED_GB=60
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
    echo "⚠️  无法检测 GPU（缺少 rocm-smi / amd-smi / lspci）"
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
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' || echo '未安装')"
echo "CUDA 可用: $(python3 -c 'import torch; print(torch.cuda.is_available())' || echo 'N/A')"
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "GPU 名称: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU 显存: $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")')"
fi

echo ""
echo "=========================================="
echo "✅ 系统诊断完成"
echo "=========================================="

# ============================================================
# 步骤 0.5: 创建 Python 虚拟环境
# ============================================================
VENV_DIR="$WORKSPACE/.venv"
echo ""
echo "=========================================="
echo "🐍 创建 Python 虚拟环境"
echo "=========================================="

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✅ 虚拟环境已创建: $VENV_DIR"
else
    echo "✅ 虚拟环境已存在: $VENV_DIR"
fi

# 激活虚拟环境（后续所有 pip/uv 命令都在此环境中执行）
source "$VENV_DIR/bin/activate"
echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# ---- 创建目录 ----
echo ""
echo "📁 创建目录结构..."
mkdir -p $WORKSPACE/{src,logs,results,models,data_cache}
mkdir -p $WORKSPACE/LMUData/datasets/ChartQA/{images,annotations}
mkdir -p $WORKSPACE/results/{gemma4,llava,baseline,comparison}
mkdir -p $WORKSPACE/tsne_visualization
mkdir -p $WORKSPACE/error_figures
echo "✅ 目录已创建"

# ============================================================
# 步骤 1: 配置 pip 镜像源
# ============================================================
echo ""
echo "=========================================="
echo "📦 步骤 1: 配置 pip 镜像源 + 安装 modelscope"
echo "=========================================="

pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/
echo "✅ pip 镜像已设置为腾讯云"

pip install modelscope
echo "✅ modelscope 安装完成"

# ============================================================
# 步骤 2: 下载模型文件
# ============================================================
echo ""
echo "=========================================="
echo "📥 步骤 2: 下载模型文件"
echo "=========================================="

# 定义绝对路径（防止 WORKSPACE 变量嵌套）
BASE_MODEL_DIR="/MtoNLP/models"
mkdir -p "$BASE_MODEL_DIR"

# Gemma 4 E4B
GEMMA_DIR="$BASE_MODEL_DIR/google/gemma-4-E4B-it"
if [ ! -d "$GEMMA_DIR" ]; then
    echo "下载 Gemma 4 E4B 到: $GEMMA_DIR"
    modelscope download --model google/gemma-4-E4B-it --local_dir "$GEMMA_DIR"
    echo "✅ Gemma 4 下载完成"
else
    echo "✅ Gemma 4 模型已存在: $GEMMA_DIR"
fi
echo "验证 Gemma 4 模型文件:"
ls -lh "$GEMMA_DIR" | head -5

# LLaVA 1.5-7B
LLAVA_DIR="$BASE_MODEL_DIR/swift/llava-1.5-7b-hf"
if [ ! -d "$LLAVA_DIR" ]; then
    echo ""
    echo "下载 LLaVA 1.5-7B 到: $LLAVA_DIR"
    pip install huggingface_hub -q
    export HF_ENDPOINT=https://hf-mirror.com
    hf download llava-hf/llava-1.5-7b-hf \
        --local-dir "$LLAVA_DIR"
    echo "✅ LLaVA 下载完成"
else
    echo "✅ LLaVA 模型已存在: $LLAVA_DIR"
fi
echo "验证 LLaVA 模型文件:"
ls -lh "$LLAVA_DIR" | head -5

# ============================================================
# 步骤 3: 安装 vLLM + PyTorch ROCm 全家桶
# ============================================================
echo ""
echo "=========================================="
echo "📦 步骤 3: 安装 vLLM + PyTorch ROCm 全家桶"
echo "=========================================="

# 先卸载旧版（云环境兼容性要求）
echo "🧹 卸载旧版 torchvision / torchaudio..."
uv pip uninstall -- -y torchvision torchaudio || echo "  (未安装或已卸载，跳过)"

# 一键安装 vLLM + torch 相关 + fastapi
echo ""
echo "📦 安装 vLLM + torchvision + torchaudio + fastapi..."
# uv pip install 'vllm==0.23.0+rocm723' torchvision torchaudio 'fastapi[standard]==0.136.0' \
#     --no-cache \
#     --index-url https://mirrors.aliyun.com/pypi/simple/ \
#     --extra-index-url https://wheels.vllm.ai/rocm/ \
#     -U
# 放宽版本约束，安装 0.23.x 系列的最新兼容版本
uv pip install 'vllm>=0.23.0,<0.24.0' torchvision torchaudio 'fastapi[standard]==0.136.0' \
    --no-cache \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://wheels.vllm.ai/rocm/ \
    --index-strategy unsafe-best-match \
    -U

echo "✅ vLLM + PyTorch ROCm 全家桶安装完成"

# ============================================================
# 步骤 4: 安装其他 Python 依赖（原 requirements.txt 内容）
# ============================================================
echo ""
echo "=========================================="
echo "📦 步骤 4: 安装其他 Python 依赖"
echo "=========================================="

uv pip install \
    datasets \
    pandas \
    numpy \
    Pillow \
    paddlepaddle \
    paddleocr \
    openai \
    gradio \
    transformers \
    accelerate \
    sentencepiece \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    huggingface_hub \
    urllib3 \
    --no-cache \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://wheels.vllm.ai/rocm/ \
    -U

echo "✅ 所有 Python 依赖安装完成"

# ============================================================
# 步骤 5: 标记安装完成 + 写入环境变量
# ============================================================
touch $WORKSPACE/.deps_installed

echo ""
echo "=========================================="
echo "⚙️  步骤 5: 配置环境变量"
echo "=========================================="

if ! grep -q "WORKSPACE=" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# === ChartQA-X 评测系统 ===
export WORKSPACE=/MtoNLP
export LMUData=$WORKSPACE/LMUData
export CHARTQA_DATA=$LMUData/datasets/ChartQA
export RESULTS_DIR=$WORKSPACE/results
export LOGS_DIR=$WORKSPACE/logs
export MODELS_DIR=$WORKSPACE/models
export PYTHONPATH=$WORKSPACE/src:$PYTHONPATH
export MAX_WORKERS=4

# 激活项目虚拟环境
if [ -f "$WORKSPACE/.venv/bin/activate" ]; then
    source "$WORKSPACE/.venv/bin/activate"
fi
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
echo "  1. source ~/.bashrc                  # 加载环境变量（含 venv 自动激活）"
echo "  2. 启动 vLLM 服务（⚠️ 两个模型不能同时启动，显存不够）"
echo "  3. ./run.sh test"
echo ""
echo "🚀 启动 Gemma 4 服务:"
echo "  cd $WORKSPACE"
echo "  vllm serve ./models/google/gemma-4-E4B-it/ \\"
echo "      --served-model-name gemma-4-E4B-it \\"
echo "      --port 8000 --max-model-len 4096"
echo ""
echo "🚀 启动 LLaVA 服务（Gemma 4 跑完 Ctrl+C 停止后再启动）:"
echo "  vllm serve ./models/swift/llava-1.5-7b-hf \\"
echo "      --served-model-name llava-1.5-7b-hf \\"
echo "      --port 8001 --max-model-len 4096 --trust-remote-code"
echo ""
echo "🌐 启动 Gradio 演示:"
echo "  python -m src.gradio_app"
echo "=========================================="
