#!/bin/bash
# ============================================================
# install_models.sh — 下载模型文件到 /models
# ============================================================
set -e

echo "=========================================="
echo "📥 下载 ChartQA-X 评测所需模型"
echo "=========================================="

MODELS_DIR="/models"
mkdir -p "$MODELS_DIR"

# ---- Gemma 4 E4B ----
GEMMA_DIR="$MODELS_DIR/google/gemma-4-E4B-it"
echo ""
echo "--- Gemma 4 E4B ---"
if [ ! -d "$GEMMA_DIR" ]; then
    echo "下载 Gemma 4 E4B 到: $GEMMA_DIR"
    modelscope download --model google/gemma-4-E4B-it --local_dir "$GEMMA_DIR"
    echo "✅ Gemma 4 下载完成"
else
    echo "✅ Gemma 4 模型已存在: $GEMMA_DIR"
fi
echo "验证:"
ls -lh "$GEMMA_DIR" | head -5

# ---- LLaVA 1.5-7B ----
LLAVA_DIR="$MODELS_DIR/swift/llava-1.5-7b-hf"
echo ""
echo "--- LLaVA 1.5-7B ---"
if [ ! -d "$LLAVA_DIR" ]; then
    echo "下载 LLaVA 1.5-7B 到: $LLAVA_DIR"
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download llava-hf/llava-1.5-7b-hf \
        --local-dir "$LLAVA_DIR"
    echo "✅ LLaVA 下载完成"
else
    echo "✅ LLaVA 模型已存在: $LLAVA_DIR"
fi
echo "验证:"
ls -lh "$LLAVA_DIR" | head -5

# ---- 磁盘用量 ----
echo ""
echo "--- /models 磁盘用量 ---"
du -sh "$MODELS_DIR"/*/

echo ""
echo "=========================================="
echo "✅ 模型下载完成！"
echo ""
echo "模型位置:"
echo "  Gemma 4:  $GEMMA_DIR"
echo "  LLaVA:    $LLAVA_DIR"
echo ""
echo "启动服务:"
echo "  vllm serve $GEMMA_DIR --port 8000 --max-model-len 4096"
echo "  vllm serve $LLAVA_DIR --port 8001 --max-model-len 4096 --trust-remote-code"
echo "=========================================="
