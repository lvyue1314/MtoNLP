#!/bin/bash
# ============================================================
# run.sh — ChartQA-X 评测系统一键运行脚本
# ============================================================
set -e

cd "$(dirname "$0")"  # 切换到脚本所在目录

# ---- 环境变量 ----
export WORKSPACE=${WORKSPACE:-/network-workspace}
export LMUData=$WORKSPACE/LMUData
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$WORKSPACE/src:$PYTHONPATH
export MAX_WORKERS=${MAX_WORKERS:-4}

# ============================================================
# 帮助
# ============================================================

show_help() {
    echo "ChartQA-X 三模型评测系统 — 运行脚本"
    echo ""
    echo "用法: ./run.sh [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  test          测试模式（基线模型，5 个样本）"
    echo "  quick         快速模式（所有模型，10 个样本）"
    echo "  full          完整模式（所有模型，所有样本）"
    echo "  gemma         只跑 Gemma 4"
    echo "  llava         只跑 LLaVA"
    echo "  baseline      只跑 OCR+LLM 基线"
    echo "  eval          只运行评测（已有推理结果）"
    echo "  viz           只生成可视化图表"
    echo "  tsne          运行 t-SNE 交融机制分析"
    echo "  gradio        启动 Gradio 演示界面"
    echo "  help          显示此帮助"
    echo ""
    echo "高级选项可直接传给 Python:"
    echo "  ./run.sh --verbose --max-samples 20"
    echo ""
}

# ============================================================
# 前置检查
# ============================================================

check_deps() {
    if [ ! -f "$WORKSPACE/.deps_installed" ]; then
        echo "📦 首次运行，安装依赖..."
        bash install.sh
    fi
}

check_pythonpath() {
    if [[ ":$PYTHONPATH:" != *":$WORKSPACE/src:"* ]]; then
        export PYTHONPATH=$WORKSPACE/src:$PYTHONPATH
    fi
}

# ============================================================
# 运行
# ============================================================

run_python() {
    cd "$WORKSPACE"
    python -m src.main "$@"
}

# ============================================================
# 主逻辑
# ============================================================

case "${1:-help}" in
    test)
        check_deps
        run_python --models baseline --max-samples 5
        ;;
    quick)
        check_deps
        run_python --max-samples 10
        ;;
    full)
        check_deps
        run_python
        ;;
    gemma)
        check_deps
        run_python --models gemma4
        ;;
    llava)
        check_deps
        run_python --models llava
        ;;
    baseline)
        check_deps
        run_python --models baseline
        ;;
    eval)
        check_deps
        echo "📊 只运行评测..."
        cd "$WORKSPACE"
        python -c "
from src.evaluator import Evaluator
from src.error_analyzer import ErrorAnalyzer
from src.visualizer import generate_all_plots

evaluator = Evaluator()
for name in ['gemma4', 'llava', 'baseline']:
    evaluator.evaluate_results(f'results/{name}/{name}_results.json', name)
comparison = evaluator.generate_comparison(['gemma4', 'llava', 'baseline'])

analyzer = ErrorAnalyzer()
classified = {}
for name in ['gemma4', 'llava', 'baseline']:
    path = f'results/{name}/{name}_results.json'
    import os, json
    if os.path.exists(path):
        classified[name] = analyzer.classify_errors(path, name)
if classified:
    analyzer.generate_error_matrix(classified)
    analyzer.generate_report(classified)

generate_all_plots()
print('评测完成！')
"
        ;;
    viz)
        check_deps
        echo "📊 生成可视化图表..."
        cd "$WORKSPACE"
        python -c "from src.visualizer import generate_all_plots; generate_all_plots()"
        ;;
    tsne)
        check_deps
        shift   # 去掉 "tsne"，剩余参数透传给 Python
        echo "🧬 运行 t-SNE 交融机制分析..."
        cd "$WORKSPACE"
        python -m src.visualize_tsne "$@"   # 支持: ./run.sh tsne --max-samples 20 --skip-llava
        ;;
    gradio)
        shift
        echo "🌐 启动 Gradio 演示界面..."
        cd "$WORKSPACE"
        python -m src.gradio_app "$@"       # 支持: ./run.sh gradio --port 8080 --share
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        # 传递所有参数给 Python
        check_deps
        run_python "$@"
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 完成！结果保存在: $WORKSPACE/results/"
echo "=========================================="
