#!/usr/bin/env bash
# ======================================================================
#  soo_centered_v3 单方法实验脚本
#
#  只跑 soo_centered_v3，然后与指定的 baseline 结果目录逐数据集对比。
#  不会重跑 selforg/soo/dylan_* 等已有算法 — 依赖 baseline 目录中的现成评估。
#
#  用法:
#    bash exp/run_v3.sh --quick
#        30 条样本快速验证，默认与 20260416_111010 对比
#
#    bash exp/run_v3.sh --full
#        全量样本，默认与 20260416_152034 对比
#
#    bash exp/run_v3.sh --max_samples 100 --compare_with results_ablation/20260416_152034
#        自定义样本量和 baseline
#
#    bash exp/run_v3.sh --quick --datasets "GSM8K AQUA-RAT"
#        只跑指定数据集
#
#  可选参数:
#    --model_name       默认 qwen25-3b-instruct
#    --datasets         默认 "GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
#    --compare_with     对比的 baseline 目录（相对 MASLab/）
#    --tag              输出目录后缀（默认 quick/full/custom）
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f scripts/prepend_nvjitlink_ld_path.sh ]]; then
    source scripts/prepend_nvjitlink_ld_path.sh
fi
export TOKENIZERS_PARALLELISM=false

# -------------------- 默认参数 --------------------
DEFAULT_MODEL="qwen25-3b-instruct"
DEFAULT_DATASETS="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
METHOD="soo_centered_v3"

# baselines
QUICK_BASELINE="results_ablation/20260416_111010"   # 30 samples, has all methods incl. dylan/soo_math
FULL_BASELINE="results_ablation/20260416_152034"    # full samples (500/254/30), has all methods

# -------------------- 解析命令行参数 --------------------
EFF_MODEL="$DEFAULT_MODEL"
EFF_DATASETS="$DEFAULT_DATASETS"
MAX_SAMPLES=""
COMPARE_WITH=""
TAG=""
MODE=""
OTHER_ARGS=()
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --quick)
            MODE="quick"
            ((i++)) || true ;;
        --full)
            MODE="full"
            ((i++)) || true ;;
        --model_name)
            EFF_MODEL="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --test_dataset_name|--datasets)
            EFF_DATASETS="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --max_samples)
            MAX_SAMPLES="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --compare_with)
            COMPARE_WITH="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --tag)
            TAG="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        *)
            OTHER_ARGS+=("${ARGS[$i]}")
            ((i++)) || true ;;
    esac
done

# 根据 MODE 设置默认 max_samples 和 baseline
case "$MODE" in
    quick)
        : "${MAX_SAMPLES:=30}"
        : "${COMPARE_WITH:=$QUICK_BASELINE}"
        : "${TAG:=quick}"
        ;;
    full)
        # no cap -> empty MAX_SAMPLES
        : "${COMPARE_WITH:=$FULL_BASELINE}"
        : "${TAG:=full}"
        ;;
    *)
        # No mode specified -> require at least --compare_with or --max_samples
        if [[ -z "$COMPARE_WITH" ]]; then
            echo "[ERROR] no --quick / --full / --compare_with specified"
            echo "       use: bash exp/run_v3.sh --quick   (30 samples vs $QUICK_BASELINE)"
            echo "       or:  bash exp/run_v3.sh --full    (full samples vs $FULL_BASELINE)"
            exit 1
        fi
        : "${TAG:=custom}"
        ;;
esac

MAX_SAMPLES_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARGS=(--max_samples "$MAX_SAMPLES")
fi

# -------------------- 输出目录 --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./results_ablation/v3_${TAG}_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/v3_run.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "======================================================================"
echo "  soo_centered_v3 Run"
echo "  Model:        ${EFF_MODEL}"
echo "  Datasets:     ${EFF_DATASETS}"
echo "  MaxSamples:   ${MAX_SAMPLES:-all}"
echo "  Baseline:     ${COMPARE_WITH}"
echo "  Output:       ${BASE_OUTPUT_DIR}"
echo "  Time:         $(date)"
echo "======================================================================"
echo ""

# -------------------- 逐数据集运行 --------------------
read -ra DATASET_ARRAY <<< "$EFF_DATASETS"

for dataset in "${DATASET_ARRAY[@]}"; do
    DATASET_DIR="${BASE_OUTPUT_DIR}/${dataset}"
    mkdir -p "$DATASET_DIR"

    echo "########## ${dataset} ##########"

    # 推理
    OUTPUT_FILE="${DATASET_DIR}/${METHOD}_infer.jsonl"
    echo "[INFER] ${METHOD} on ${dataset} ..."
    python inference.py \
        --method_name "${METHOD}" \
        --test_dataset_name "${dataset}" \
        --model_name "${EFF_MODEL}" \
        --output_path "${OUTPUT_FILE}" \
        "${MAX_SAMPLES_ARGS[@]}" \
        "${OTHER_ARGS[@]}" || {
        echo "[WARNING] Inference ${METHOD}/${dataset} failed, continuing..."
        continue
    }

    # 评估
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "[EVAL]  ${METHOD} on ${dataset} ..."
        python evaluate.py \
            --tested_method_name "${METHOD}" \
            --tested_dataset_name "${dataset}" \
            --tested_mas_model_name "${EFF_MODEL}" \
            --tested_infer_path "${OUTPUT_FILE}" \
            "${OTHER_ARGS[@]}" || {
            echo "[WARNING] Evaluation ${METHOD}/${dataset} failed, continuing..."
        }
    fi
    echo ""
done

# -------------------- 对比与汇总 --------------------
# baseline 路径解析：支持绝对路径，或相对于 PROJECT_ROOT / 其父目录。
# 历史数据实际落在 D:/Projects/MASLab/results_ablation/（PROJECT_ROOT 的父目录），
# 而 run_ablation.sh 默认写到 PROJECT_ROOT/results_ablation/ 下，因此两处都检查。
resolve_baseline() {
    local path="$1"
    if [[ "$path" = /* ]] || [[ "$path" =~ ^[A-Za-z]:[/\\] ]]; then
        echo "$path"; return
    fi
    if [[ -d "${PROJECT_ROOT}/${path}/$(ls "${PROJECT_ROOT}/${path}" 2>/dev/null | head -n1)" ]]; then
        # exists AND has subdir(s)
        echo "${PROJECT_ROOT}/${path}"; return
    fi
    if [[ -d "${PROJECT_ROOT}/../${path}" ]]; then
        echo "$(cd "${PROJECT_ROOT}/../${path}" && pwd)"; return
    fi
    if [[ -d "${PROJECT_ROOT}/${path}" ]]; then
        echo "${PROJECT_ROOT}/${path}"; return
    fi
    echo "${PROJECT_ROOT}/${path}"
}
BASELINE_FULL="$(resolve_baseline "${COMPARE_WITH}")"

echo "======================================================================"
echo "  v3 vs baseline comparison"
echo "======================================================================"
if [[ -d "$BASELINE_FULL" ]]; then
    python exp/compare_v3.py \
        --v3_dir "${BASE_OUTPUT_DIR}" \
        --baseline_dir "${BASELINE_FULL}" \
        --datasets "${EFF_DATASETS}" || {
        echo "[WARNING] compare_v3.py failed, dumping v3 raw results instead:"
        python exp/summarize.py --results_dir "${BASE_OUTPUT_DIR}" || true
    }
else
    echo "[WARNING] baseline dir not found: ${BASELINE_FULL}"
    echo "  dumping v3 raw results only:"
    python exp/summarize.py --results_dir "${BASE_OUTPUT_DIR}" || true
fi

echo ""
echo "======================================================================"
echo "  Finished at $(date)"
echo "  v3 results:     ${BASE_OUTPUT_DIR}"
echo "  Compared with:  ${COMPARE_WITH}"
echo "======================================================================"
