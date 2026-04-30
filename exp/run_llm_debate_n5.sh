#!/usr/bin/env bash
# ======================================================================
#  llm_debate (N=5) — fair-budget rebuttal experiment for soo_centered_v3
#
#  目的：原 llm_debate baseline 用 agents_num=3，与我们方法 num_agents=5
#  存在 sample-budget 不对称。本脚本把 LLM-Debate 的 agents 拉到 5（rounds 仍为 2），
#  在三个模型规模 × 五个数据集上重新评估，用于公平对比。
#
#  实现方式：
#    - 新增 methods/llm_debate/configs/config_main_n5.yaml（agents_num: 5）
#    - 通过 inference.py --method_config_name config_main_n5 加载
#    - 输出文件名带 "llm_debate_n5" 后缀，避免覆盖原 llm_debate 结果
#
#  用法：
#    bash exp/run_llm_debate_n5.sh
#    bash exp/run_llm_debate_n5.sh --models "qwen25-7b-instruct"
#    bash exp/run_llm_debate_n5.sh --datasets "GSM8K MMLU-Pro"
#    bash exp/run_llm_debate_n5.sh --max_samples 100             # 快速验证
#    bash exp/run_llm_debate_n5.sh --resume                      # 跳过已有 eval
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
DEFAULT_MODELS="qwen25-7b-instruct qwen25-3b-instruct qwen25-1.5b-instruct"
DEFAULT_DATASETS="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"

METHOD_NAME="llm_debate"
METHOD_CONFIG="config_main_n5"
OUTPUT_TAG="llm_debate_n5"   # 用于输出文件名前缀，避免与原 llm_debate 结果冲突

# -------------------- 解析命令行 --------------------
EFF_MODELS="$DEFAULT_MODELS"
EFF_DATASETS="$DEFAULT_DATASETS"
MAX_SAMPLES=""
RESUME=0
OTHER_ARGS=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --models)       EFF_MODELS="${ARGS[$((i+1))]}";   ((i+=2)); continue ;;
        --datasets)     EFF_DATASETS="${ARGS[$((i+1))]}"; ((i+=2)); continue ;;
        --max_samples)  MAX_SAMPLES="${ARGS[$((i+1))]}";  ((i+=2)); continue ;;
        --resume)       RESUME=1; ((i++)) || true ;;
        *)              OTHER_ARGS+=("${ARGS[$i]}"); ((i++)) || true ;;
    esac
done

MAX_SAMPLES_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARGS=(--max_samples "$MAX_SAMPLES")
fi

# -------------------- 输出目录 --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$MAX_SAMPLES" ]]; then
    TAG="quick_${MAX_SAMPLES}"
else
    TAG="full"
fi
BASE_OUTPUT_DIR="./results_ablation/v3_llm_debate_n5_${TAG}_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/run.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "======================================================================"
echo "  LLM-Debate (N=5) — fair-budget rebuttal run"
echo "  Method:        ${METHOD_NAME}  (config: ${METHOD_CONFIG}, agents_num=5)"
echo "  Models:        ${EFF_MODELS}"
echo "  Datasets:      ${EFF_DATASETS}"
echo "  MaxSamples:    ${MAX_SAMPLES:-all}"
echo "  Output:        ${BASE_OUTPUT_DIR}"
echo "  Time:          $(date)"
echo "======================================================================"

# -------------------- 逐模型 × 数据集 --------------------
read -ra MODEL_ARRAY   <<< "$EFF_MODELS"
read -ra DATASET_ARRAY <<< "$EFF_DATASETS"

for model in "${MODEL_ARRAY[@]}"; do
    MODEL_DIR="${BASE_OUTPUT_DIR}/${model}"
    mkdir -p "$MODEL_DIR"

    echo ""
    echo "########## Model: ${model} ##########"

    for dataset in "${DATASET_ARRAY[@]}"; do
        DATASET_DIR="${MODEL_DIR}/${dataset}"
        mkdir -p "$DATASET_DIR"

        OUTPUT_FILE="${DATASET_DIR}/${OUTPUT_TAG}_infer.jsonl"
        EVAL_FILE="${DATASET_DIR}/${OUTPUT_TAG}_xverify_eval.jsonl"

        if [[ $RESUME -eq 1 && -s "$EVAL_FILE" ]]; then
            echo "[RESUME-SKIP] ${OUTPUT_TAG} | ${model} | ${dataset} (eval file exists)"
            continue
        fi

        echo ""
        echo "[INFER] ${OUTPUT_TAG} | ${model} | ${dataset} ..."
        python inference.py \
            --method_name        "${METHOD_NAME}" \
            --method_config_name "${METHOD_CONFIG}" \
            --test_dataset_name  "${dataset}" \
            --model_name         "${model}" \
            --output_path        "${OUTPUT_FILE}" \
            "${MAX_SAMPLES_ARGS[@]}" \
            "${OTHER_ARGS[@]}" || {
            echo "[WARNING] Inference ${OUTPUT_TAG}/${model}/${dataset} failed, continuing..."
            continue
        }

        if [[ ! -s "$OUTPUT_FILE" ]]; then
            echo "[NO-INFER] ${OUTPUT_TAG}/${model}/${dataset} (skip eval)"
            continue
        fi

        echo "[EVAL]  ${OUTPUT_TAG} | ${model} | ${dataset} ..."
        python evaluate.py \
            --tested_method_name     "${METHOD_NAME}" \
            --tested_dataset_name    "${dataset}" \
            --tested_mas_model_name  "${model}" \
            --tested_infer_path      "${OUTPUT_FILE}" \
            "${OTHER_ARGS[@]}" || {
            echo "[WARNING] Evaluation ${OUTPUT_TAG}/${model}/${dataset} failed, continuing..."
        }
    done
done

# -------------------- 汇总 --------------------
echo ""
echo "======================================================================"
echo "  Aggregating LLM-Debate (N=5) results"
echo "======================================================================"

for model in "${MODEL_ARRAY[@]}"; do
    MODEL_DIR="${BASE_OUTPUT_DIR}/${model}"
    echo ""
    echo "==== Model: ${model} ===="
    for dataset in "${DATASET_ARRAY[@]}"; do
        DATASET_DIR="${MODEL_DIR}/${dataset}"
        if [[ -d "$DATASET_DIR" ]]; then
            echo "---- ${dataset} ----"
            python exp/summarize.py --results_dir "${DATASET_DIR}" || true
        fi
    done
done

echo ""
echo "======================================================================"
echo "  Done at $(date)"
echo "  Results: ${BASE_OUTPUT_DIR}"
echo ""
echo "  Compare against:"
echo "    - llm_debate (N=3) original results in v3_external_full_*"
echo "    - Ours (N=5)         results in v3_tau_sweep_*"
echo "======================================================================"
