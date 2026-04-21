#!/usr/bin/env bash
# ======================================================================
#  soo_centered_v3: 外部对比 baseline 一次全跑
#
#  覆盖 10 个外部方法（不含我们自家 selforg / soo / soo_centered_* 消融线）：
#    单 agent / 采样：
#      - cot                   Wei et al., NeurIPS 2022
#      - self_consistency      Wang et al., ICLR 2023
#    辩论：
#      - llm_debate            Du et al., ICML 2024
#      - mad                   Liang et al., EMNLP 2024
#    角色专业化：
#      - agentverse            Chen et al., ICLR 2024
#    图协作：
#      - macnet                Qian et al., 2024
#      - dylan (math / mmlu)   Liu et al., ICLR 2024  ← 按数据集路由
#      - h_swarm               2024-2025（PSO 拓扑，最直接竞品）
#      - evomac                Hu et al., 2024（进化拓扑）
#    验证：
#      - mav (math / mmlu)     UniMAS 团队, 2024  ← 按数据集路由
#
#  按数据集分类（与 run_ablation.sh 一致）：
#    MATH_DATASETS : GSM8K, GSM-Hard, AIME-2024, AQUA-RAT
#    MCQ_DATASETS  : MMLU-Pro
#  dylan / mav 会根据当前数据集自动选 *_math 或 *_mmlu 变体。
#
#  用法：
#    bash exp/run_v3_external_baselines.sh
#    bash exp/run_v3_external_baselines.sh --datasets "GSM8K MMLU-Pro"
#    bash exp/run_v3_external_baselines.sh --max_samples 100        # 快速验证
#    bash exp/run_v3_external_baselines.sh --resume                 # 跳过已有 eval
#    bash exp/run_v3_external_baselines.sh --skip "h_swarm evomac"  # 跳过指定方法
#
#  注意：
#    h_swarm 默认需要 pre-optimized adjacency matrix，否则可能 inference 报错；
#    evomac / agentverse / macnet / mad 是较重的方法，--full 跑可能很慢。
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

MATH_DATASETS="GSM8K GSM-Hard AIME-2024 AQUA-RAT"
MCQ_DATASETS="MMLU-Pro"

# 方法池（按数据集无关的方法 + 需按数据集路由的方法）
BASE_METHODS=(cot self_consistency llm_debate mad agentverse macnet evomac h_swarm)
# dylan 和 mav 需要按数据集分派（*_math / *_mmlu），在循环里处理

# -------------------- 解析命令行 --------------------
EFF_MODEL="$DEFAULT_MODEL"
EFF_DATASETS="$DEFAULT_DATASETS"
MAX_SAMPLES=""
RESUME=0
SKIP_LIST=""
OTHER_ARGS=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --model_name)   EFF_MODEL="${ARGS[$((i+1))]}";    ((i+=2)); continue ;;
        --datasets)     EFF_DATASETS="${ARGS[$((i+1))]}"; ((i+=2)); continue ;;
        --max_samples)  MAX_SAMPLES="${ARGS[$((i+1))]}";  ((i+=2)); continue ;;
        --skip)         SKIP_LIST="${ARGS[$((i+1))]}";    ((i+=2)); continue ;;
        --resume)       RESUME=1; ((i++)) || true ;;
        *)              OTHER_ARGS+=("${ARGS[$i]}"); ((i++)) || true ;;
    esac
done

MAX_SAMPLES_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARGS=(--max_samples "$MAX_SAMPLES")
fi

is_skipped() {
    local m="$1"
    for s in $SKIP_LIST; do
        [[ "$s" == "$m" ]] && return 0
    done
    return 1
}

# -------------------- 输出目录 --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$MAX_SAMPLES" ]]; then
    TAG="quick_${MAX_SAMPLES}"
else
    TAG="full"
fi
BASE_OUTPUT_DIR="./results_ablation/v3_external_${TAG}_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/run.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "======================================================================"
echo "  soo_centered_v3 — External Baselines Full Run"
echo "  Model:        ${EFF_MODEL}"
echo "  Datasets:     ${EFF_DATASETS}"
echo "  MaxSamples:   ${MAX_SAMPLES:-all}"
echo "  Base methods: ${BASE_METHODS[*]}"
echo "  Routed:       dylan_math|mmlu, mav_math|mmlu (per dataset)"
echo "  Skipped:      ${SKIP_LIST:-none}"
echo "  Output:       ${BASE_OUTPUT_DIR}"
echo "  Time:         $(date)"
echo "======================================================================"

# -------------------- 逐数据集 × 方法 --------------------
read -ra DATASET_ARRAY <<< "$EFF_DATASETS"

for dataset in "${DATASET_ARRAY[@]}"; do
    DATASET_DIR="${BASE_OUTPUT_DIR}/${dataset}"
    mkdir -p "$DATASET_DIR"

    # 按数据集类型扩展方法列表
    EXTRA_METHODS=()
    if echo " $MATH_DATASETS " | grep -qi " ${dataset} "; then
        EXTRA_METHODS=(dylan_math mav_math)
    elif echo " $MCQ_DATASETS " | grep -qi " ${dataset} "; then
        EXTRA_METHODS=(dylan_mmlu mav_mmlu)
    else
        # 默认当 math 数据集处理
        EXTRA_METHODS=(dylan_math mav_math)
    fi
    METHODS=("${BASE_METHODS[@]}" "${EXTRA_METHODS[@]}")

    echo ""
    echo "########## Dataset: ${dataset}  (extra: ${EXTRA_METHODS[*]}) ##########"

    # 阶段 1：推理
    for method in "${METHODS[@]}"; do
        if is_skipped "$method"; then
            echo "[SKIP-USER] ${method} on ${dataset}"
            continue
        fi
        OUTPUT_FILE="${DATASET_DIR}/${method}_infer.jsonl"
        EVAL_FILE="${DATASET_DIR}/${method}_xverify_eval.jsonl"

        if [[ $RESUME -eq 1 && -s "$EVAL_FILE" ]]; then
            echo "[RESUME-SKIP] ${method} on ${dataset} (eval file exists)"
            continue
        fi

        echo "[INFER] ${method} on ${dataset} ..."
        python inference.py \
            --method_name "${method}" \
            --test_dataset_name "${dataset}" \
            --model_name "${EFF_MODEL}" \
            --output_path "${OUTPUT_FILE}" \
            "${MAX_SAMPLES_ARGS[@]}" \
            "${OTHER_ARGS[@]}" || {
            echo "[WARNING] Inference ${method}/${dataset} failed, continuing..."
        }
    done

    # 阶段 2：评估
    for method in "${METHODS[@]}"; do
        if is_skipped "$method"; then
            continue
        fi
        INFER_FILE="${DATASET_DIR}/${method}_infer.jsonl"
        EVAL_FILE="${DATASET_DIR}/${method}_xverify_eval.jsonl"

        if [[ ! -s "$INFER_FILE" ]]; then
            echo "[NO-INFER] ${method}/${dataset} (skip eval)"
            continue
        fi
        if [[ $RESUME -eq 1 && -s "$EVAL_FILE" ]]; then
            continue
        fi

        echo "[EVAL]  ${method} on ${dataset} ..."
        python evaluate.py \
            --tested_method_name "${method}" \
            --tested_dataset_name "${dataset}" \
            --tested_mas_model_name "${EFF_MODEL}" \
            --tested_infer_path "${INFER_FILE}" \
            "${OTHER_ARGS[@]}" || {
            echo "[WARNING] Evaluation ${method}/${dataset} failed, continuing..."
        }
    done
done

# -------------------- 汇总 --------------------
echo ""
echo "======================================================================"
echo "  Aggregating external baseline results"
echo "======================================================================"
# 复用 exp/summarize.py（扫每个 dataset 子目录的 *_xverify_eval.jsonl）
for dataset in "${DATASET_ARRAY[@]}"; do
    DATASET_DIR="${BASE_OUTPUT_DIR}/${dataset}"
    echo ""
    echo "---- ${dataset} ----"
    python exp/summarize.py --results_dir "${DATASET_DIR}" || true
done

echo ""
echo "======================================================================"
echo "  Done at $(date)"
echo "  Results: ${BASE_OUTPUT_DIR}"
echo ""
echo "  Next step: combine with soo_centered_v3 / selforg baseline results"
echo "  and produce the final comparison table for the paper."
echo "======================================================================"
