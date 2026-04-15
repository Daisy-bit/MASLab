#!/usr/bin/env bash
# 完整流水线：先对 8 个方法依次做推理，再对推理结果依次做评估
# 方法列表: vanilla、cot、autogen、macnet、dylan_math、selforg、soo、soo_math、soo_pu、soo_pu_pro、soo_pu_final
#
# 用法:
#   bash scripts/run_full_pipeline.sh
#   bash scripts/run_full_pipeline.sh --test_dataset_name gsm8k --model_name other-model
#   bash scripts/run_full_pipeline.sh --sequential  # 推理与评估均串行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# 避免 torch+cu12 与系统/CUDA lib 中旧版 libnvJitLink 冲突（见 scripts/prepend_nvjitlink_ld_path.sh）
# shellcheck source=prepend_nvjitlink_ld_path.sh
source "${SCRIPT_DIR}/prepend_nvjitlink_ld_path.sh"

# 抑制 huggingface tokenizers 在 fork 多进程后的重复警告，并避免潜在死锁
export TOKENIZERS_PARALLELISM=false

# METHODS=(vanilla cot autogen macnet dylan_math selforg soo soo_pu)
# METHODS=(selforg soo)
METHODS=(soo_math)

# 默认数据集与模型（可通过命令行参数覆盖）
EFF_DATASET="GSM-Hard"
EFF_MODEL="qwen25-3b-instruct"

# 解析用户参数：提取 --test_dataset_name / --model_name，其余参数保留
OTHER_ARGS=()
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    if [[ "${ARGS[$i]}" == "--test_dataset_name" && $((i+1)) -lt ${#ARGS[@]} ]]; then
        EFF_DATASET="${ARGS[$i+1]}"
        ((i+=2))
        continue
    elif [[ "${ARGS[$i]}" == "--model_name" && $((i+1)) -lt ${#ARGS[@]} ]]; then
        EFF_MODEL="${ARGS[$i+1]}"
        ((i+=2))
        continue
    fi
    OTHER_ARGS+=("${ARGS[$i]}")
    ((i++)) || true
done

# 本轮流水线共用同一时间戳与输出目录，所有算法结果保存在同一文件夹下
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/${EFF_DATASET}/${EFF_MODEL}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
exec > >(tee "$LOG_FILE") 2>&1
echo ">> Output directory for this run: ${OUTPUT_DIR}"
echo ">> Log file: ${LOG_FILE}"

# 推理与评估使用的参数（通过 output_path / tested_infer_path 指定到同一目录）
INFERENCE_ARGS=(--test_dataset_name "$EFF_DATASET" --model_name "$EFF_MODEL" "${OTHER_ARGS[@]}")
EVAL_ARGS=(--tested_dataset_name "$EFF_DATASET" --tested_mas_model_name "$EFF_MODEL" "${OTHER_ARGS[@]}")

echo "========== 阶段 1/2: 推理 =========="
for method in "${METHODS[@]}"; do
    echo "------------------------------------------"
    echo ">> Running inference: ${method}"
    echo "------------------------------------------"
    python inference.py --method_name "${method}" --output_path "${OUTPUT_DIR}/${method}_infer.jsonl" "${INFERENCE_ARGS[@]}" || {
        echo "[WARNING] Inference ${method} exited with non-zero status, continuing..."
    }
done
echo ">> Inference phase finished."
echo ""

echo "========== 阶段 2/2: 评估 =========="
for method in "${METHODS[@]}"; do
    echo "------------------------------------------"
    echo ">> Evaluating: ${method}"
    echo "------------------------------------------"
    python evaluate.py --tested_method_name "${method}" --tested_infer_path "${OUTPUT_DIR}/${method}_infer.jsonl" "${EVAL_ARGS[@]}" || {
        echo "[WARNING] Evaluation for ${method} exited with non-zero status, continuing..."
    }
done
echo ">> Evaluation phase finished."
echo ""
echo "=========================================="
printf '%s\n' '>> Full pipeline (inference + evaluation) finished.'
echo "=========================================="
