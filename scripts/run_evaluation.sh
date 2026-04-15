#!/usr/bin/env bash
# 对 vanilla、cot、autogen、macnet、dylan_math、selforg、soo、soo_pu、soo_pu_pro、soo_pu_final 的推理结果进行评估
# 用法:
#   # 默认从 ./results/{tested_dataset_name}/{tested_mas_model_name}/{method}_infer.jsonl 读取
#   bash scripts/run_evaluation.sh
#   bash scripts/run_evaluation.sh --tested_dataset_name GSM-Hard --tested_mas_model_name qwen25-3b-instruct
#   bash scripts/run_evaluation.sh --tested_dataset_name gsm8k --sequential --overwrite
#   # 指定推理结果所在文件夹 (该文件夹下需包含 {method}_infer.jsonl)
#   bash scripts/run_evaluation.sh --infer-dir path/to/dir --tested_dataset_name GSM-Hard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# METHODS=(vanilla cot autogen macnet dylan_math selforg soo soo_pu soo_pu_pro soo_pu_final)
METHODS=(soo_pu_final_math)

# 解析自定义参数：--infer-dir <DIR>，其余参数原样传给 evaluate.py
INFER_DIR=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --infer-dir|--infer_dir)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --infer-dir 需要一个参数 (目录路径)." >&2
                exit 1
            fi
            INFER_DIR="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# 可选的通用参数，直接传给 evaluate.py（如 --tested_dataset_name、--tested_mas_model_name、--eval_protocol、--sequential、--overwrite 等）
EXTRA_ARGS=("${PASSTHROUGH_ARGS[@]}")

for method in "${METHODS[@]}"; do
    echo "=========================================="
    echo ">> Evaluating method: ${method}"
    echo "=========================================="

    CMD=(python evaluate.py --tested_method_name "${method}" "${EXTRA_ARGS[@]}")

    # 如果指定了推理结果文件夹，则自动拼接 {method}_infer.jsonl 作为 --tested_infer_path
    if [[ -n "$INFER_DIR" ]]; then
        INFER_FILE="${INFER_DIR%/}/${method}_infer.jsonl"
        CMD+=(--tested_infer_path "$INFER_FILE")
    fi

    "${CMD[@]}" || {
        echo "[WARNING] Evaluation for ${method} exited with non-zero status, continuing..."
    }
done

echo "=========================================="
echo ">> All evaluations finished."
echo "=========================================="
