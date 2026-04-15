#!/usr/bin/env bash
# 依次运行 vanilla、cot、autogen、macnet、dylan_math、selforg、soo、soo_pu、soo_pu_pro、soo_pu_final
# 用法:
#   bash scripts/run_all_methods.sh
#   bash scripts/run_all_methods.sh --test_dataset_name gsm8k --model_name qwen25-3b-instruct
#   bash scripts/run_all_methods.sh --sequential  # 串行推理

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

METHODS=(vanilla cot autogen macnet dylan_math selforg soo soo_pu soo_pu_pro soo_pu_final)

# 可选的通用参数（如 --test_dataset_name、--model_name 等），直接传给 inference.py
EXTRA_ARGS=("$@")

for method in "${METHODS[@]}"; do
    echo "=========================================="
    echo ">> Running method: ${method}"
    echo "=========================================="
    python inference.py --method_name "${method}" "${EXTRA_ARGS[@]}" || {
        echo "[WARNING] Method ${method} exited with non-zero status, continuing..."
    }
done

echo "=========================================="
echo ">> All methods finished."
echo "=========================================="
