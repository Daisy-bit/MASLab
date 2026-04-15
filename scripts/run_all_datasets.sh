#!/usr/bin/env bash
# 按顺序对 datasets/data 下所有数据集执行完整流水线（推理 + 评估）
# 用法示例：
#   bash scripts/run_all_datasets.sh
#   bash scripts/run_all_datasets.sh --model_name qwen25-3b-instruct
#   bash scripts/run_all_datasets.sh --sequential --model_name other-model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="${PROJECT_ROOT}/datasets/data"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_DIR}"
    exit 1
fi

# 收集所有数据集名称（按文件名前缀去重，如 GSM-Hard.json / GSM-Hard.jsonl -> GSM-Hard）
declare -A SEEN_DATASETS
DATASETS=()

shopt -s nullglob
for path in "${DATA_DIR}"/*.json*; do
    [[ -f "$path" ]] || continue
    base_name="$(basename "$path")"
    dataset_name="${base_name%%.*}"   # 去掉扩展名

    if [[ -z "${SEEN_DATASETS[$dataset_name]:-}" ]]; then
        SEEN_DATASETS["$dataset_name"]=1
        DATASETS+=("$dataset_name")
    fi
done
shopt -u nullglob

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "[ERROR] 在 ${DATA_DIR} 下未找到任何 *.json / *.jsonl 数据集文件。"
    exit 1
fi

echo ">> 将依次对以下数据集运行完整流水线："
for ds in "${DATASETS[@]}"; do
    echo "   - ${ds}"
done
echo ""

# 用户额外参数（透传给 run_full_pipeline.sh）
USER_ARGS=("$@")

for ds in "${DATASETS[@]}"; do
    echo "=================================================="
    echo ">> 开始运行数据集：${ds}"
    echo "=================================================="
    bash "${SCRIPT_DIR}/run_full_pipeline.sh" \
        --test_dataset_name "${ds}" \
        "${USER_ARGS[@]}" || {
        echo "[WARNING] 数据集 ${ds} 的流水线执行返回非零退出码，继续处理下一个数据集..."
    }
    echo ""
done

echo "=================================================="
echo ">> 所有数据集的完整流水线运行结束。"
echo "=================================================="

