#!/usr/bin/env bash
# ======================================================================
#  SelfOrg / SOO 消融实验 — 一键运行
#
#  消融设计:
#    1. selforg              — 完整 SelfOrg（DAG 辩论 + Shapley 贡献度）
#    2. selforg_no_debate    — 仅首轮采样 + 质心选择（无辩论）
#    3. selforg_random_graph — 随机通信图 + 辩论（无结构化编排）
#    4. soo                  — Perron-Frobenius 可靠性替代 Shapley 贡献度
#    5. soo_centered         — Double-Centered Spectral（PC1 of HSH 替代 Perron）
#    6. soo_centered_v2      — HSH + 方差共识早停（tr(S_c) < threshold）
#
#  用法:
#    bash exp/run_ablation.sh
#    bash exp/run_ablation.sh --model_name qwen25-7b-instruct
#    bash exp/run_ablation.sh --test_dataset_name GSM8K --model_name qwen25-3b-instruct
#    bash exp/run_ablation.sh --datasets "GSM8K GSM-Hard"
#    bash exp/run_ablation.sh --max_samples 30           # 每数据集最多30条（快速测试）
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# 如果 prepend_nvjitlink_ld_path.sh 存在则 source（GPU 环境兼容）
if [[ -f scripts/prepend_nvjitlink_ld_path.sh ]]; then
    source scripts/prepend_nvjitlink_ld_path.sh
fi
export TOKENIZERS_PARALLELISM=false

# -------------------- 默认参数 --------------------
DEFAULT_MODEL="qwen25-3b-instruct"
DEFAULT_DATASETS="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
# 核心对比方法（所有数据集都跑）
CORE_METHODS=(selforg selforg_no_debate selforg_random_graph soo soo_centered soo_centered_v2)
# 数学类数据集追加 dylan_math，选择题类数据集追加 dylan_mmlu
MATH_DATASETS="GSM8K GSM-Hard AIME-2024 MATH"
CHOICE_DATASETS="MMLU-Pro MMLU AQUA-RAT MedMCQA"

# -------------------- 解析命令行参数 --------------------
EFF_MODEL="$DEFAULT_MODEL"
EFF_DATASETS="$DEFAULT_DATASETS"
MAX_SAMPLES=""
OTHER_ARGS=()
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --model_name)
            EFF_MODEL="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --test_dataset_name)
            EFF_DATASETS="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --datasets)
            EFF_DATASETS="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        --max_samples)
            MAX_SAMPLES="${ARGS[$((i+1))]}"
            ((i+=2)); continue ;;
        *)
            OTHER_ARGS+=("${ARGS[$i]}")
            ((i++)) || true ;;
    esac
done

# 构建 max_samples 参数
MAX_SAMPLES_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARGS=(--max_samples "$MAX_SAMPLES")
fi

# -------------------- 输出目录 --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./results_ablation/${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/ablation.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "======================================================================"
echo "  SelfOrg Ablation Experiments"
echo "  Model:      ${EFF_MODEL}"
echo "  Datasets:   ${EFF_DATASETS}"
echo "  Methods:    ${CORE_METHODS[*]} + dylan_math/dylan_mmlu (auto)"
echo "  MaxSamples: ${MAX_SAMPLES:-all}"
echo "  Output:     ${BASE_OUTPUT_DIR}"
echo "  Time:       $(date)"
echo "======================================================================"
echo ""

# -------------------- 逐数据集 × 逐方法 运行 --------------------
read -ra DATASET_ARRAY <<< "$EFF_DATASETS"

for dataset in "${DATASET_ARRAY[@]}"; do
    DATASET_DIR="${BASE_OUTPUT_DIR}/${dataset}"
    mkdir -p "$DATASET_DIR"

    # 根据数据集类型选择 dylan 变体
    DYLAN_METHOD=""
    if echo " $MATH_DATASETS " | grep -qi " ${dataset} "; then
        DYLAN_METHOD="dylan_math"
    elif echo " $CHOICE_DATASETS " | grep -qi " ${dataset} "; then
        DYLAN_METHOD="dylan_mmlu"
    else
        DYLAN_METHOD="dylan_math"   # 默认用 dylan_math
    fi
    METHODS=("${CORE_METHODS[@]}" "$DYLAN_METHOD")

    echo "########## Dataset: ${dataset} (dylan: ${DYLAN_METHOD}) ##########"
    echo ""

    # ===== 阶段 1: 推理 =====
    for method in "${METHODS[@]}"; do
        OUTPUT_FILE="${DATASET_DIR}/${method}_infer.jsonl"
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
    echo ""

    # ===== 阶段 2: 评估 =====
    for method in "${METHODS[@]}"; do
        INFER_FILE="${DATASET_DIR}/${method}_infer.jsonl"
        if [[ ! -f "$INFER_FILE" ]]; then
            echo "[SKIP] No inference file for ${method}/${dataset}"
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
    echo ""

    # ===== 阶段 3: 该数据集的小结 =====
    echo "[SUMMARY] Dataset: ${dataset}"
    python exp/summarize.py --results_dir "${DATASET_DIR}" \
        --output_csv "${DATASET_DIR}/ablation_results.csv" || true
    echo ""
done

# -------------------- 全局汇总 --------------------
echo "======================================================================"
echo "  All Datasets Summary"
echo "======================================================================"

# 合并所有数据集的评估结果到一个汇总表
python -c "
import json, os, sys

base = '${BASE_OUTPUT_DIR}'
datasets = '${EFF_DATASETS}'.split()

# 自动发现所有方法（从评估文件名中提取）
method_set = set()
for ds in datasets:
    ds_dir = os.path.join(base, ds)
    if not os.path.isdir(ds_dir):
        continue
    for f in os.listdir(ds_dir):
        if f.endswith('_xverify_eval.jsonl'):
            method_set.add(f.replace('_xverify_eval.jsonl', ''))
# 排序：核心方法优先，dylan 放最后
core_order = ['selforg', 'selforg_no_debate', 'selforg_random_graph', 'soo', 'soo_centered', 'soo_centered_v2']
methods = [m for m in core_order if m in method_set] + sorted(method_set - set(core_order))

header = f\"{'Method':<30}\" + ''.join(f'{d:>14}' for d in datasets) + f\"{'Average':>14}\"
sep = '-' * len(header)
print(sep)
print(header)
print(sep)

for method in methods:
    accs = []
    cells = []
    for ds in datasets:
        path = os.path.join(base, ds, f'{method}_xverify_eval.jsonl')
        if not os.path.exists(path):
            cells.append(f\"{'N/A':>14}\")
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: records.append(json.loads(line))
                    except: pass
        valid = [r for r in records if r.get('eval_score') is not None]
        correct = sum(1 for r in valid if r['eval_score'] == 1)
        total = len(valid)
        if total > 0:
            acc = correct / total * 100
            accs.append(acc)
            cells.append(f'{acc:>13.2f}%')
        else:
            cells.append(f\"{'N/A':>14}\")
    avg = sum(accs) / len(accs) if accs else 0
    avg_str = f'{avg:>13.2f}%' if accs else f\"{'N/A':>14}\"
    print(f'{method:<30}' + ''.join(cells) + avg_str)

print(sep)
" || true

echo ""
echo "======================================================================"
echo "  Ablation experiments finished at $(date)"
echo "  Results saved to: ${BASE_OUTPUT_DIR}"
echo "======================================================================"
