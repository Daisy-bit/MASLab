#!/usr/bin/env bash
# ======================================================================
#  soo_centered_v3: --full 对比 000 vs 010
#
#  旧 AAS (answer, spectral, similarity) 三位命名：
#    000 = 所有共识检查关闭（Sim 已硬编码 off, A off, S off）
#    010 = 仅谱共识开启 (A off, S on @ tau=0.35, Sim off)
#           tau=0.35 对应"平均余弦 >= 0.93"，中等触发难度
#
#  目的：
#    隔离"谱共识单独能提供多少增益"。
#    tau 扫描里 A=1 一直开，看到的是"谱共识与符号多数票的叠加效应"；
#    这里把 A 关掉，直接测 spectral-alone vs no-early-stop。
#
#  控制变量:
#    diversity_p  = 0.0   （关掉 DAG 构图随机扰动）
#    temperature  = 0.7   （锁定 LLM 采样温度）
#    变化的只有 enable_spectral_consensus ∈ {false, true}
#
#  用法:
#    bash exp/run_v3_000_vs_010.sh
#    bash exp/run_v3_000_vs_010.sh --datasets "GSM8K MMLU-Pro"
#    bash exp/run_v3_000_vs_010.sh --resume
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f scripts/prepend_nvjitlink_ld_path.sh ]]; then
    source scripts/prepend_nvjitlink_ld_path.sh
fi
export TOKENIZERS_PARALLELISM=false

METHOD="soo_centered_v3"
CONFIG_DIR="methods/${METHOD}/configs"
BASE_YAML="${CONFIG_DIR}/config_main.yaml"

DEFAULT_MODEL="qwen25-3b-instruct"
DEFAULT_DATASETS="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"

EFF_MODEL="$DEFAULT_MODEL"
EFF_DATASETS="$DEFAULT_DATASETS"
RESUME=0
OTHER_ARGS=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --model_name) EFF_MODEL="${ARGS[$((i+1))]}";    ((i+=2)); continue ;;
        --datasets)   EFF_DATASETS="${ARGS[$((i+1))]}"; ((i+=2)); continue ;;
        --resume)     RESUME=1; ((i++)) || true ;;
        *)            OTHER_ARGS+=("${ARGS[$i]}"); ((i++)) || true ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./results_ablation/v3_000_vs_010_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/run.log"
exec > >(tee "$LOG_FILE") 2>&1

# -------------------- 生成 2 个 trial yaml --------------------
TRIAL_PREFIX="AS_${TIMESTAMP}"
TRIALS_JSON="${BASE_OUTPUT_DIR}/trials.json"

python - "$BASE_YAML" "$CONFIG_DIR" "$TRIAL_PREFIX" "$TRIALS_JSON" <<'PYEOF'
import json, os, sys, yaml

base_yaml, config_dir, prefix, out_json = sys.argv[1:5]
with open(base_yaml, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

# Similarity consensus is already hardcoded off in v3 source.
trials = [
    {
        "name": f"{prefix}_000",  # all-off baseline
        "overrides": {
            "enable_answer_consensus": False,
            "enable_spectral_consensus": False,
            "diversity_p": 0.0,
            "temperature": 0.7,
        },
    },
    {
        "name": f"{prefix}_010",  # spectral only, tau=0.35 (avg_cos >= 0.93)
        "overrides": {
            "enable_answer_consensus": False,
            "enable_spectral_consensus": True,
            "variance_consensus_thr": 0.35,
            "diversity_p": 0.0,
            "temperature": 0.7,
        },
    },
]

os.makedirs(config_dir, exist_ok=True)
for t in trials:
    cfg = dict(base); cfg.update(t["overrides"])
    with open(os.path.join(config_dir, f"{t['name']}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(trials, f, indent=2, ensure_ascii=False)

print(f"[GEN] {len(trials)} trial configs written to {config_dir}/")
for t in trials:
    ov = t["overrides"]
    print(f"   {t['name']}   A={int(ov['enable_answer_consensus'])}  S={int(ov['enable_spectral_consensus'])}  "
          f"diversity_p={ov['diversity_p']}  temp={ov['temperature']}")
PYEOF

read -ra DATASET_ARRAY <<< "$EFF_DATASETS"
TRIAL_NAMES=( $(python -c "import json; print(' '.join(t['name'] for t in json.load(open('$TRIALS_JSON'))))") )

echo ""
echo "======================================================================"
echo "  soo_centered_v3 — 000 vs 010 (full samples)"
echo "  Model:       ${EFF_MODEL}"
echo "  Datasets:    ${EFF_DATASETS}"
echo "  Controlled:  diversity_p=0, temperature=0.7, similarity off (hardcoded)"
echo "  Output:      ${BASE_OUTPUT_DIR}"
echo "  Time:        $(date)"
echo "======================================================================"

# -------------------- 循环 trial × dataset --------------------
TOTAL=$((${#TRIAL_NAMES[@]} * ${#DATASET_ARRAY[@]}))
IDX=0
for TRIAL_NAME in "${TRIAL_NAMES[@]}"; do
    TRIAL_DIR="${BASE_OUTPUT_DIR}/${TRIAL_NAME}"
    mkdir -p "$TRIAL_DIR"
    cp "${CONFIG_DIR}/${TRIAL_NAME}.yaml" "${TRIAL_DIR}/config.yaml"

    TAG=$(echo "$TRIAL_NAME" | grep -oE '[01]{3}$')
    echo ""
    echo "#################### ${TRIAL_NAME}  (AAS=${TAG}) ####################"

    for dataset in "${DATASET_ARRAY[@]}"; do
        IDX=$((IDX+1))
        DATASET_DIR="${TRIAL_DIR}/${dataset}"
        mkdir -p "$DATASET_DIR"
        OUTPUT_FILE="${DATASET_DIR}/${METHOD}_infer.jsonl"
        EVAL_FILE="${DATASET_DIR}/${METHOD}_xverify_eval.jsonl"

        if [[ $RESUME -eq 1 && -s "$EVAL_FILE" ]]; then
            echo "[SKIP ${IDX}/${TOTAL}] ${dataset} (resume, eval present)"
            continue
        fi

        echo "---- [${IDX}/${TOTAL}] ${dataset} ----"
        python inference.py \
            --method_name "${METHOD}" \
            --method_config_name "${TRIAL_NAME}" \
            --test_dataset_name "${dataset}" \
            --model_name "${EFF_MODEL}" \
            --output_path "${OUTPUT_FILE}" \
            "${OTHER_ARGS[@]}" || {
            echo "[WARN] inference failed: ${TRIAL_NAME}/${dataset}"
            continue
        }

        if [[ -f "$OUTPUT_FILE" ]]; then
            python evaluate.py \
                --tested_method_name "${METHOD}" \
                --tested_dataset_name "${dataset}" \
                --tested_mas_model_name "${EFF_MODEL}" \
                --tested_infer_path "${OUTPUT_FILE}" \
                "${OTHER_ARGS[@]}" || {
                echo "[WARN] evaluate failed: ${TRIAL_NAME}/${dataset}"
            }
        fi
    done
done

# -------------------- 汇总 --------------------
echo ""
echo "======================================================================"
echo "  Aggregating 000 vs 010 results"
echo "======================================================================"
python exp/summarize_gridsearch.py \
    --results_dir "${BASE_OUTPUT_DIR}" \
    --datasets "${EFF_DATASETS}" \
    --top_k 0 \
    --output_csv "${BASE_OUTPUT_DIR}/summary.csv" || true

echo ""
echo "======================================================================"
echo "  Done at $(date)"
echo "  Results:  ${BASE_OUTPUT_DIR}"
echo "  Summary:  ${BASE_OUTPUT_DIR}/summary.csv"
echo ""
echo "  Interpretation:"
echo "    000 = no early stop, pure debate until max_rounds"
echo "    010 = spectral-only early stop at tau=0.35 (avg_cos >= 0.93)"
echo "  Delta (010 - 000) measures the spectral-alone contribution,"
echo "  independent of the answer-plurality check (which is off in both)."
echo "======================================================================"
