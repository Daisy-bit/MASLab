#!/usr/bin/env bash
# ======================================================================
#  soo_centered_v3: --full 扫描谱共识阈值 variance_consensus_thr
#
#  配置：A on + Sim 已硬编码关闭（v3 _check_for_consensus 只保留 spectral）
#  扫描：tau ∈ {0, 0.05, 0.15, 0.35, 0.75, 1.50, 2.0}
#
#  翻译表（N=5, tr(S_c)=N(1-avg_cos^2)：
#    tau=0.00  → 需要 avg_cos=1.00 → 永不触发    (= 100 baseline, S off)
#    tau=0.05  → 需要 avg_cos≥0.99 → v2 默认
#    tau=0.15  → 需要 avg_cos≥0.97
#    tau=0.35  → 需要 avg_cos≥0.93
#    tau=0.75  → 需要 avg_cos≥0.85
#    tau=1.50  → 需要 avg_cos≥0.70
#    tau=2.00  → 需要 avg_cos≥0.60 → 容易触发
#
#  因此 tau=0 的 trial 自然给出 100 vs 110 对比所需的 baseline。
#
#  用法:
#    bash exp/run_v3_tau_sweep.sh
#    bash exp/run_v3_tau_sweep.sh --datasets "GSM8K MMLU-Pro"
#    bash exp/run_v3_tau_sweep.sh --resume
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
BASE_OUTPUT_DIR="./results_ablation/v3_tau_sweep_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/run.log"
exec > >(tee "$LOG_FILE") 2>&1

# -------------------- 生成 7 个 tau trial yaml --------------------
TRIAL_PREFIX="tau_${TIMESTAMP}"
TRIALS_JSON="${BASE_OUTPUT_DIR}/trials.json"

python - "$BASE_YAML" "$CONFIG_DIR" "$TRIAL_PREFIX" "$TRIALS_JSON" <<'PYEOF'
import json, os, sys, yaml

base_yaml, config_dir, prefix, out_json = sys.argv[1:5]
with open(base_yaml, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

TAUS = [0.0, 0.05, 0.15, 0.35, 0.75, 1.50, 2.00]

trials = []
for tau in TAUS:
    slug = f"{tau:.2f}".replace(".", "p")
    trials.append({
        "name": f"{prefix}_{slug}",
        "overrides": {
            "enable_answer_consensus": True,
            "enable_spectral_consensus": True,
            "variance_consensus_thr": float(tau),
        },
    })

os.makedirs(config_dir, exist_ok=True)
for t in trials:
    cfg = dict(base); cfg.update(t["overrides"])
    with open(os.path.join(config_dir, f"{t['name']}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(trials, f, indent=2, ensure_ascii=False)

print(f"[GEN] {len(trials)} tau-sweep configs written to {config_dir}/")
for t in trials:
    tau = t["overrides"]["variance_consensus_thr"]
    avg_cos = 1 - tau/5.0
    print(f"   {t['name']}   tau={tau:.2f}   (requires avg pairwise cos >= {avg_cos:.2f})")
PYEOF

read -ra DATASET_ARRAY <<< "$EFF_DATASETS"
TRIAL_NAMES=( $(python -c "import json; print(' '.join(t['name'] for t in json.load(open('$TRIALS_JSON'))))") )

echo ""
echo "======================================================================"
echo "  soo_centered_v3 tau Sweep  (enable_answer_consensus=true, similarity hardcoded off)"
echo "  Model:     ${EFF_MODEL}"
echo "  Datasets:  ${EFF_DATASETS}"
echo "  #Trials:   ${#TRIAL_NAMES[@]}  (tau=0 reproduces the 'no spectral' baseline)"
echo "  Output:    ${BASE_OUTPUT_DIR}"
echo "  Time:      $(date)"
echo "======================================================================"

# -------------------- 循环 trial × dataset --------------------
TOTAL=$((${#TRIAL_NAMES[@]} * ${#DATASET_ARRAY[@]}))
IDX=0
for TRIAL_NAME in "${TRIAL_NAMES[@]}"; do
    TRIAL_DIR="${BASE_OUTPUT_DIR}/${TRIAL_NAME}"
    mkdir -p "$TRIAL_DIR"
    cp "${CONFIG_DIR}/${TRIAL_NAME}.yaml" "${TRIAL_DIR}/config.yaml"

    TAU=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG_DIR}/${TRIAL_NAME}.yaml'))['variance_consensus_thr'])")
    echo ""
    echo "#################### ${TRIAL_NAME}  (tau=${TAU}) ####################"

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
echo "  Aggregating tau-sweep results"
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
echo "  Interpretation guide:"
echo "    tau=0.00 row = 100 baseline (spectral never fires)"
echo "    other rows  = 110 with varying spectral trigger sensitivity"
echo "  Look for the accuracy-vs-tau curve: accuracy should rise as tau"
echo "  moves from 0 toward the optimum, then drop when tau becomes so"
echo "  loose that spectral fires on unreliable consensus cases."
echo "======================================================================"
