#!/usr/bin/env bash
# ======================================================================
#  soo_centered_v3 超参数网格搜索
#
#  在 methods/soo_centered_v3/configs/ 下动态生成 trial yaml，
#  对每个 trial 跑指定数据集上的 inference + evaluate，最后调
#  exp/summarize_gridsearch.py 汇总并给出均值最优组合。
#
#  用法：
#    bash exp/run_v3_gridsearch.sh --quick
#        30 条样本、默认坐标下降搜索 (≈16 trials)
#
#    bash exp/run_v3_gridsearch.sh --full
#        全量样本、默认坐标下降搜索
#
#    bash exp/run_v3_gridsearch.sh --quick --sweep_mode product
#        笛卡尔积搜索 (answer_consensus × diversity_p × max_rounds = 36 trials)
#
#    bash exp/run_v3_gridsearch.sh --quick --trials_json my_trials.json
#        自定义 trial 列表，JSON 格式见 --help
#
#  其它参数:
#    --max_samples N        覆盖每个数据集的样本数
#    --datasets "D1 D2"     覆盖数据集列表
#    --model_name NAME      推理用后端模型 (默认 qwen25-3b-instruct)
#    --resume               跳过已有完整 eval 文件的 trial/dataset 组合
#    --keep_configs         结束时保留 gs_*.yaml（默认保留，便于复现）
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f scripts/prepend_nvjitlink_ld_path.sh ]]; then
    source scripts/prepend_nvjitlink_ld_path.sh
fi
export TOKENIZERS_PARALLELISM=false

# -------------------- 默认 --------------------
METHOD="soo_centered_v3"
CONFIG_DIR="methods/${METHOD}/configs"
BASE_YAML="${CONFIG_DIR}/config_main.yaml"

DEFAULT_MODEL="qwen25-3b-instruct"
DEFAULT_DATASETS="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"

EFF_MODEL="$DEFAULT_MODEL"
EFF_DATASETS="$DEFAULT_DATASETS"
MAX_SAMPLES=""
MODE=""
SWEEP_MODE="coord"      # coord | product
TRIALS_JSON=""
RESUME=0
KEEP_CONFIGS=1
TAG=""
OTHER_ARGS=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --quick)           MODE="quick"; ((i++)) || true ;;
        --full)            MODE="full";  ((i++)) || true ;;
        --model_name)      EFF_MODEL="${ARGS[$((i+1))]}";      ((i+=2)); continue ;;
        --datasets)        EFF_DATASETS="${ARGS[$((i+1))]}";   ((i+=2)); continue ;;
        --max_samples)     MAX_SAMPLES="${ARGS[$((i+1))]}";    ((i+=2)); continue ;;
        --sweep_mode)      SWEEP_MODE="${ARGS[$((i+1))]}";     ((i+=2)); continue ;;
        --trials_json)     TRIALS_JSON="${ARGS[$((i+1))]}";    ((i+=2)); continue ;;
        --tag)             TAG="${ARGS[$((i+1))]}";            ((i+=2)); continue ;;
        --resume)          RESUME=1; ((i++)) || true ;;
        --no_keep_configs) KEEP_CONFIGS=0; ((i++)) || true ;;
        --keep_configs)    KEEP_CONFIGS=1; ((i++)) || true ;;
        *) OTHER_ARGS+=("${ARGS[$i]}"); ((i++)) || true ;;
    esac
done

case "$MODE" in
    quick) : "${MAX_SAMPLES:=30}"; : "${TAG:=quick}" ;;
    full)  : "${TAG:=full}" ;;
    *)
        if [[ -z "$MAX_SAMPLES" && -z "$TRIALS_JSON" ]]; then
            echo "[ERROR] --quick / --full not set; must pass --max_samples or --trials_json"
            exit 1
        fi
        : "${TAG:=custom}"
        ;;
esac

MAX_SAMPLES_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARGS=(--max_samples "$MAX_SAMPLES")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./results_ablation/v3_gridsearch_${TAG}_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/gridsearch.log"
exec > >(tee "$LOG_FILE") 2>&1

TRIAL_PREFIX="gs_${TIMESTAMP}"
TRIALS_OUT_JSON="${BASE_OUTPUT_DIR}/trials.json"

# -------------------- 生成 trial yaml 和 trials.json --------------------
python - "$BASE_YAML" "$CONFIG_DIR" "$TRIAL_PREFIX" "$SWEEP_MODE" "${TRIALS_JSON:-}" "$TRIALS_OUT_JSON" <<'PYEOF'
import json
import os
import sys
import yaml
from itertools import product

base_yaml, config_dir, prefix, sweep_mode, user_json, out_json = sys.argv[1:7]

with open(base_yaml, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

# ---- 搜索空间 ------------------------------------------------------
# 坐标下降: baseline + 对每个关键超参单独扫描非默认值
COORD_SWEEPS = [
    ("answer_consensus_min_initial", [2, 4]),
    ("answer_consensus_min_round",   [2, 4]),
    ("diversity_p",                  [0.0, 0.1, 0.3]),
    ("max_rounds",                   [1, 3]),
    ("top_k",                        [1, 3]),
    ("consensus_min_sim",            [0.90, 0.99]),
    ("sim_threshold",                [0.70, 0.85]),
    ("variance_consensus_thr",       [0.03, 0.10]),
    ("temperature",                  [0.3, 0.7]),
    ("math_mode",                    ["simple"]),
    ("include_math_few_shot",        [False]),
    ("include_mcq_format_hint",      [False]),
    ("reform",                       [False]),
    ("aggregate_mode",               ["single"]),
    ("enable_answer_consensus",      [False]),
    ("enable_spectral_consensus",    [False]),
]

# 笛卡尔积: 围绕 v3 核心参数 + 采样温度 + 谱共识阈值
PRODUCT_SWEEP = {
    "answer_consensus_min_initial": [2, 3, 4],
    "diversity_p":                  [0.0, 0.1, 0.2, 0.3],
    "variance_consensus_thr":       [0.03, 0.05, 0.10],
    "temperature":                  [0.3, 0.5, 0.7],
}

trials = []
if user_json:
    with open(user_json, "r", encoding="utf-8") as f:
        trials = json.load(f)
    # 允许给定 [{overrides: {...}}] 或 [{name: "...", overrides: {...}}]
    for idx, t in enumerate(trials):
        t.setdefault("name", f"{prefix}_user{idx:03d}")
        t.setdefault("overrides", {})
elif sweep_mode == "product":
    keys = list(PRODUCT_SWEEP.keys())
    for idx, combo in enumerate(product(*(PRODUCT_SWEEP[k] for k in keys))):
        ov = dict(zip(keys, combo))
        trials.append({"name": f"{prefix}_p{idx:03d}", "overrides": ov})
else:  # coord
    trials.append({"name": f"{prefix}_base", "overrides": {}})
    idx = 0
    for key, vals in COORD_SWEEPS:
        for v in vals:
            trials.append({"name": f"{prefix}_c{idx:03d}", "overrides": {key: v}})
            idx += 1

os.makedirs(config_dir, exist_ok=True)
for t in trials:
    cfg = dict(base)
    cfg.update(t["overrides"])
    path = os.path.join(config_dir, f"{t['name']}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(trials, f, indent=2, ensure_ascii=False)

print(f"[GEN] {len(trials)} trial configs written to {config_dir}/")
PYEOF

read -ra DATASET_ARRAY <<< "$EFF_DATASETS"

NUM_TRIALS=$(python -c "import json; print(len(json.load(open('$TRIALS_OUT_JSON'))))")

echo "======================================================================"
echo "  soo_centered_v3 Grid Search"
echo "  Model:       ${EFF_MODEL}"
echo "  Datasets:    ${EFF_DATASETS}"
echo "  MaxSamples:  ${MAX_SAMPLES:-all}"
echo "  Sweep mode:  ${SWEEP_MODE}${TRIALS_JSON:+ (overridden by ${TRIALS_JSON})}"
echo "  Num trials:  ${NUM_TRIALS}"
echo "  Output:      ${BASE_OUTPUT_DIR}"
echo "  Time:        $(date)"
echo "======================================================================"

# -------------------- 逐 trial × 数据集运行 --------------------
TRIAL_NAMES=( $(python -c "import json; print(' '.join(t['name'] for t in json.load(open('$TRIALS_OUT_JSON'))))") )

TRIAL_IDX=0
for TRIAL_NAME in "${TRIAL_NAMES[@]}"; do
    TRIAL_IDX=$((TRIAL_IDX+1))
    TRIAL_DIR="${BASE_OUTPUT_DIR}/${TRIAL_NAME}"
    mkdir -p "$TRIAL_DIR"
    # 记录该 trial 的有效 overrides（相对于 base）到 trial dir
    cp "${CONFIG_DIR}/${TRIAL_NAME}.yaml" "${TRIAL_DIR}/config.yaml"

    OVERRIDES_DESC=$(python -c "
import json
with open('$TRIALS_OUT_JSON') as f: ts=json.load(f)
for t in ts:
    if t['name']=='${TRIAL_NAME}':
        print(' '.join(f'{k}={v}' for k,v in t['overrides'].items()) or '(baseline)')
        break
")
    echo ""
    echo "#################### [${TRIAL_IDX}/${NUM_TRIALS}] ${TRIAL_NAME} ####################"
    echo "Overrides: ${OVERRIDES_DESC}"

    for dataset in "${DATASET_ARRAY[@]}"; do
        DATASET_DIR="${TRIAL_DIR}/${dataset}"
        mkdir -p "$DATASET_DIR"
        OUTPUT_FILE="${DATASET_DIR}/${METHOD}_infer.jsonl"
        EVAL_FILE="${DATASET_DIR}/${METHOD}_xverify_eval.jsonl"

        if [[ $RESUME -eq 1 && -s "$EVAL_FILE" ]]; then
            echo "[SKIP] ${dataset} (resume, eval file present)"
            continue
        fi

        echo "---- ${dataset} ----"
        python inference.py \
            --method_name "${METHOD}" \
            --method_config_name "${TRIAL_NAME}" \
            --test_dataset_name "${dataset}" \
            --model_name "${EFF_MODEL}" \
            --output_path "${OUTPUT_FILE}" \
            "${MAX_SAMPLES_ARGS[@]}" \
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
echo "  Aggregating grid-search results"
echo "======================================================================"
python exp/summarize_gridsearch.py \
    --results_dir "${BASE_OUTPUT_DIR}" \
    --datasets "${EFF_DATASETS}" \
    --top_k 15 \
    --output_csv "${BASE_OUTPUT_DIR}/summary.csv" || true

# -------------------- 清理 trial yaml（可选）--------------------
if [[ $KEEP_CONFIGS -eq 0 ]]; then
    echo "Removing generated trial configs ${CONFIG_DIR}/${TRIAL_PREFIX}_*.yaml"
    rm -f "${CONFIG_DIR}/${TRIAL_PREFIX}_"*.yaml || true
fi

echo ""
echo "======================================================================"
echo "  Done at $(date)"
echo "  Results:  ${BASE_OUTPUT_DIR}"
echo "  Summary:  ${BASE_OUTPUT_DIR}/summary.csv"
echo "======================================================================"
