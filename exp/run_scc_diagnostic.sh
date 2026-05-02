#!/usr/bin/env bash
# ==============================================================================
# SCC (soo_centered_v3) diagnostic experiment — Experiment 1 (regime analysis).
#
# Mirrors exp/run_subjective_diagnostic.sh but runs SCC-full with the
# diagnostic-emission config (config_diag.yaml) so we get a `diagnostic` field
# matching mad_vote's schema. The same analyzer
# (scripts/diagnostic/analyze_diagnostic.py) then produces Tables 1-6 per
# (model × method) cell.
#
# Defaults: 3 models × 5 datasets × full samples (matches the existing
# results_diagnostic/run_20260501_* mad_vote runs for parity).
#
# Usage (from MASLab/ project root):
#   bash exp/run_scc_diagnostic.sh
#   bash exp/run_scc_diagnostic.sh --max_samples 50
#   DATASETS="GSM8K AIME-2024" MODELS="qwen25-3b-instruct" \
#     bash exp/run_scc_diagnostic.sh
#   ANALYZE_ONLY=1 ANALYZE_DIR=./results_diagnostic_scc/run_20260101_120000 \
#     bash exp/run_scc_diagnostic.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=false

DATASETS_DEFAULT="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
MODELS_DEFAULT="qwen25-1.5b-instruct qwen25-3b-instruct qwen25-7b-instruct"
METHOD_NAME="soo_centered_v3"
METHOD_CONFIG_NAME="config_diag"
OUTPUT_ROOT_DEFAULT="./results_diagnostic_scc"

DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
MODELS="${MODELS:-$MODELS_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
ANALYZE_DIR="${ANALYZE_DIR:-}"

PASSTHROUGH_ARGS=("$@")

if [[ -n "$ANALYZE_DIR" && "$ANALYZE_ONLY" == "1" ]]; then
    RUN_ROOT="$ANALYZE_DIR"
    if [[ ! -d "$RUN_ROOT" ]]; then
        echo "[ERR] ANALYZE_DIR not found: ${RUN_ROOT}" >&2
        exit 1
    fi
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_ROOT="${OUTPUT_ROOT}/run_${TIMESTAMP}"
    mkdir -p "$RUN_ROOT"
fi

LOG_FILE="${RUN_ROOT}/pipeline.log"
exec > >(tee "$LOG_FILE") 2>&1

echo ">> Run directory : ${RUN_ROOT}"
echo ">> Method        : ${METHOD_NAME} (${METHOD_CONFIG_NAME})"
echo ">> Datasets      : ${DATASETS}"
echo ">> Models        : ${MODELS}"
echo ">> Passthrough   : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "=================================================="

# ------------------------------------------------------------------
# Stage 1: inference (per model x per dataset)
# ------------------------------------------------------------------
if [[ "$ANALYZE_ONLY" != "1" ]]; then
    for model in $MODELS; do
        echo ""
        echo "============== Inference :: model=${model} =============="
        MODEL_DIR="${RUN_ROOT}/${model}"
        mkdir -p "$MODEL_DIR"

        for ds in $DATASETS; do
            DATA_FILE="${PROJECT_ROOT}/datasets/data/${ds}.json"
            if [[ ! -f "$DATA_FILE" ]]; then
                echo "[WARN] dataset file not found: ${DATA_FILE} -- skipping ${ds}"
                continue
            fi
            OUT_FILE="${MODEL_DIR}/${METHOD_NAME}_${ds}_infer.jsonl"
            echo "------------------------------------------"
            echo ">> [${model}] inference on ${ds}"
            echo ">> output: ${OUT_FILE}"
            echo "------------------------------------------"
            python inference.py \
                --method_name "${METHOD_NAME}" \
                --method_config_name "${METHOD_CONFIG_NAME}" \
                --model_name "${model}" \
                --test_dataset_name "${ds}" \
                --output_path "${OUT_FILE}" \
                "${PASSTHROUGH_ARGS[@]}" || {
                echo "[WARN] inference failed for ${model} / ${ds} -- continuing"
            }
        done
    done
    echo ""
    echo ">> Inference phase finished."
else
    echo ">> ANALYZE_ONLY=1 -- skipping inference."
fi

# ------------------------------------------------------------------
# Stage 2: analysis (per model)
# ------------------------------------------------------------------
echo ""
echo "================= Analysis ====================="
for model in $MODELS; do
    MODEL_DIR="${RUN_ROOT}/${model}"
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "[WARN] no inference output for ${model} -- skipping analysis"
        continue
    fi
    TABLES_DIR="${MODEL_DIR}/_tables"
    echo "------------------------------------------"
    echo ">> [${model}] building Tables 1..6 -> ${TABLES_DIR}"
    echo "------------------------------------------"
    python scripts/diagnostic/analyze_diagnostic.py \
        --infer_dir "${MODEL_DIR}" \
        --output_dir "${TABLES_DIR}" \
        --datasets ${DATASETS} \
        --filename_pattern "${METHOD_NAME}_{dataset}_infer.jsonl" \
        --strict || {
        echo "[WARN] analysis failed (or sanity checks failed) for ${model}"
    }
done

echo ""
echo "=================================================="
echo ">> SCC diagnostic experiment finished."
echo ">> Run dir : ${RUN_ROOT}"
echo ">> Tables  : ${RUN_ROOT}/<model>/_tables/table1..6_*.csv"
echo "=================================================="
