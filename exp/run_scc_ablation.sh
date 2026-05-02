#!/usr/bin/env bash
# ==============================================================================
# SCC ablation experiment -- Experiment 2.
#
# Loops 4 SCC variants on a single representative model (default 3B):
#   scc_full         -> config_diag                (all three modules ON)
#   no_triggering    -> config_diag_no_triggering  (consensus checks OFF)
#   no_routing       -> config_diag_no_routing     (full mesh, contribution
#                                                    routing/diversity OFF)
#   no_aggregation   -> config_diag_no_aggregation (plain plurality, no
#                                                    centroid fallback)
#
# Each variant produces the same diagnostic schema as mad_vote, so the
# existing analyze_diagnostic.py builds Tables 1-6 per variant.
# scripts/diagnostic/build_ablation_table.py then merges them with the
# existing 3B mad_vote run and an Initial-Vote row derived from mad_vote.
#
# Usage:
#   bash exp/run_scc_ablation.sh
#   MODEL=qwen25-7b-instruct bash exp/run_scc_ablation.sh
#   VARIANTS="scc_full no_triggering" bash exp/run_scc_ablation.sh
#   bash exp/run_scc_ablation.sh --max_samples 100
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=false

DATASETS_DEFAULT="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
VARIANTS_DEFAULT="scc_full no_triggering no_routing no_aggregation"
MODEL_DEFAULT="qwen25-3b-instruct"
METHOD_NAME="soo_centered_v3"
OUTPUT_ROOT_DEFAULT="./results_ablation_scc"

DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
VARIANTS="${VARIANTS:-$VARIANTS_DEFAULT}"
MODEL="${MODEL:-$MODEL_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
ANALYZE_DIR="${ANALYZE_DIR:-}"

PASSTHROUGH_ARGS=("$@")

# variant -> config_name
config_for_variant() {
    case "$1" in
        scc_full)        echo "config_diag" ;;
        no_triggering)   echo "config_diag_no_triggering" ;;
        no_routing)      echo "config_diag_no_routing" ;;
        no_aggregation)  echo "config_diag_no_aggregation" ;;
        *) echo "[ERR] unknown variant: $1" >&2; return 1 ;;
    esac
}

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
echo ">> Method        : ${METHOD_NAME}"
echo ">> Model         : ${MODEL}"
echo ">> Datasets      : ${DATASETS}"
echo ">> Variants      : ${VARIANTS}"
echo ">> Passthrough   : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "=================================================="

# ------------------------------------------------------------------
# Stage 1: inference (per variant x per dataset)
# ------------------------------------------------------------------
if [[ "$ANALYZE_ONLY" != "1" ]]; then
    for variant in $VARIANTS; do
        cfg="$(config_for_variant "$variant")"
        VARIANT_DIR="${RUN_ROOT}/${variant}"
        mkdir -p "$VARIANT_DIR"
        echo ""
        echo "============== Inference :: variant=${variant} (${cfg}) =============="

        for ds in $DATASETS; do
            DATA_FILE="${PROJECT_ROOT}/datasets/data/${ds}.json"
            if [[ ! -f "$DATA_FILE" ]]; then
                echo "[WARN] dataset file not found: ${DATA_FILE} -- skipping ${ds}"
                continue
            fi
            OUT_FILE="${VARIANT_DIR}/${METHOD_NAME}_${ds}_infer.jsonl"
            echo "------------------------------------------"
            echo ">> [${variant}] inference on ${ds}"
            echo ">> output: ${OUT_FILE}"
            echo "------------------------------------------"
            python inference.py \
                --method_name "${METHOD_NAME}" \
                --method_config_name "${cfg}" \
                --model_name "${MODEL}" \
                --test_dataset_name "${ds}" \
                --output_path "${OUT_FILE}" \
                "${PASSTHROUGH_ARGS[@]}" || {
                echo "[WARN] inference failed for ${variant} / ${ds} -- continuing"
            }
        done
    done
    echo ""
    echo ">> Inference phase finished."
else
    echo ">> ANALYZE_ONLY=1 -- skipping inference."
fi

# ------------------------------------------------------------------
# Stage 2: analysis (per variant)
# ------------------------------------------------------------------
echo ""
echo "================= Analysis ====================="
for variant in $VARIANTS; do
    VARIANT_DIR="${RUN_ROOT}/${variant}"
    if [[ ! -d "$VARIANT_DIR" ]]; then
        echo "[WARN] no inference output for ${variant} -- skipping analysis"
        continue
    fi
    TABLES_DIR="${VARIANT_DIR}/_tables"
    echo "------------------------------------------"
    echo ">> [${variant}] building Tables 1..6 -> ${TABLES_DIR}"
    echo "------------------------------------------"
    python scripts/diagnostic/analyze_diagnostic.py \
        --infer_dir "${VARIANT_DIR}" \
        --output_dir "${TABLES_DIR}" \
        --datasets ${DATASETS} \
        --filename_pattern "${METHOD_NAME}_{dataset}_infer.jsonl" \
        --strict || {
        echo "[WARN] analysis failed (or sanity checks failed) for ${variant}"
    }
done

echo ""
echo "=================================================="
echo ">> SCC ablation experiment finished."
echo ">> Run dir : ${RUN_ROOT}"
echo ">> Tables  : ${RUN_ROOT}/<variant>/_tables/table1..6_*.csv"
echo "=================================================="
