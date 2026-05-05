#!/usr/bin/env bash
# ==============================================================================
# Cluster A — 1.5B end-to-end pipeline for `mad_scc` (mad_vote baseline +
# scc_components SCC modules). Mirrors run_madvote_scc_1.5b.sh but targets
# the v3-faithful refactor (methods/mad_scc/) and writes outputs under
# results_mad_scc/.
#
#   Stage 1 (Inference)     : 6 variants x 5 datasets under
#                             results_mad_scc/run_<TS>/<variant>/
#       A0_vanilla       -> all three SCC flags off
#       A1_triggering    -> +Spectral / Answer-plurality Triggering
#       A2_routing       -> +Contribution-DAG Routing (sim_thr + diversity_p)
#       A3_aggregation   -> +Contribution-weighted Aggregation
#       A4_all           -> +Triggering +Routing +Aggregation (full overlay)
#       A5_trig_rout     -> +Triggering +Routing (no weighted Aggregation)
#
#   Stage 2 (Per-variant analysis): writes table1..6 to <variant>/_tables/
#                                    via scripts/diagnostic/analyze_diagnostic.py
#
#   Stage 3 (Paper tables)  : aggregates A1 (6 rows) and A2 (5 datasets) into
#                             paper_tables/mad_scc_A1_1.5b.{csv,md}
#                             paper_tables/mad_scc_A2_1.5b.{csv,md}
#
# Round 0 is replayed from the SHARED initial pool at
#   results_archive/results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged
# so all six variants debate from identical initial answer pools (eliminates
# the bucket-distribution confound the legacy run hit).
#
# Usage (from MASLab/ project root):
#   bash exp/run_mad_scc_1.5b.sh
#   bash exp/run_mad_scc_1.5b.sh --max_samples 5         # smoke test
#   ANALYZE_ONLY=1 ANALYZE_DIR=results_mad_scc/run_<TS> \
#     bash exp/run_mad_scc_1.5b.sh                       # rebuild tables
#   DATASETS="GSM8K AQUA-RAT" bash exp/run_mad_scc_1.5b.sh
#   VARIANTS_FILTER="A0_vanilla A4_all" bash exp/run_mad_scc_1.5b.sh
#   POOL_DIR=results_archive/results_diagnostic/run_other \
#     bash exp/run_mad_scc_1.5b.sh                       # different pool
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------
# Defaults (overridable via env vars)
# ------------------------------------------------------------------
DATASETS_DEFAULT="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
MODEL="${MODEL:-qwen25-1.5b-instruct}"
MODEL_LABEL="${MODEL_LABEL:-1.5b}"
METHOD_NAME="mad_scc"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results_mad_scc}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
ANALYZE_DIR="${ANALYZE_DIR:-}"
VARIANTS_FILTER="${VARIANTS_FILTER:-}"
POOL_DIR="${POOL_DIR:-results_archive/results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged}"
POOL_PATTERN="${POOL_PATTERN:-mad_vote_{dataset}_infer.jsonl}"

ALL_VARIANTS=(
    "A0_vanilla:config_a0_vanilla"
    "A1_triggering:config_a1_triggering"
    "A2_routing:config_a2_routing"
    "A3_aggregation:config_a3_aggregation"
    "A4_all:config_a4_all"
    "A5_trig_rout:config_a5_trig_rout"
)

VARIANTS=()
if [[ -n "$VARIANTS_FILTER" ]]; then
    for entry in "${ALL_VARIANTS[@]}"; do
        VAR_ID="${entry%%:*}"
        for keep in $VARIANTS_FILTER; do
            if [[ "$keep" == "$VAR_ID" ]]; then
                VARIANTS+=("$entry")
                break
            fi
        done
    done
else
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

PASSTHROUGH_ARGS=("$@")

# ------------------------------------------------------------------
# Resolve RUN_ROOT
# ------------------------------------------------------------------
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

echo "=================================================="
echo "  Cluster A — 1.5B end-to-end pipeline (mad_scc)"
echo "  Time         : $(date)"
echo "  Run dir      : ${RUN_ROOT}"
echo "  Model        : ${MODEL} (label=${MODEL_LABEL})"
echo "  Datasets     : ${DATASETS}"
echo "  Variants     : ${VARIANTS[*]}"
echo "  Pool dir     : ${POOL_DIR}"
echo "  Pool pattern : ${POOL_PATTERN}"
echo "  Pass-thru    : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "  Log          : ${LOG_FILE}"
echo "=================================================="

# ------------------------------------------------------------------
# Stage 1: inference (per variant x per dataset)
# ------------------------------------------------------------------
if [[ "$ANALYZE_ONLY" != "1" ]]; then
    for entry in "${VARIANTS[@]}"; do
        VAR_ID="${entry%%:*}"
        CFG_NAME="${entry##*:}"
        VAR_DIR="${RUN_ROOT}/${VAR_ID}"
        mkdir -p "$VAR_DIR"

        echo ""
        echo "============== Inference :: ${VAR_ID} (${CFG_NAME}) =============="
        for ds in $DATASETS; do
            DATA_FILE="${PROJECT_ROOT}/datasets/data/${ds}.json"
            if [[ ! -f "$DATA_FILE" ]]; then
                echo "[WARN] dataset file not found: ${DATA_FILE} -- skipping ${ds}"
                continue
            fi
            OUT_FILE="${VAR_DIR}/${METHOD_NAME}_${ds}_infer.jsonl"
            echo "------------------------------------------"
            echo ">> [${VAR_ID}] inference on ${ds}"
            echo ">> output: ${OUT_FILE}"
            echo "------------------------------------------"
            python inference.py \
                --method_name "${METHOD_NAME}" \
                --method_config_name "${CFG_NAME}" \
                --model_name "${MODEL}" \
                --test_dataset_name "${ds}" \
                --output_path "${OUT_FILE}" \
                --initial_pool_dir "${POOL_DIR}" \
                --initial_pool_filename_pattern "${POOL_PATTERN}" \
                "${PASSTHROUGH_ARGS[@]}" || {
                echo "[WARN] inference failed for ${VAR_ID}/${ds} -- continuing"
            }
        done
    done
    echo ""
    echo ">> Inference phase finished."
else
    echo ">> ANALYZE_ONLY=1 -- skipping inference."
fi

# ------------------------------------------------------------------
# Stage 2: per-variant analysis
# ------------------------------------------------------------------
echo ""
echo "================= Analysis ====================="
for entry in "${VARIANTS[@]}"; do
    VAR_ID="${entry%%:*}"
    VAR_DIR="${RUN_ROOT}/${VAR_ID}"
    if [[ ! -d "$VAR_DIR" ]]; then
        echo "[WARN] no inference output for ${VAR_ID} -- skipping analysis"
        continue
    fi
    TABLES_DIR="${VAR_DIR}/_tables"
    echo "------------------------------------------"
    echo ">> [${VAR_ID}] building Tables 1..6 -> ${TABLES_DIR}"
    echo "------------------------------------------"
    python scripts/diagnostic/analyze_diagnostic.py \
        --infer_dir "${VAR_DIR}" \
        --output_dir "${TABLES_DIR}" \
        --datasets ${DATASETS} \
        --filename_pattern "${METHOD_NAME}_{dataset}_infer.jsonl" \
        --strict || {
        echo "[WARN] analysis failed (or sanity checks failed) for ${VAR_ID}"
    }
done

# ------------------------------------------------------------------
# Stage 3: aggregate Cluster A paper tables (A1 + A2) for this model
# ------------------------------------------------------------------
echo ""
echo "================ Paper Tables =================="
mkdir -p paper_tables
python scripts/diagnostic/build_madvote_scc_tables.py \
    --run_dir "${RUN_ROOT}" \
    --model_label "${MODEL_LABEL}" \
    --filename_pattern "${METHOD_NAME}_{dataset}_infer.jsonl" \
    --datasets ${DATASETS} \
    --out_dir paper_tables \
    --prefix "${METHOD_NAME}" || {
    echo "[WARN] paper_tables aggregation failed."
}

echo ""
echo "=================================================="
echo ">> Cluster A 1.5B pipeline (mad_scc) finished: $(date)"
echo ">> Run dir : ${RUN_ROOT}"
echo ">> Per-variant tables : ${RUN_ROOT}/<variant>/_tables/"
echo ">> Paper tables       : paper_tables/${METHOD_NAME}_A1_${MODEL_LABEL}.{csv,md}"
echo ">>                      paper_tables/${METHOD_NAME}_A2_${MODEL_LABEL}.{csv,md}"
echo "=================================================="
