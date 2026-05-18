#!/usr/bin/env bash
# ==============================================================================
# Cluster A — 3B code pipeline, **topk3 family**, mad_scc only.
#
# Differences from exp/run_scc_code_3b.sh:
#   * METHODS default: "mad_scc" (NOT "soo_scc mad_scc")
#   * Configs:  config_code_topk3_aX_*.yaml (top_k=3 instead of 2)
#   * Output root: results_mad_scc_code_topk3/run_<TS>/
#   * Paper tables suffix: _3b_code_topk3 (does not collide with the
#                          original _3b_code tables)
#
# Hypothesis under test:
#   mad_scc's negative ablation on code at top_k=2 (full SCC -2.4%
#   HumanEval / -9.3% MBPP vs A0) is caused by contribution-DAG routing
#   sparsifying mad_vote's broadcast topology. Each agent at top_k=2
#   sees only 2 of 4 peers, breaking mad_vote's core advantage. Raising
#   top_k to 3 recovers most of the broadcast information density and
#   should narrow / close the negative gap on routing-active cells
#   (A2 + A4).
#
# Cells (same 5-cell code ablation matrix as the base run):
#   A0   |  -    |   -   | BLEU plurality (baseline)
#   A3   |  -    |   -   | argmax(contribution)
#   A1   |  ✓   |   -   | argmax
#   A2   |  -    |   ✓   | argmax  ← top_k=3 active
#   A4   |  ✓   |   ✓   | argmax  ← top_k=3 active (headline cell)
#
# Per method: 5 variants × 2 datasets = 10 inference + 10 analysis.
# Total: 1 method × 20 jobs = 20 jobs.
#
# Stage 1 (Inference): inference.py with config_code_topk3_aX_*.yaml.
#                       emit_diagnostic=true, so per-agent / per-round
#                       code is graded inline via subprocess.
# Stage 2 (Per-variant analysis): scripts/diagnostic/analyze_diagnostic.py
#                       — same 6 tables as math/MCQ.
# Stage 3 (Paper tables): scripts/diagnostic/build_madvote_scc_tables.py
#                       — paper_tables/mad_scc_A{1,2}_3b_code_topk3.{csv,md}
#
# Usage (from MASLab/ project root):
#   bash exp/run_scc_code_topk3_3b.sh                         # full run
#   bash exp/run_scc_code_topk3_3b.sh --max_samples 5         # smoke test
#   DATASETS=HumanEval bash exp/run_scc_code_topk3_3b.sh      # single dataset
#   VARIANTS_FILTER="A0_vanilla A4_all" \
#     bash exp/run_scc_code_topk3_3b.sh                       # contrast only
#   EVAL_ONLY=1 ANALYZE_DIR_MAD_SCC=results_mad_scc_code_topk3/run_<TS> \
#     bash exp/run_scc_code_topk3_3b.sh                       # rebuild tables
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------
# Defaults (overridable via env vars)
# ------------------------------------------------------------------
METHODS_DEFAULT="mad_scc"
METHODS="${METHODS:-$METHODS_DEFAULT}"
DATASETS_DEFAULT="HumanEval MBPP"
DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
MODEL="${MODEL:-qwen25-3b-instruct}"
MODEL_LABEL="${MODEL_LABEL:-3b}"
EVAL_ONLY="${EVAL_ONLY:-0}"
VARIANTS_FILTER="${VARIANTS_FILTER:-}"

# Validate methods. Only mad_scc has topk3 configs in this experiment
# family — soo_scc lives in the original code run, since it already
# shows positive ablation at top_k=2 (no need to widen).
for m in $METHODS; do
    if [[ "$m" != "mad_scc" ]]; then
        echo "[ERR] METHODS must contain only mad_scc for the topk3 family (got: ${m})." >&2
        echo "      soo_scc topk3 configs do not exist in this experiment." >&2
        exit 1
    fi
done

# Variant order: A0 baseline → A3 (+argmax) → A1 (+trig) → A2 (+route) → A4
# Same as the base code pipeline.
ALL_VARIANTS=(
    "A0_vanilla:config_code_topk3_a0_vanilla"
    "A3_aggregation:config_code_topk3_a3_aggregation"
    "A1_triggering:config_code_topk3_a1_triggering"
    "A2_routing:config_code_topk3_a2_routing"
    "A4_all:config_code_topk3_a4_all"
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
# Resolve per-method RUN_ROOTs (own results_<method>_code_topk3 tree).
# ------------------------------------------------------------------
if [[ "$EVAL_ONLY" != "1" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi

declare -A RUN_ROOTS

for METHOD in $METHODS; do
    if [[ "$EVAL_ONLY" == "1" ]]; then
        var_name="ANALYZE_DIR_$(echo "$METHOD" | tr '[:lower:]' '[:upper:]')"
        explicit_dir="${!var_name:-}"
        if [[ -n "$explicit_dir" ]]; then
            if [[ ! -d "$explicit_dir" ]]; then
                echo "[ERR] ${var_name}=${explicit_dir} not found" >&2
                exit 1
            fi
            RUN_ROOTS[$METHOD]="$explicit_dir"
        else
            latest=$(ls -dt results_${METHOD}_code_topk3/run_* 2>/dev/null | head -1 || true)
            if [[ -z "$latest" ]]; then
                echo "[ERR] no results_${METHOD}_code_topk3/run_* found; set ${var_name}" >&2
                exit 1
            fi
            RUN_ROOTS[$METHOD]="$latest"
        fi
    else
        RUN_ROOT="./results_${METHOD}_code_topk3/run_${TIMESTAMP}"
        mkdir -p "$RUN_ROOT"
        RUN_ROOTS[$METHOD]="$RUN_ROOT"
    fi
done

first_method=$(echo $METHODS | awk '{print $1}')
LOG_FILE="${RUN_ROOTS[$first_method]}/pipeline.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=================================================="
echo "  Cluster A — 3B code pipeline (topk3 family, mad_scc only)"
echo "  Time         : $(date)"
echo "  Methods      : ${METHODS}"
for METHOD in $METHODS; do
    echo "  Run dir [${METHOD}] : ${RUN_ROOTS[$METHOD]}"
done
echo "  Model        : ${MODEL} (label=${MODEL_LABEL})"
echo "  Datasets     : ${DATASETS}"
echo "  Variants     : ${VARIANTS[*]}"
echo "  Pass-thru    : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "  Log          : ${LOG_FILE}"
echo "=================================================="

# ==================================================================
# Per-method pipeline
# ==================================================================
for METHOD in $METHODS; do
    RUN_ROOT="${RUN_ROOTS[$METHOD]}"
    echo ""
    echo "##############################################################"
    echo "#                     METHOD = ${METHOD}"
    echo "#                     RUN   = ${RUN_ROOT}"
    echo "##############################################################"

    # --------------------------------------------------------------
    # Stage 1: inference (per variant × per dataset)
    # --------------------------------------------------------------
    if [[ "$EVAL_ONLY" != "1" ]]; then
        for entry in "${VARIANTS[@]}"; do
            VAR_ID="${entry%%:*}"
            CFG_NAME="${entry##*:}"
            VAR_DIR="${RUN_ROOT}/${VAR_ID}"
            mkdir -p "$VAR_DIR"

            echo ""
            echo "============== [${METHOD}] Inference :: ${VAR_ID} (${CFG_NAME}) =============="
            for ds in $DATASETS; do
                DATA_FILE="${PROJECT_ROOT}/datasets/data/${ds}.json"
                if [[ ! -f "$DATA_FILE" ]]; then
                    echo "[WARN] dataset file not found: ${DATA_FILE} -- skipping ${ds}"
                    continue
                fi
                OUT_FILE="${VAR_DIR}/${METHOD}_${ds}_infer.jsonl"
                echo "------------------------------------------"
                echo ">> [${METHOD}/${VAR_ID}] inference on ${ds}"
                echo ">> output: ${OUT_FILE}"
                echo "------------------------------------------"
                python inference.py \
                    --method_name "${METHOD}" \
                    --method_config_name "${CFG_NAME}" \
                    --model_name "${MODEL}" \
                    --test_dataset_name "${ds}" \
                    --output_path "${OUT_FILE}" \
                    "${PASSTHROUGH_ARGS[@]}" || {
                    echo "[WARN] inference failed for ${METHOD}/${VAR_ID}/${ds} -- continuing"
                }
            done
        done
        echo ""
        echo ">> [${METHOD}] Inference phase finished."
    else
        echo ">> [${METHOD}] EVAL_ONLY=1 -- skipping inference."
    fi

    # --------------------------------------------------------------
    # Stage 2: per-variant analysis (6 diagnostic tables)
    # --------------------------------------------------------------
    echo ""
    echo "================= [${METHOD}] Analysis ====================="
    for entry in "${VARIANTS[@]}"; do
        VAR_ID="${entry%%:*}"
        VAR_DIR="${RUN_ROOT}/${VAR_ID}"
        if [[ ! -d "$VAR_DIR" ]]; then
            echo "[WARN] no inference output for ${METHOD}/${VAR_ID} -- skipping analysis"
            continue
        fi
        TABLES_DIR="${VAR_DIR}/_tables"
        echo "------------------------------------------"
        echo ">> [${METHOD}/${VAR_ID}] building Tables 1..6 -> ${TABLES_DIR}"
        echo "------------------------------------------"
        python scripts/diagnostic/analyze_diagnostic.py \
            --infer_dir "${VAR_DIR}" \
            --output_dir "${TABLES_DIR}" \
            --datasets ${DATASETS} \
            --filename_pattern "${METHOD}_{dataset}_infer.jsonl" \
            --strict || {
            echo "[WARN] analysis failed (or sanity checks failed) for ${METHOD}/${VAR_ID}"
        }
    done

    # --------------------------------------------------------------
    # Stage 3: aggregate paper tables (A1: per-variant; A2: per-dataset)
    # --------------------------------------------------------------
    echo ""
    echo "================ [${METHOD}] Paper Tables =================="
    mkdir -p paper_tables
    python scripts/diagnostic/build_madvote_scc_tables.py \
        --run_dir "${RUN_ROOT}" \
        --model_label "${MODEL_LABEL}_code_topk3" \
        --filename_pattern "${METHOD}_{dataset}_infer.jsonl" \
        --datasets ${DATASETS} \
        --out_dir paper_tables \
        --prefix "${METHOD}" || {
        echo "[WARN] paper_tables aggregation failed for ${METHOD}."
    }
done

echo ""
echo "=================================================="
echo ">> Cluster A 3B code pipeline (topk3, mad_scc) finished: $(date)"
for METHOD in $METHODS; do
    echo ">> [${METHOD}] Run dir          : ${RUN_ROOTS[$METHOD]}"
    echo ">> [${METHOD}] Per-variant tables: ${RUN_ROOTS[$METHOD]}/<variant>/_tables/"
    echo ">> [${METHOD}] Paper tables     : paper_tables/${METHOD}_A1_${MODEL_LABEL}_code_topk3.{csv,md}"
    echo ">>                                paper_tables/${METHOD}_A2_${MODEL_LABEL}_code_topk3.{csv,md}"
done
echo "=================================================="
