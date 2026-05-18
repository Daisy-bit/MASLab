#!/usr/bin/env bash
# ==============================================================================
# Cluster A — 3B code pipeline, **topk4 family**, mad_scc only.
#
# Differences from exp/run_scc_code_3b.sh:
#   * METHODS default: "mad_scc" (NOT "soo_scc mad_scc")
#   * Configs:  config_code_topk4_aX_*.yaml (top_k=4 instead of 2)
#   * Output root: results_mad_scc_code_topk4/run_<TS>/
#   * Paper tables suffix: _3b_code_topk4 (does not collide with the
#                          original _3b_code tables)
#
# Hypothesis under test:
#   Follow-up to the topk3 family (which still left mad_scc A4 below
#   A0 on code, ruling out "partial broadcast" as the sole cause). With
#   5 agents and top_k=4, every agent receives from all 4 other peers
#   — contribution-DAG routing FULLY DEGENERATES to a full mesh
#   (identical connectivity to mad_vote's broadcast baseline). Any
#   remaining gap between A4_topk4 and A0 is therefore attributable
#   purely to the three SCC algorithmic components:
#     1. contribution-based edge ORDERING (vs no ordering in A0)
#     2. spectral early-stop triggering (vs full-debate in A0)
#     3. argmax-contribution aggregation (vs BLEU plurality in A0)
#   If A4_topk4 still loses to A0, the SCC modules themselves are the
#   issue on code, independent of routing density.
#
# Cells (same 5-cell code ablation matrix as the base run):
#   A0   |  -    |   -   | BLEU plurality (baseline)
#   A3   |  -    |   -   | argmax(contribution)
#   A1   |  ✓   |   -   | argmax
#   A2   |  -    |   ✓   | argmax  ← top_k=4 active
#   A4   |  ✓   |   ✓   | argmax  ← top_k=4 active (headline cell)
#
# Per method: 5 variants × 2 datasets = 10 inference + 10 analysis.
# Total: 1 method × 20 jobs = 20 jobs.
#
# Stage 1 (Inference): inference.py with config_code_topk4_aX_*.yaml.
#                       emit_diagnostic=true, so per-agent / per-round
#                       code is graded inline via subprocess.
# Stage 2 (Per-variant analysis): scripts/diagnostic/analyze_diagnostic.py
#                       — same 6 tables as math/MCQ.
# Stage 3 (Paper tables): scripts/diagnostic/build_madvote_scc_tables.py
#                       — paper_tables/mad_scc_A{1,2}_3b_code_topk4.{csv,md}
#
# Usage (from MASLab/ project root):
#   bash exp/run_scc_code_topk4_3b.sh                         # full run
#   bash exp/run_scc_code_topk4_3b.sh --max_samples 5         # smoke test
#   DATASETS=HumanEval bash exp/run_scc_code_topk4_3b.sh      # single dataset
#   VARIANTS_FILTER="A0_vanilla A4_all" \
#     bash exp/run_scc_code_topk4_3b.sh                       # contrast only
#   EVAL_ONLY=1 ANALYZE_DIR_MAD_SCC=results_mad_scc_code_topk4/run_<TS> \
#     bash exp/run_scc_code_topk4_3b.sh                       # rebuild tables
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

# Validate methods. Only mad_scc has topk4 configs in this experiment
# family — soo_scc lives in the original code run, since it already
# shows positive ablation at top_k=2 (no need to widen).
for m in $METHODS; do
    if [[ "$m" != "mad_scc" ]]; then
        echo "[ERR] METHODS must contain only mad_scc for the topk4 family (got: ${m})." >&2
        echo "      soo_scc topk4 configs do not exist in this experiment." >&2
        exit 1
    fi
done

# Variant order: A0 baseline → A3 (+argmax) → A1 (+trig) → A2 (+route) → A4
# Same as the base code pipeline.
ALL_VARIANTS=(
    "A0_vanilla:config_code_topk4_a0_vanilla"
    "A3_aggregation:config_code_topk4_a3_aggregation"
    "A1_triggering:config_code_topk4_a1_triggering"
    "A2_routing:config_code_topk4_a2_routing"
    "A4_all:config_code_topk4_a4_all"
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
# Resolve per-method RUN_ROOTs (own results_<method>_code_topk4 tree).
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
            latest=$(ls -dt results_${METHOD}_code_topk4/run_* 2>/dev/null | head -1 || true)
            if [[ -z "$latest" ]]; then
                echo "[ERR] no results_${METHOD}_code_topk4/run_* found; set ${var_name}" >&2
                exit 1
            fi
            RUN_ROOTS[$METHOD]="$latest"
        fi
    else
        RUN_ROOT="./results_${METHOD}_code_topk4/run_${TIMESTAMP}"
        mkdir -p "$RUN_ROOT"
        RUN_ROOTS[$METHOD]="$RUN_ROOT"
    fi
done

first_method=$(echo $METHODS | awk '{print $1}')
LOG_FILE="${RUN_ROOTS[$first_method]}/pipeline.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=================================================="
echo "  Cluster A — 3B code pipeline (topk4 family, mad_scc only)"
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
    # Stage 1: inference, two-pass for fair ablation.
    #
    # Pass 1 (no pool): A0_vanilla samples fresh round-0 LLM outputs and
    #                   writes them to its diagnostic JSONL.
    # Pass 2 (pool):    A1..A4 replay round-0 from A0's output via
    #                   --initial_pool_dir so all variants debate from
    #                   the SAME initial answer pool. Pure top_k effect
    #                   isolation (no round-0 sampling noise).
    #
    # Pool source: ${RUN_ROOT}/A0_vanilla/${METHOD}_{dataset}_infer.jsonl
    # If A0 is filtered out (VARIANTS_FILTER excludes A0_vanilla), Pass 1
    # is skipped and Pass 2 runs with --initial_pool_dir auto-resolved
    # to the most recent results_${METHOD}_code_topk4/run_*/A0_vanilla;
    # if no such dir exists, variants sample fresh (with a warning).
    # --------------------------------------------------------------
    if [[ "$EVAL_ONLY" != "1" ]]; then
        # Split variants into A0 (pool producer) and the rest (consumers).
        A0_VARIANT=""
        OTHER_VARIANTS=()
        for entry in "${VARIANTS[@]}"; do
            VAR_ID="${entry%%:*}"
            if [[ "$VAR_ID" == "A0_vanilla" ]]; then
                A0_VARIANT="$entry"
            else
                OTHER_VARIANTS+=("$entry")
            fi
        done

        # ---- Pass 1: A0 fresh ----
        if [[ -n "$A0_VARIANT" ]]; then
            VAR_ID="${A0_VARIANT%%:*}"
            CFG_NAME="${A0_VARIANT##*:}"
            VAR_DIR="${RUN_ROOT}/${VAR_ID}"
            mkdir -p "$VAR_DIR"
            echo ""
            echo "============== [${METHOD}] Pass 1 :: ${VAR_ID} fresh (pool source) =============="
            for ds in $DATASETS; do
                DATA_FILE="${PROJECT_ROOT}/datasets/data/${ds}.json"
                if [[ ! -f "$DATA_FILE" ]]; then
                    echo "[WARN] dataset file not found: ${DATA_FILE} -- skipping ${ds}"
                    continue
                fi
                OUT_FILE="${VAR_DIR}/${METHOD}_${ds}_infer.jsonl"
                echo "------------------------------------------"
                echo ">> [${METHOD}/${VAR_ID}] inference on ${ds} (no pool)"
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
        fi

        # ---- Resolve pool dir for Pass 2 ----
        POOL_DIR="${RUN_ROOT}/A0_vanilla"
        if [[ ! -d "$POOL_DIR" ]]; then
            latest_a0=$(ls -dt results_${METHOD}_code_topk4/run_*/A0_vanilla 2>/dev/null | head -1 || true)
            if [[ -n "$latest_a0" ]]; then
                POOL_DIR="$latest_a0"
                echo "[INFO] A0 not run in this pass; reusing pool from ${POOL_DIR}"
            else
                POOL_DIR=""
                echo "[WARN] no A0 pool available; A1..A4 will sample round-0 fresh"
                echo "       (round-0 randomness will confound the ablation)"
            fi
        fi

        # ---- Pass 2: A1..A4 replay from pool ----
        for entry in "${OTHER_VARIANTS[@]}"; do
            VAR_ID="${entry%%:*}"
            CFG_NAME="${entry##*:}"
            VAR_DIR="${RUN_ROOT}/${VAR_ID}"
            mkdir -p "$VAR_DIR"

            echo ""
            echo "============== [${METHOD}] Pass 2 :: ${VAR_ID} (${CFG_NAME}) =============="
            for ds in $DATASETS; do
                DATA_FILE="${PROJECT_ROOT}/datasets/data/${ds}.json"
                if [[ ! -f "$DATA_FILE" ]]; then
                    echo "[WARN] dataset file not found: ${DATA_FILE} -- skipping ${ds}"
                    continue
                fi
                OUT_FILE="${VAR_DIR}/${METHOD}_${ds}_infer.jsonl"
                echo "------------------------------------------"
                echo ">> [${METHOD}/${VAR_ID}] inference on ${ds}"
                if [[ -n "$POOL_DIR" ]]; then
                    echo ">> initial pool: ${POOL_DIR}"
                fi
                echo ">> output: ${OUT_FILE}"
                echo "------------------------------------------"
                POOL_ARGS=()
                if [[ -n "$POOL_DIR" ]]; then
                    POOL_ARGS=(--initial_pool_dir "$POOL_DIR"
                               --initial_pool_filename_pattern "${METHOD}_{dataset}_infer.jsonl")
                fi
                python inference.py \
                    --method_name "${METHOD}" \
                    --method_config_name "${CFG_NAME}" \
                    --model_name "${MODEL}" \
                    --test_dataset_name "${ds}" \
                    --output_path "${OUT_FILE}" \
                    "${POOL_ARGS[@]}" \
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
        --model_label "${MODEL_LABEL}_code_topk4" \
        --filename_pattern "${METHOD}_{dataset}_infer.jsonl" \
        --datasets ${DATASETS} \
        --out_dir paper_tables \
        --prefix "${METHOD}" || {
        echo "[WARN] paper_tables aggregation failed for ${METHOD}."
    }
done

echo ""
echo "=================================================="
echo ">> Cluster A 3B code pipeline (topk4, mad_scc) finished: $(date)"
for METHOD in $METHODS; do
    echo ">> [${METHOD}] Run dir          : ${RUN_ROOTS[$METHOD]}"
    echo ">> [${METHOD}] Per-variant tables: ${RUN_ROOTS[$METHOD]}/<variant>/_tables/"
    echo ">> [${METHOD}] Paper tables     : paper_tables/${METHOD}_A1_${MODEL_LABEL}_code_topk4.{csv,md}"
    echo ">>                                paper_tables/${METHOD}_A2_${MODEL_LABEL}_code_topk4.{csv,md}"
done
echo "=================================================="
