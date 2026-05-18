#!/usr/bin/env bash
# ==============================================================================
# Cluster A — 1.5B code pipeline. Runs **both** soo_scc and mad_scc end-to-end
# with the 5-cell code ablation matrix on HumanEval + MBPP-500.
#
#   Cell      | Trig | Route | Aggregation
#   ----------|------|-------|-----------------------------
#   A0        |  ✗   |   ✗   | BLEU plurality (baseline)
#   A3        |  ✗   |   ✗   | argmax(contribution)
#   A1        |  ✓   |   ✗   | argmax
#   A2        |  ✗   |   ✓   | argmax
#   A4        |  ✓   |   ✓   | argmax (full SCC)
#
# A5 is dropped on code because the plurality module is disabled
# (enable_answer_consensus=false on SCC variants), making the "weighted vs
# plain plurality" axis collapse. See methods/<method>/configs/
# config_code_a5_trig_rout.yaml header notes — kept only for tooling
# symmetry, would produce numbers identical to A4 if run.
#
# Per method: 5 variants × 2 datasets = 10 inference + 10 eval.
# Total: 2 methods × 20 jobs = 40 jobs.
#
# Per-method outputs go to results_<method>_code/run_<TS>/<variant>/ with
# the SAME timestamp across both methods (so paired runs are easy to spot).
# Per-method paper tables: paper_tables/<method>_code_1.5b.{csv,md}.
#
# Stage 1 (Inference): inference.py with config_code_aX_*.yaml. Configs now
#                       have emit_diagnostic=true, so each agent's per-round
#                       code is graded inline via subprocess (the diagnostic
#                       JSONL schema is identical to math/MCQ).
# Stage 2 (Per-variant analysis): scripts/diagnostic/analyze_diagnostic.py
#                       — same 6 tables as math/MCQ (initial_diagnostics,
#                       regime_analysis, accuracy_decomposition, ...).
# Stage 3 (Paper tables): scripts/diagnostic/build_madvote_scc_tables.py
#                       — same aggregation as math/MCQ, so paper_tables files
#                       align column-for-column.
#
# Usage (from MASLab/ project root):
#   bash exp/run_scc_code_1.5b.sh                      # full both-methods run
#   bash exp/run_scc_code_1.5b.sh --max_samples 5      # smoke test
#   DATASETS=HumanEval bash exp/run_scc_code_1.5b.sh   # single dataset
#   METHODS=soo_scc bash exp/run_scc_code_1.5b.sh      # one method only
#   VARIANTS_FILTER="A0_vanilla A4_all" bash exp/run_scc_code_1.5b.sh
#   EVAL_ONLY=1 ANALYZE_DIR_SOO_SCC=results_soo_scc_code/run_<TS> \
#     ANALYZE_DIR_MAD_SCC=results_mad_scc_code/run_<TS> \
#     bash exp/run_scc_code_1.5b.sh                    # rebuild tables only
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------
# Defaults (overridable via env vars)
# ------------------------------------------------------------------
METHODS_DEFAULT="soo_scc mad_scc"
METHODS="${METHODS:-$METHODS_DEFAULT}"
DATASETS_DEFAULT="HumanEval MBPP"
DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
MODEL="${MODEL:-qwen25-1.5b-instruct}"
MODEL_LABEL="${MODEL_LABEL:-1.5b}"
EVAL_ONLY="${EVAL_ONLY:-0}"
VARIANTS_FILTER="${VARIANTS_FILTER:-}"

# Validate methods.
for m in $METHODS; do
    if [[ "$m" != "soo_scc" && "$m" != "mad_scc" ]]; then
        echo "[ERR] unknown method in METHODS: ${m} (expected soo_scc | mad_scc)" >&2
        exit 1
    fi
done

# Variant order in the table:
#   A0 baseline → A3 (+ argmax) → A1 (+ trig) → A2 (+ route) → A4 (full SCC)
ALL_VARIANTS=(
    "A0_vanilla:config_code_a0_vanilla"
    "A3_aggregation:config_code_a3_aggregation"
    "A1_triggering:config_code_a1_triggering"
    "A2_routing:config_code_a2_routing"
    "A4_all:config_code_a4_all"
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
# Resolve per-method RUN_ROOTs (shared timestamp for paired runs).
# ------------------------------------------------------------------
if [[ "$EVAL_ONLY" != "1" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi

declare -A RUN_ROOTS

for METHOD in $METHODS; do
    if [[ "$EVAL_ONLY" == "1" ]]; then
        # Per-method analyze dir resolution.
        # Priority: ANALYZE_DIR_SOO_SCC / ANALYZE_DIR_MAD_SCC env var, then
        # latest results_<method>_code/run_* by mtime.
        var_name="ANALYZE_DIR_$(echo "$METHOD" | tr '[:lower:]' '[:upper:]')"
        explicit_dir="${!var_name:-}"
        if [[ -n "$explicit_dir" ]]; then
            if [[ ! -d "$explicit_dir" ]]; then
                echo "[ERR] ${var_name}=${explicit_dir} not found" >&2
                exit 1
            fi
            RUN_ROOTS[$METHOD]="$explicit_dir"
        else
            latest=$(ls -dt results_${METHOD}_code/run_* 2>/dev/null | head -1 || true)
            if [[ -z "$latest" ]]; then
                echo "[ERR] no results_${METHOD}_code/run_* found; set ${var_name}" >&2
                exit 1
            fi
            RUN_ROOTS[$METHOD]="$latest"
        fi
    else
        RUN_ROOT="./results_${METHOD}_code/run_${TIMESTAMP}"
        mkdir -p "$RUN_ROOT"
        RUN_ROOTS[$METHOD]="$RUN_ROOT"
    fi
done

# Combined log: write to the first method's run dir for simplicity; per-method
# stdout still shows method tags.
first_method=$(echo $METHODS | awk '{print $1}')
LOG_FILE="${RUN_ROOTS[$first_method]}/pipeline.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=================================================="
echo "  Cluster A — 1.5B code pipeline (both methods)"
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
    #                   the SAME initial answer pool, isolating SCC
    #                   module effects from round-0 sampling noise.
    #
    # Pool source: ${RUN_ROOT}/A0_vanilla/${METHOD}_{dataset}_infer.jsonl
    # If A0 is filtered out, falls back to the most recent
    # results_${METHOD}_code/run_*/A0_vanilla; if none exists, variants
    # sample fresh (with a warning).
    # --------------------------------------------------------------
    if [[ "$EVAL_ONLY" != "1" ]]; then
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
            latest_a0=$(ls -dt results_${METHOD}_code/run_*/A0_vanilla 2>/dev/null | head -1 || true)
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
    #
    # analyze_diagnostic.py reads the diagnostic JSONL emitted by
    # inference (now containing per-agent / per-round code-grading
    # records) and writes table1..6 to <variant>/_tables/. Same script
    # and schema as math/MCQ — no code-specific fork.
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
        --model_label "${MODEL_LABEL}_code" \
        --filename_pattern "${METHOD}_{dataset}_infer.jsonl" \
        --datasets ${DATASETS} \
        --out_dir paper_tables \
        --prefix "${METHOD}" || {
        echo "[WARN] paper_tables aggregation failed for ${METHOD}."
    }
done

echo ""
echo "=================================================="
echo ">> Cluster A 1.5B code pipeline (both methods) finished: $(date)"
for METHOD in $METHODS; do
    echo ">> [${METHOD}] Run dir          : ${RUN_ROOTS[$METHOD]}"
    echo ">> [${METHOD}] Per-variant tables: ${RUN_ROOTS[$METHOD]}/<variant>/_tables/"
    echo ">> [${METHOD}] Paper tables     : paper_tables/${METHOD}_A1_${MODEL_LABEL}_code.{csv,md}"
    echo ">>                                paper_tables/${METHOD}_A2_${MODEL_LABEL}_code.{csv,md}"
done
echo "=================================================="
