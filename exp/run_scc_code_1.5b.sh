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
# Stage 1 (Inference): inference.py with config_code_aX_*.yaml
# Stage 2 (Eval): evaluate.py --eval_protocol code (subprocess pass@1)
# Stage 3 (Paper tables): scripts/diagnostic/build_code_tables.py
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
    # Stage 2: evaluation (pass@1 via subprocess sandbox)
    #
    # evaluations/__init__.py auto-routes HumanEval / MBPP / MBPP-500 to
    # eval_func_code regardless of --eval_protocol, so the "code" tag here
    # is just for clarity. evaluate.py writes <basename>_xverify_eval.jsonl
    # next to the infer file (filename suffix is historical).
    # --------------------------------------------------------------
    echo ""
    echo "================= [${METHOD}] Evaluation ====================="
    for entry in "${VARIANTS[@]}"; do
        VAR_ID="${entry%%:*}"
        CFG_NAME="${entry##*:}"
        VAR_DIR="${RUN_ROOT}/${VAR_ID}"
        if [[ ! -d "$VAR_DIR" ]]; then
            echo "[WARN] no inference output for ${METHOD}/${VAR_ID} -- skipping eval"
            continue
        fi
        for ds in $DATASETS; do
            INFER_FILE="${VAR_DIR}/${METHOD}_${ds}_infer.jsonl"
            if [[ ! -f "$INFER_FILE" ]]; then
                echo "[WARN] no infer file for ${METHOD}/${VAR_ID}/${ds} -- skipping eval"
                continue
            fi
            echo "------------------------------------------"
            echo ">> [${METHOD}/${VAR_ID}] eval pass@1 on ${ds}"
            echo "------------------------------------------"
            python evaluate.py \
                --eval_protocol code \
                --model_name "${MODEL}" \
                --tested_dataset_name "${ds}" \
                --tested_method_name "${METHOD}" \
                --tested_method_config_name "${CFG_NAME}" \
                --tested_mas_model_name "${MODEL}" \
                --tested_infer_path "${INFER_FILE}" \
                --overwrite || {
                echo "[WARN] eval failed for ${METHOD}/${VAR_ID}/${ds}"
            }
        done
    done

    # --------------------------------------------------------------
    # Stage 3: aggregate paper table (pass@1 by variant × dataset)
    # --------------------------------------------------------------
    echo ""
    echo "================ [${METHOD}] Paper Tables =================="
    mkdir -p paper_tables
    python scripts/diagnostic/build_code_tables.py \
        --run_dir "${RUN_ROOT}" \
        --method_label "${METHOD}" \
        --model_label "${MODEL_LABEL}" \
        --datasets ${DATASETS} \
        --out_dir paper_tables \
        --prefix "${METHOD}_code" || {
        echo "[WARN] paper_tables aggregation failed for ${METHOD}."
    }
done

echo ""
echo "=================================================="
echo ">> Cluster A 1.5B code pipeline (both methods) finished: $(date)"
for METHOD in $METHODS; do
    echo ">> [${METHOD}] Run dir      : ${RUN_ROOTS[$METHOD]}"
    echo ">> [${METHOD}] Paper tables : paper_tables/${METHOD}_code_${MODEL_LABEL}.{csv,md}"
done
echo "=================================================="
