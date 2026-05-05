#!/usr/bin/env bash
# ==============================================================================
# Run BOTH SCC pipelines (mad_scc + soo_scc) THREE times, then aggregate.
#
# Same seed / same configs / same shared initial-response pool across all
# three runs — only debate-stage LLM sampling varies (temperature=0.5).
# This isolates the effect of debate-phase randomness from any structural
# variance: every variant in every run starts from byte-identical round-0
# answer pools, so per-question bucket assignment is identical across the
# three runs.
#
# Layout produced under ./paper_tables/3runs_<TS>/:
#   mad_scc_A1_1.5b_run{1,2,3}.csv / .md      (per-run paper tables)
#   mad_scc_A1_1.5b_mean.csv                  (cross-run mean only)
#   mad_scc_A1_1.5b_meanstd.csv               (mean ± std)
#   mad_scc_A2_1.5b_run{1,2,3}.csv / .md
#   mad_scc_A2_1.5b_mean.csv / _meanstd.csv
#   soo_scc_A1_1.5b_run{1,2,3}.csv / .md
#   soo_scc_A1_1.5b_mean.csv / _meanstd.csv
#   soo_scc_A2_1.5b_run{1,2,3}.csv / .md
#   soo_scc_A2_1.5b_mean.csv / _meanstd.csv
#
# Per-run inference outputs land in
#   ./results_mad_scc/run_<runTS>/...
#   ./results_soo_scc/run_<runTS>/...
# (each individual run gets its own timestamped sub-dir, kept).
#
# Usage (from MASLab/ project root):
#   bash exp/run_both_scc_1.5b_3runs.sh
#   bash exp/run_both_scc_1.5b_3runs.sh --max_samples 5  # smoke test
#   N_RUNS=5 bash exp/run_both_scc_1.5b_3runs.sh         # change run count
#   STOP_ON_FAIL=1 bash exp/run_both_scc_1.5b_3runs.sh   # abort on any failure
#
# Total inference invocations:  N_RUNS x 2 methods x 6 variants x 5 datasets
#                               = 3 x 60 = 180 (default)
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

N_RUNS="${N_RUNS:-3}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
MODEL_LABEL="${MODEL_LABEL:-1.5b}"
PASSTHROUGH_ARGS=("$@")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUNS_TABLE_DIR="paper_tables/3runs_${TIMESTAMP}"
mkdir -p "${RUNS_TABLE_DIR}"
mkdir -p logs
LOG_FILE="logs/run_both_scc_1.5b_3runs_${TIMESTAMP}.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=================================================="
echo "  Both SCC pipelines x ${N_RUNS} runs"
echo "  Time         : $(date)"
echo "  Model label  : ${MODEL_LABEL}"
echo "  Run table dir: ${RUNS_TABLE_DIR}"
echo "  Log          : ${LOG_FILE}"
echo "  Pass-thru    : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "  Stop-on-fail : ${STOP_ON_FAIL}"
echo "=================================================="

PREFIXES=(
    "mad_scc_A1_${MODEL_LABEL}"
    "mad_scc_A2_${MODEL_LABEL}"
    "soo_scc_A1_${MODEL_LABEL}"
    "soo_scc_A2_${MODEL_LABEL}"
)

run_step() {
    local name="$1"; shift
    echo ""
    echo "----------------------------------------------------------"
    echo ">> [$(date)] starting ${name}"
    echo "----------------------------------------------------------"
    if bash "$@" "${PASSTHROUGH_ARGS[@]}"; then
        echo ">> [$(date)] ${name} finished OK"
        return 0
    else
        local rc=$?
        echo "[ERR] ${name} failed (exit ${rc})"
        if [[ "$STOP_ON_FAIL" == "1" ]]; then
            echo "[ERR] STOP_ON_FAIL=1 — aborting."
            exit 1
        fi
        echo "[WARN] continuing."
        return ${rc}
    fi
}

# Snapshot per-run paper_tables outputs into the 3-runs dir, suffixed by
# run index. Done after each pipeline so the next iteration's outputs
# don't overwrite these.
snapshot_paper_tables() {
    local run_idx="$1"
    for prefix in "${PREFIXES[@]}"; do
        for ext in csv md; do
            local src="paper_tables/${prefix}.${ext}"
            local dst="${RUNS_TABLE_DIR}/${prefix}_run${run_idx}.${ext}"
            if [[ -f "$src" ]]; then
                cp "$src" "$dst"
                echo "  snapshot ${src} -> ${dst}"
            else
                echo "  [WARN] missing ${src} (run${run_idx}); skipping snapshot"
            fi
        done
    done
}

for ((i=1; i<=N_RUNS; i++)); do
    echo ""
    echo "=========================================================="
    echo "  RUN ${i} / ${N_RUNS}  ($(date))"
    echo "=========================================================="

    run_step "mad_scc 1.5B (run ${i})" "${SCRIPT_DIR}/run_mad_scc_1.5b.sh" || true
    run_step "soo_scc 1.5B (run ${i})" "${SCRIPT_DIR}/run_soo_scc_1.5b.sh" || true

    echo ""
    echo "----- snapshot run ${i} paper_tables -----"
    snapshot_paper_tables "${i}"
done

echo ""
echo "=========================================================="
echo "  AGGREGATION ($(date))"
echo "=========================================================="
python scripts/diagnostic/aggregate_3runs.py \
    --runs_dir "${RUNS_TABLE_DIR}" \
    --n_runs "${N_RUNS}" \
    --prefixes "${PREFIXES[@]}" || {
    echo "[WARN] aggregation failed."
}

echo ""
echo "=================================================="
echo ">> All ${N_RUNS} runs finished : $(date)"
echo ">> Per-run + aggregated tables : ${RUNS_TABLE_DIR}/"
echo ">> Log                         : ${LOG_FILE}"
echo "=================================================="
