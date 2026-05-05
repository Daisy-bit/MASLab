#!/usr/bin/env bash
# ==============================================================================
# Run both SCC ablation pipelines sequentially.
#
#   1) exp/run_mad_scc_1.5b.sh   — mad_vote baseline + scc_components
#   2) exp/run_soo_scc_1.5b.sh   — soo_centered_v3 baseline + scc_components
#
# Each pipeline runs A0-A5 x 5 datasets, replays round 0 from the shared
# initial pool under results_archive/, and writes its own paper_tables/
# entries (mad_scc_A{1,2}_1.5b and soo_scc_A{1,2}_1.5b).
#
# Usage (from MASLab/ project root):
#   bash exp/run_both_scc_1.5b.sh
#   bash exp/run_both_scc_1.5b.sh --max_samples 5      # smoke test (passed thru)
#   STOP_ON_FAIL=1 bash exp/run_both_scc_1.5b.sh        # abort if either fails
#
# Default: continues to the second pipeline even if the first fails, so a
# partial run still produces some tables.
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
PASSTHROUGH_ARGS=("$@")

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/run_both_scc_1.5b_${TIMESTAMP}.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=================================================="
echo "  Both SCC pipelines — sequential"
echo "  Time      : $(date)"
echo "  Log       : ${LOG_FILE}"
echo "  Pass-thru : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "  Stop-on-fail : ${STOP_ON_FAIL}"
echo "=================================================="

run_step() {
    local name="$1"; shift
    echo ""
    echo "=========================================================="
    echo ">> [$(date)] starting ${name}"
    echo "=========================================================="
    if bash "$@" "${PASSTHROUGH_ARGS[@]}"; then
        echo ""
        echo ">> [$(date)] ${name} finished OK"
        return 0
    else
        echo ""
        echo "[ERR] ${name} failed (exit $?)"
        if [[ "$STOP_ON_FAIL" == "1" ]]; then
            echo "[ERR] STOP_ON_FAIL=1 — aborting."
            exit 1
        fi
        echo "[WARN] continuing to next pipeline."
        return 1
    fi
}

run_step "mad_scc 1.5B" "${SCRIPT_DIR}/run_mad_scc_1.5b.sh"
run_step "soo_scc 1.5B" "${SCRIPT_DIR}/run_soo_scc_1.5b.sh"

echo ""
echo "=================================================="
echo ">> All pipelines done: $(date)"
echo ">> Log    : ${LOG_FILE}"
echo ">> Tables : paper_tables/mad_scc_A{1,2}_1.5b.{csv,md}"
echo ">>          paper_tables/soo_scc_A{1,2}_1.5b.{csv,md}"
echo "=================================================="
