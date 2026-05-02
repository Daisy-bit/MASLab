#!/usr/bin/env bash
# ==============================================================================
# 一键运行 SCC v3 论文实验全流程：
#   阶段 A: SCC diagnostic (3 模型 × 5 数据集) -> results_diagnostic_scc/run_<TS>/
#   阶段 B: SCC ablation   (3B × 5 数据集 × 4 variants) -> results_ablation_scc/run_<TS>/
#   阶段 C: 聚合论文表格 -> paper_tables/
#       - regime_comparison.csv / .md  (Vanilla MAD vs SCC, 1.5B/3B/7B 共 6 行)
#       - ablation_3b.csv / .md        (Initial Vote / Vanilla MAD / 4 SCC 变体, 共 6 行)
#
# 默认会复用以下既有 mad_vote 诊断目录作为 Vanilla MAD 基线：
#   1.5B: results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged
#   3B:   results_diagnostic/run_20260501_130540/qwen25-3b-instruct
#   7B:   results_diagnostic/run_20260501_185633/qwen25-7b-instruct
#
# 用法（在 MASLab/ 项目根目录下执行）：
#   bash exp/run_scc_all.sh                        # 全流程
#   bash exp/run_scc_all.sh --max_samples 5        # 冒烟测试（约几分钟）
#   SKIP_DIAG=1 bash exp/run_scc_all.sh            # 跳过阶段 A
#   SKIP_ABLATION=1 bash exp/run_scc_all.sh        # 跳过阶段 B
#   SKIP_AGGREGATE=1 bash exp/run_scc_all.sh       # 跳过阶段 C
#   DIAG_RUN=results_diagnostic_scc/run_xxx \
#   ABLATION_RUN=results_ablation_scc/run_yyy \
#     SKIP_DIAG=1 SKIP_ABLATION=1 bash exp/run_scc_all.sh   # 仅聚合既有结果
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------
# 默认参数（可被环境变量覆盖）
# ------------------------------------------------------------------
SKIP_DIAG="${SKIP_DIAG:-0}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

# Vanilla MAD 基线目录（已有的 3 个 mad_vote 诊断 run）
MAD_15B="${MAD_15B:-results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged}"
MAD_3B="${MAD_3B:-results_diagnostic/run_20260501_130540/qwen25-3b-instruct}"
MAD_7B="${MAD_7B:-results_diagnostic/run_20260501_185633/qwen25-7b-instruct}"

# 透传给 inference.py 的命令行参数（如 --max_samples 5）
PASSTHROUGH_ARGS=("$@")

# 主日志（汇总三个阶段的 stdout/stderr）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ALL_LOG_DIR="logs"
mkdir -p "$ALL_LOG_DIR"
ALL_LOG="${ALL_LOG_DIR}/run_scc_all_${TIMESTAMP}.log"
exec > >(tee "$ALL_LOG") 2>&1

echo "=================================================================="
echo "  SCC v3 全流程一键脚本"
echo "  时间        : $(date)"
echo "  项目根目录  : ${PROJECT_ROOT}"
echo "  日志        : ${ALL_LOG}"
echo "  透传参数    : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "  SKIP_DIAG   : ${SKIP_DIAG}"
echo "  SKIP_ABLATION: ${SKIP_ABLATION}"
echo "  SKIP_AGGREGATE: ${SKIP_AGGREGATE}"
echo "=================================================================="

# ------------------------------------------------------------------
# 前置检查：mad_vote 基线目录是否齐全
# ------------------------------------------------------------------
echo ""
echo ">> 前置检查：mad_vote 基线目录"
for d in "$MAD_15B" "$MAD_3B" "$MAD_7B"; do
    t2="${d}/_tables/table2_regime_analysis.csv"
    t6="${d}/_tables/table6_coverage_survival.csv"
    if [[ -f "$t2" && -f "$t6" ]]; then
        echo "   OK: $d"
    else
        echo "   [WARN] 缺少表格文件: $d/_tables/{table2,table6}_*.csv"
        echo "          阶段 C 的 regime_comparison 行可能为 N/A。"
    fi
done

# ------------------------------------------------------------------
# 阶段 A：SCC diagnostic (3 模型 × 5 数据集)
# ------------------------------------------------------------------
DIAG_RUN="${DIAG_RUN:-}"
if [[ "$SKIP_DIAG" != "1" ]]; then
    echo ""
    echo "=================================================================="
    echo "  阶段 A: SCC diagnostic 推理 + 分析"
    echo "=================================================================="

    # 记录阶段开始前的目录列表，跑完后取差集就能拿到本次新建的 run_<TS> 目录
    BEFORE_DIAG=$(ls -1 results_diagnostic_scc/ 2>/dev/null | sort || true)

    bash exp/run_scc_diagnostic.sh "${PASSTHROUGH_ARGS[@]}" || {
        echo "[ERR] 阶段 A 失败，终止。"
        exit 1
    }

    AFTER_DIAG=$(ls -1 results_diagnostic_scc/ 2>/dev/null | sort || true)
    NEW_DIAG=$(comm -13 <(echo "$BEFORE_DIAG") <(echo "$AFTER_DIAG") | grep '^run_' | tail -n 1 || true)
    if [[ -n "$NEW_DIAG" ]]; then
        DIAG_RUN="results_diagnostic_scc/${NEW_DIAG}"
    else
        # 兜底：取最新的 run_*
        DIAG_RUN=$(ls -1dt results_diagnostic_scc/run_* 2>/dev/null | head -n 1 || true)
    fi
    echo ">> 阶段 A 输出目录: ${DIAG_RUN}"
else
    echo ""
    echo ">> SKIP_DIAG=1，跳过阶段 A。"
    if [[ -z "$DIAG_RUN" ]]; then
        DIAG_RUN=$(ls -1dt results_diagnostic_scc/run_* 2>/dev/null | head -n 1 || true)
        echo ">> 自动选用最新 SCC diagnostic run: ${DIAG_RUN:-(无)}"
    fi
fi

# ------------------------------------------------------------------
# 阶段 B：SCC ablation (3B × 5 数据集 × 4 variants)
# ------------------------------------------------------------------
ABLATION_RUN="${ABLATION_RUN:-}"
if [[ "$SKIP_ABLATION" != "1" ]]; then
    echo ""
    echo "=================================================================="
    echo "  阶段 B: SCC ablation 推理 + 分析"
    echo "=================================================================="

    BEFORE_ABL=$(ls -1 results_ablation_scc/ 2>/dev/null | sort || true)

    bash exp/run_scc_ablation.sh "${PASSTHROUGH_ARGS[@]}" || {
        echo "[ERR] 阶段 B 失败，终止。"
        exit 1
    }

    AFTER_ABL=$(ls -1 results_ablation_scc/ 2>/dev/null | sort || true)
    NEW_ABL=$(comm -13 <(echo "$BEFORE_ABL") <(echo "$AFTER_ABL") | grep '^run_' | tail -n 1 || true)
    if [[ -n "$NEW_ABL" ]]; then
        ABLATION_RUN="results_ablation_scc/${NEW_ABL}"
    else
        ABLATION_RUN=$(ls -1dt results_ablation_scc/run_* 2>/dev/null | head -n 1 || true)
    fi
    echo ">> 阶段 B 输出目录: ${ABLATION_RUN}"
else
    echo ""
    echo ">> SKIP_ABLATION=1，跳过阶段 B。"
    if [[ -z "$ABLATION_RUN" ]]; then
        ABLATION_RUN=$(ls -1dt results_ablation_scc/run_* 2>/dev/null | head -n 1 || true)
        echo ">> 自动选用最新 SCC ablation run: ${ABLATION_RUN:-(无)}"
    fi
fi

# ------------------------------------------------------------------
# 阶段 C：聚合论文表格
# ------------------------------------------------------------------
if [[ "$SKIP_AGGREGATE" != "1" ]]; then
    echo ""
    echo "=================================================================="
    echo "  阶段 C: 聚合论文表格 -> paper_tables/"
    echo "=================================================================="
    mkdir -p paper_tables

    # ---- 表 1: regime_comparison ----
    if [[ -z "$DIAG_RUN" ]]; then
        echo "[WARN] 未找到 SCC diagnostic 目录，跳过 regime_comparison。"
    else
        # 在 DIAG_RUN 下找 1.5B/3B/7B 三个 model 子目录（容错：可能只有部分模型）
        SCC_15B="${DIAG_RUN}/qwen25-1.5b-instruct"
        SCC_3B="${DIAG_RUN}/qwen25-3b-instruct"
        SCC_7B="${DIAG_RUN}/qwen25-7b-instruct"

        MAD_PAIRS=()
        SCC_PAIRS=()
        for pair in "1.5B:${MAD_15B}:${SCC_15B}" \
                    "3B:${MAD_3B}:${SCC_3B}" \
                    "7B:${MAD_7B}:${SCC_7B}"; do
            label="${pair%%:*}"
            rest="${pair#*:}"
            mad_dir="${rest%%:*}"
            scc_dir="${rest##*:}"
            if [[ -d "$mad_dir" && -d "$scc_dir" ]]; then
                MAD_PAIRS+=("${label}:${mad_dir}")
                SCC_PAIRS+=("${label}:${scc_dir}")
            else
                echo "   [WARN] 跳过 ${label}（mad_dir=${mad_dir} 或 scc_dir=${scc_dir} 不存在）"
            fi
        done

        if [[ ${#MAD_PAIRS[@]} -gt 0 ]]; then
            MAD_ARG=$(IFS=, ; echo "${MAD_PAIRS[*]}")
            SCC_ARG=$(IFS=, ; echo "${SCC_PAIRS[*]}")
            echo ""
            echo ">> 生成 regime_comparison ..."
            python scripts/diagnostic/build_regime_comparison.py \
                --mad_runs "$MAD_ARG" \
                --scc_runs "$SCC_ARG" \
                --output_csv paper_tables/regime_comparison.csv \
                --output_md  paper_tables/regime_comparison.md || {
                echo "[WARN] regime_comparison 生成失败"
            }
        else
            echo "[WARN] 没有可用的 (mad, scc) 模型对，跳过 regime_comparison。"
        fi
    fi

    # ---- 表 2: ablation_3b ----
    if [[ -z "$ABLATION_RUN" ]]; then
        echo "[WARN] 未找到 SCC ablation 目录，跳过 ablation_3b。"
    elif [[ ! -d "$MAD_3B" ]]; then
        echo "[WARN] mad_vote 3B 目录缺失（${MAD_3B}），跳过 ablation_3b。"
    else
        echo ""
        echo ">> 生成 ablation_3b ..."
        python scripts/diagnostic/build_ablation_table.py \
            --mad_run "$MAD_3B" \
            --scc_run "$ABLATION_RUN" \
            --output_csv paper_tables/ablation_3b.csv \
            --output_md  paper_tables/ablation_3b.md || {
            echo "[WARN] ablation_3b 生成失败"
        }
    fi
else
    echo ""
    echo ">> SKIP_AGGREGATE=1，跳过阶段 C。"
fi

# ------------------------------------------------------------------
# 收尾汇总
# ------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  全流程完成: $(date)"
echo "  Diagnostic run : ${DIAG_RUN:-(skipped)}"
echo "  Ablation run   : ${ABLATION_RUN:-(skipped)}"
echo "  论文表格        : paper_tables/regime_comparison.{csv,md}"
echo "                    paper_tables/ablation_3b.{csv,md}"
echo "  主日志          : ${ALL_LOG}"
echo "=================================================================="
