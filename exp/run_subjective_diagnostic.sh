#!/usr/bin/env bash
# ==============================================================================
# 主观点(诊断)实验：vanilla MAD + plurality voting，按 exp/实验流程.md 执行
#
# 流程：
#   1) 对 N=5 个 agent 让其独立作答 (round 0)；
#   2) 进行 3 轮全连接 vanilla MAD，每轮记录每个 agent 回答；
#   3) 对每一轮的抽取答案做 plurality vote，得到 V(0)..V(L)；
#   4) 计算 oracle coverage、bucket(already_solved/recoverable/unrecoverable)
#      并保存 JSONL；
#   5) 跑分析脚本生成 Table 1~6 CSV。
#
# 默认 5 数据集 × 3 个模型 (1.5b/3b/7b)，可通过环境变量覆盖。
#
# 用法（在 MASLab/ 项目根目录下执行）：
#   bash exp/run_subjective_diagnostic.sh
#   bash exp/run_subjective_diagnostic.sh --max_samples 50
#   DATASETS="GSM8K AIME-2024" MODELS="qwen25-3b-instruct" \
#     bash exp/run_subjective_diagnostic.sh
#   ANALYZE_ONLY=1 bash exp/run_subjective_diagnostic.sh
#   ANALYZE_ONLY=1 ANALYZE_DIR=./results_diagnostic/run_20260101_120000 \
#     bash exp/run_subjective_diagnostic.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# 抑制 huggingface tokenizers 在 fork 多进程后的重复警告
export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------------
# 默认参数 (可被环境变量覆盖)
# ------------------------------------------------------------------
DATASETS_DEFAULT="GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro"
MODELS_DEFAULT="qwen25-1.5b-instruct qwen25-3b-instruct qwen25-7b-instruct"
METHOD_NAME="mad_vote"
METHOD_CONFIG_NAME="config_main"
OUTPUT_ROOT_DEFAULT="./results_diagnostic"

DATASETS="${DATASETS:-$DATASETS_DEFAULT}"
MODELS="${MODELS:-$MODELS_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
# 阶段 2 分析根目录：未设置则用本次 RUN_ROOT；设置后阶段 2 会改为分析该目录
ANALYZE_DIR="${ANALYZE_DIR:-}"

# 把命令行参数原样透传给 inference.py（如 --max_samples / --sequential 等）
PASSTHROUGH_ARGS=("$@")

if [[ -n "$ANALYZE_DIR" && "$ANALYZE_ONLY" == "1" ]]; then
    # 仅分析既有目录：复用其作为 RUN_ROOT，不再新建带时间戳的目录
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

# 同时输出到日志文件，避免长跑丢日志
LOG_FILE="${RUN_ROOT}/pipeline.log"
exec > >(tee "$LOG_FILE") 2>&1

echo ">> Run directory : ${RUN_ROOT}"
echo ">> Method        : ${METHOD_NAME} (${METHOD_CONFIG_NAME})"
echo ">> Datasets      : ${DATASETS}"
echo ">> Models        : ${MODELS}"
echo ">> Passthrough   : ${PASSTHROUGH_ARGS[*]:-(none)}"
echo "=================================================="

# ------------------------------------------------------------------
# 阶段 1：推理 (per model x per dataset)
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
                echo "[WARN] dataset file not found: ${DATA_FILE} — skipping ${ds} for ${model}"
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
                echo "[WARN] inference failed for ${model} / ${ds} — continuing"
            }
        done
    done
    echo ""
    echo ">> Inference phase finished."
else
    echo ">> ANALYZE_ONLY=1 — skipping inference."
fi

# ------------------------------------------------------------------
# 阶段 2：分析 (per model)
# ------------------------------------------------------------------
echo ""
echo "================= Analysis ====================="
for model in $MODELS; do
    MODEL_DIR="${RUN_ROOT}/${model}"
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "[WARN] no inference output for ${model} — skipping analysis"
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
echo ">> Diagnostic experiment finished."
echo ">> Run dir : ${RUN_ROOT}"
echo ">> Tables  : ${RUN_ROOT}/<model>/_tables/table1..6_*.csv"
echo "=================================================="
