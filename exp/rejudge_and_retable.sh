#!/usr/bin/env bash
# ==============================================================================
# 把已有 results_diagnostic 跑过的 JSONL 重新做 xverify 判分（按规范化答案缓存），
# 并重新生成 6 张诊断表 + sample-level debug + sanity-check 报告。
#
# 不重跑生成阶段：只在 LLM judge 上动手。原 raw_response / extracted_answer / token
# 全部保留。
#
# 用法：
#   bash exp/rejudge_and_retable.sh results_diagnostic/run_20260501_113709
#
# 也可以指定单个模型目录：
#   bash exp/rejudge_and_retable.sh results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct
#
# 重新判分后的 JSONL 写到 <input_dir>_rejudged/，6 张表写到 <input_dir>_rejudged/_tables/。
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <results_diagnostic/run_XXXX[/<model_name>]>"
    exit 1
fi

INPUT_PATH="$1"
shift || true

if [[ ! -d "$INPUT_PATH" ]]; then
    echo "[ERROR] input directory not found: $INPUT_PATH"
    exit 1
fi

# 探测是 run 级别还是 model 级别：
# - 如果 INPUT_PATH 直接含 mad_vote_*_infer.jsonl，认为是单个模型目录
# - 否则枚举其中的子目录作为模型目录
MODEL_DIRS=()
shopt -s nullglob
direct_files=("${INPUT_PATH}"/mad_vote_*_infer.jsonl)
if [[ ${#direct_files[@]} -gt 0 ]]; then
    MODEL_DIRS=("$INPUT_PATH")
else
    for d in "${INPUT_PATH}"/*/; do
        [[ -d "$d" ]] || continue
        # 只保留含 jsonl 的目录
        if compgen -G "${d}/mad_vote_*_infer.jsonl" > /dev/null; then
            MODEL_DIRS+=("${d%/}")
        fi
    done
fi
shopt -u nullglob

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] no mad_vote_*_infer.jsonl found under $INPUT_PATH"
    exit 1
fi

echo ">> 共发现 ${#MODEL_DIRS[@]} 个模型目录待重判："
for d in "${MODEL_DIRS[@]}"; do echo "   - $d"; done
echo ""

for model_dir in "${MODEL_DIRS[@]}"; do
    out_dir="${model_dir}_rejudged"
    tables_dir="${out_dir}/_tables"
    echo "=================================================="
    echo ">> 重判: ${model_dir}"
    echo ">> 输出: ${out_dir}"
    echo "=================================================="

    python scripts/diagnostic/rejudge_diagnostic.py \
        --input_dir  "${model_dir}" \
        --output_dir "${out_dir}" \
        "$@"

    echo ""
    echo ">> 重新生成诊断表 -> ${tables_dir}"
    python scripts/diagnostic/analyze_diagnostic.py \
        --infer_dir  "${out_dir}" \
        --output_dir "${tables_dir}" \
        --strict || {
        echo "[FAIL] sanity checks failed for ${model_dir}; see logs above."
        exit 1
    }
    echo ""
done

echo "=================================================="
echo ">> 全部重判完成，每个模型目录旁边出现 _rejudged/ 同时含表与 sample-level debug。"
echo "=================================================="
