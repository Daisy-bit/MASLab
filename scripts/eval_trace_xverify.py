#!/usr/bin/env python3
"""
对 soo_math 等写入的 inference_trace 各阶段调用 xverify（与 evaluate.py / evaluations.evaluate_xverify 相同协议）。

推理阶段不在 trace 里判对错；本脚本读取 jsonl 中每条样本的 agent 全文、plurality 结果等，逐条请求评判模型，
将结果写入 trace_xverify 字段（不修改 inference_trace 结构）。

用法:
  python scripts/eval_trace_xverify.py \\
    --infer_jsonl results/GSM-Hard/qwen25-3b-instruct_xxx/soo_math_infer.jsonl \\
    --output_jsonl results/GSM-Hard/qwen25-3b-instruct_xxx/soo_math_trace_xverify.jsonl

与 infer 合并后可视化:
  python scripts/merge_trace_xverify_into_infer.py \\
    --infer_jsonl .../soo_math_infer.jsonl \\
    --trace_xverify_jsonl .../soo_math_trace_xverify.jsonl \\
    --out_jsonl .../soo_math_merged.jsonl
  python scripts/visualize_soo_math_trace.py --jsonl .../soo_math_merged.jsonl --out-prefix ...
"""

import argparse
import json
import threading
import traceback
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from evaluations import get_eval_func
from methods import get_method_class
from utils import load_model_api_config, read_valid_jsonl, reserve_unprocessed_queries, write_to_jsonl


def _eval_one(
    eval_func,
    llm,
    query: str,
    gt: Any,
    response_text: Optional[str],
) -> Tuple[str, Optional[int]]:
    if gt is None or str(gt).strip() == "":
        return "Eval Error: missing gt", None
    text = response_text if response_text is not None else ""
    if not str(text).strip():
        return "Eval Error: empty response", None
    return eval_func({"query": query, "response": text, "gt": gt}, llm)


def _agent_texts_for_stage(stage: Dict[str, Any]) -> List[str]:
    texts = stage.get("agent_responses")
    if texts is not None:
        return [t if t is not None else "" for t in texts]
    return [x if x is not None else "" for x in (stage.get("extracted_answers") or [])]


def eval_trace_stages(
    eval_func,
    llm,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """对单条样本的 inference_trace 各阶段做 xverify，并评判最终 row['response']。"""
    query = row.get("query", "")
    gt = row.get("gt") or row.get("answer")
    tr = row.get("inference_trace") or {}
    stages_in = tr.get("stages") or []

    stages_out: List[Dict[str, Any]] = []
    for stage in stages_in:
        texts = _agent_texts_for_stage(stage)
        per_agent: List[Dict[str, Any]] = []
        for i, resp in enumerate(texts):
            ec, es = _eval_one(eval_func, llm, query, gt, resp)
            per_agent.append({"agent_idx": i, "eval_content": ec, "eval_score": es})

        pu = stage.get("plurality_uniform_answer") or ""
        plurality_uniform = None
        if str(pu).strip():
            ec, es = _eval_one(eval_func, llm, query, gt, pu)
            plurality_uniform = {"answer": pu, "eval_content": ec, "eval_score": es}

        pw = stage.get("plurality_weighted_answer")
        plurality_weighted = None
        if pw is not None and str(pw).strip():
            ec, es = _eval_one(eval_func, llm, query, gt, pw)
            plurality_weighted = {"answer": pw, "eval_content": ec, "eval_score": es}

        stages_out.append(
            {
                "label": stage.get("label"),
                "completed_debate_rounds": stage.get("completed_debate_rounds"),
                "per_agent": per_agent,
                "plurality_uniform": plurality_uniform,
                "plurality_weighted": plurality_weighted,
                "final_idx": stage.get("final_idx"),
                "final_idx_extracted": stage.get("final_idx_extracted"),
                "anchor_indices": stage.get("anchor_indices"),
            }
        )

    final_ec, final_es = _eval_one(eval_func, llm, query, gt, row.get("response"))

    return {
        "eval_protocol": "xverify",
        "stages": stages_out,
        "final_output": {
            "eval_content": final_ec,
            "eval_score": final_es,
        },
    }


def process_row(
    args,
    row: Dict[str, Any],
    llm,
    eval_func,
) -> Dict[str, Any]:
    out = row.copy()
    tr = row.get("inference_trace") or {}
    try:
        if not tr.get("stages"):
            out["trace_xverify"] = {
                "eval_protocol": "xverify",
                "skipped": True,
                "reason": "no inference_trace.stages",
            }
            return out
        out["trace_xverify"] = eval_trace_stages(eval_func, llm, row)
    except Exception:
        out["trace_xverify"] = {
            "eval_protocol": "xverify",
            "error": traceback.format_exc(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="对 inference_trace 各阶段调用 xverify")
    parser.add_argument("--infer_jsonl", type=str, required=True, help="含 inference_trace 的推理 jsonl")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="输出路径；默认在 infer 文件名中插入 _trace_xverify",
    )
    parser.add_argument("--eval_protocol", type=str, default="xverify")
    parser.add_argument("--tested_dataset_name", type=str, default="GSM-Hard")
    parser.add_argument("--model_name", type=str, default="xverify-9b-c")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5)
    parser.add_argument("--model_max_tokens", type=int, default=2048)
    parser.add_argument("--model_timeout", type=int, default=600)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = args.output_jsonl
    if out_path is None:
        if "_infer.jsonl" in args.infer_jsonl:
            out_path = args.infer_jsonl.replace("_infer.jsonl", "_trace_xverify.jsonl")
        else:
            out_path = args.infer_jsonl.replace(".jsonl", "_trace_xverify.jsonl")

    general_config = {
        "model_name": args.model_name,
        "model_api_config": load_model_api_config(args.model_api_config, args.model_name),
        "model_temperature": args.model_temperature,
        "model_max_tokens": args.model_max_tokens,
        "model_timeout": args.model_timeout,
    }
    eval_func = get_eval_func(args.eval_protocol, args.tested_dataset_name)
    llm = get_method_class("vanilla")(general_config)

    data = read_valid_jsonl(args.infer_jsonl)
    print(f">> Loaded {len(data)} lines from {args.infer_jsonl}")

    if args.overwrite:
        import os

        if os.path.exists(out_path):
            os.remove(out_path)
            print(f">> Removed existing {out_path}")
    else:
        data = reserve_unprocessed_queries(out_path, data)
        print(f">> After reserve_unprocessed: {len(data)} samples")

    if not data:
        print(">> Nothing to do.")
        return

    lock = threading.Lock()
    max_workers = general_config["model_api_config"][args.model_name]["max_workers"]

    def _job(row: Dict[str, Any]) -> None:
        save = process_row(args, row, llm, eval_func)
        write_to_jsonl(lock, out_path, save)

    if args.sequential:
        for row in tqdm(data, desc="trace_xverify"):
            _job(row)
    else:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(
                tqdm(
                    ex.map(_job, data),
                    total=len(data),
                    desc="trace_xverify",
                )
            )

    print(f">> Wrote {out_path}")


if __name__ == "__main__":
    main()
