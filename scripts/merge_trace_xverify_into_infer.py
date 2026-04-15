#!/usr/bin/env python3
"""将 eval_trace_xverify.py 产出的 trace_xverify 按 query 合并进 infer jsonl，供 visualize_soo_math_trace 一次读取。"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_jsonl", type=Path, required=True)
    ap.add_argument("--trace_xverify_jsonl", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    args = ap.parse_args()

    trace_by_query: dict = {}
    for line in args.trace_xverify_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "trace_xverify" in o:
            trace_by_query[o["query"]] = o["trace_xverify"]

    rows_out = []
    for line in args.infer_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        q = o["query"]
        if q in trace_by_query:
            o["trace_xverify"] = trace_by_query[q]
        rows_out.append(o)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for o in rows_out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("Wrote", args.out_jsonl, "lines", len(rows_out), "| trace_xverify merged:", len(trace_by_query))


if __name__ == "__main__":
    main()
