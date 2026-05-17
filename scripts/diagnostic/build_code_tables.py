"""Aggregate per-variant code eval_score JSONLs into a pass@1 CSV+MD table.

Code datasets (HumanEval, MBPP-500) don't have the xverify-style diagnostic
schema that build_madvote_scc_tables.py reads. They produce
`<method>_<dataset>_xverify_eval.jsonl` (filename inherited from
evaluate.py) with one record per sample carrying `eval_score` (1=pass,
0=fail, None=eval-error). This script reads those eval files for each
(variant, dataset) and emits:

  paper_tables/<prefix>_<model_label>.csv
  paper_tables/<prefix>_<model_label>.md

Variant rows follow the 5-cell code ablation matrix (A5 dropped — see
methods/<method>/configs/config_code_a5_trig_rout.yaml header notes).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Ordering matches the user's 5-cell code matrix:
# A0 baseline → A3 (+ argmax aggr) → A1 (+ spectral trig) →
# A2 (+ contribution routing) → A4 (full SCC).
VARIANT_ORDER = [
    "A0_vanilla",
    "A3_aggregation",
    "A1_triggering",
    "A2_routing",
    "A4_all",
]


def _read_pass_rate(eval_file: Path) -> tuple[int, int, int]:
    """Return (passed, total_valid, total_records).

    total_valid excludes eval-error rows (eval_score is None).
    """
    if not eval_file.exists():
        return 0, 0, 0
    passed = 0
    total_valid = 0
    total = 0
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            score = rec.get("eval_score")
            if score is None:
                continue
            total_valid += 1
            if score == 1:
                passed += 1
    return passed, total_valid, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path containing <variant>/ subdirs")
    ap.add_argument("--method_label", required=True, help="soo_scc | mad_scc")
    ap.add_argument("--model_label", required=True, help="e.g. 1.5b")
    ap.add_argument("--datasets", nargs="+", required=True, help="HumanEval MBPP-500 ...")
    ap.add_argument("--out_dir", default="paper_tables")
    ap.add_argument("--prefix", default="scc_code")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows: dict[str, dict[str, tuple[int, int, int]]] = {}
    for variant in VARIANT_ORDER:
        var_dir = run_dir / variant
        if not var_dir.is_dir():
            continue
        rows[variant] = {}
        for ds in args.datasets:
            eval_file = var_dir / f"{args.method_label}_{ds}_xverify_eval.jsonl"
            rows[variant][ds] = _read_pass_rate(eval_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.prefix}_{args.model_label}.csv"
    md_path = out_dir / f"{args.prefix}_{args.model_label}.md"

    # CSV — single header, one row per variant, two cells per dataset
    # (pass@1 percent + "passed/total_valid" annotation).
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["variant"]
        for ds in args.datasets:
            header.extend([f"{ds}_pass@1(%)", f"{ds}_n"])
        w.writerow(header)
        for variant in VARIANT_ORDER:
            if variant not in rows:
                continue
            row = [variant]
            for ds in args.datasets:
                passed, total_valid, _total = rows[variant].get(ds, (0, 0, 0))
                pct = (passed / total_valid * 100.0) if total_valid else 0.0
                row.extend([f"{pct:.2f}", f"{passed}/{total_valid}"])
            w.writerow(row)

    # Markdown — single combined "X.X% (P/T)" cell per dataset.
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {args.method_label} code pass@1 ({args.model_label})\n\n")
        f.write("| variant | " + " | ".join(args.datasets) + " |\n")
        f.write("|" + "---|" * (len(args.datasets) + 1) + "\n")
        for variant in VARIANT_ORDER:
            if variant not in rows:
                continue
            cells = [variant]
            for ds in args.datasets:
                passed, total_valid, _ = rows[variant].get(ds, (0, 0, 0))
                pct = (passed / total_valid * 100.0) if total_valid else 0.0
                cells.append(f"{pct:.2f}% ({passed}/{total_valid})")
            f.write("| " + " | ".join(cells) + " |\n")

    print(f">> wrote {csv_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
