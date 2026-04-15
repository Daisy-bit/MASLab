"""
Parse ablation evaluation results and print a comparison table.

Usage:
    python exp/summarize.py --results_dir results_ablation/<TIMESTAMP>
    python exp/summarize.py --results_dir results_ablation/<TIMESTAMP> --output_csv exp/ablation_results.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def read_eval_jsonl(path):
    """Read evaluation JSONL, return (total, correct, accuracy)."""
    if not os.path.exists(path):
        return 0, 0, None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        return 0, 0, None
    valid = [r for r in records if r.get("eval_score") is not None]
    correct = sum(1 for r in valid if r["eval_score"] == 1)
    total = len(valid)
    acc = correct / total * 100 if total > 0 else 0.0
    return total, correct, acc


def main():
    parser = argparse.ArgumentParser(description="Summarize ablation experiment results.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the timestamped results directory.")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional: save results as CSV.")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    # Discover all eval files: <method>_xverify_eval.jsonl
    eval_files = [f for f in os.listdir(results_dir) if f.endswith("_xverify_eval.jsonl")]
    if not eval_files:
        print(f"[WARNING] No evaluation files found in {results_dir}")
        sys.exit(0)

    # Parse method names from filenames
    rows = []
    for fname in sorted(eval_files):
        method = fname.replace("_xverify_eval.jsonl", "")
        path = os.path.join(results_dir, fname)
        total, correct, acc = read_eval_jsonl(path)
        rows.append((method, total, correct, acc))

    # Print table
    print()
    print("=" * 70)
    print(f"  Ablation Experiment Results")
    print(f"  Directory: {results_dir}")
    print("=" * 70)
    print(f"{'Method':<30} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 70)
    for method, total, correct, acc in rows:
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        print(f"{method:<30} {total:>8} {correct:>8} {acc_str:>10}")
    print("=" * 70)

    # Ablation analysis
    result_map = {method: acc for method, _, _, acc in rows}
    print()
    selforg_acc = result_map.get("selforg")
    no_debate_acc = result_map.get("selforg_no_debate")
    random_graph_acc = result_map.get("selforg_random_graph")
    soo_acc = result_map.get("soo")

    if selforg_acc is not None and no_debate_acc is not None:
        delta = selforg_acc - no_debate_acc
        sign = "+" if delta >= 0 else ""
        print(f"  [Ablation 1] Debate effectiveness:        SelfOrg - NoDebate = {sign}{delta:.2f}%")
    if selforg_acc is not None and random_graph_acc is not None:
        delta = selforg_acc - random_graph_acc
        sign = "+" if delta >= 0 else ""
        print(f"  [Ablation 2] Orchestration effectiveness:  SelfOrg - RandomGraph = {sign}{delta:.2f}%")
    if selforg_acc is not None and soo_acc is not None:
        delta = soo_acc - selforg_acc
        sign = "+" if delta >= 0 else ""
        print(f"  [Ablation 3] SOO vs SelfOrg:               SOO - SelfOrg = {sign}{delta:.2f}%")

    soo_centered_acc = result_map.get("soo_centered")
    soo_centered_v2_acc = result_map.get("soo_centered_v2")

    if soo_acc is not None and soo_centered_acc is not None:
        delta = soo_centered_acc - soo_acc
        sign = "+" if delta >= 0 else ""
        print(f"  [Ablation 4] SOO-Centered vs SOO:          Centered - SOO = {sign}{delta:.2f}%")
    if soo_centered_acc is not None and soo_centered_v2_acc is not None:
        delta = soo_centered_v2_acc - soo_centered_acc
        sign = "+" if delta >= 0 else ""
        print(f"  [Ablation 5] SOO-Centered-v2 vs v1:        v2 - v1 = {sign}{delta:.2f}%")
    print()

    # Optional CSV output
    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write("method,total,correct,accuracy\n")
            for method, total, correct, acc in rows:
                acc_str = f"{acc:.2f}" if acc is not None else ""
                f.write(f"{method},{total},{correct},{acc_str}\n")
        print(f"  Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
