"""
Compare soo_centered_v3 results against a baseline ablation run.

Reads a v3 results directory (created by exp/run_v3.sh) and a baseline results
directory (any prior results_ablation/<TIMESTAMP>), then prints per-dataset
ranking tables and a cross-dataset summary with v3 highlighted.

Usage:
    python exp/compare_v3.py --v3_dir results_ablation/v3_quick_<TS> \\
                             --baseline_dir results_ablation/20260416_111010
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def read_eval_jsonl(path: str) -> Tuple[int, int, Optional[float]]:
    if not os.path.exists(path):
        return 0, 0, None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    valid = [r for r in records if r.get("eval_score") is not None]
    correct = sum(1 for r in valid if r["eval_score"] == 1)
    total = len(valid)
    acc = correct / total * 100 if total > 0 else None
    return total, correct, acc


def collect_results(base_dir: str, datasets: List[str]) -> Dict[str, Dict[str, Tuple[int, int, Optional[float]]]]:
    """Return {dataset: {method: (total, correct, accuracy)}}."""
    out: Dict[str, Dict[str, Tuple[int, int, Optional[float]]]] = {}
    for ds in datasets:
        ds_dir = os.path.join(base_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        per_method: Dict[str, Tuple[int, int, Optional[float]]] = {}
        for fname in os.listdir(ds_dir):
            if not fname.endswith("_xverify_eval.jsonl"):
                continue
            method = fname.replace("_xverify_eval.jsonl", "")
            per_method[method] = read_eval_jsonl(os.path.join(ds_dir, fname))
        if per_method:
            out[ds] = per_method
    return out


def discover_datasets(dirs: List[str]) -> List[str]:
    seen: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for entry in sorted(os.listdir(d)):
            full = os.path.join(d, entry)
            if os.path.isdir(full) and entry not in seen:
                seen.append(entry)
    return seen


def print_dataset_table(
    dataset: str,
    v3_acc: Optional[float],
    v3_total: int,
    baseline: Dict[str, Tuple[int, int, Optional[float]]],
) -> None:
    print("=" * 79)
    print(f"  Dataset: {dataset}   (samples tested by v3: {v3_total})")
    print("=" * 79)

    rows: List[Tuple[str, int, Optional[float], Optional[float]]] = []
    for method, (total, _correct, acc) in baseline.items():
        if method == "soo_centered_v3":
            continue
        delta = (acc - v3_acc) if (acc is not None and v3_acc is not None) else None
        rows.append((method, total, acc, delta))

    rows.sort(key=lambda r: (-(r[2] if r[2] is not None else -1.0), r[0]))

    print(f"  {'Method':<30} {'Samples':>8} {'Accuracy':>12} {'vs v3':>10}")
    print("-" * 79)
    for method, total, acc, delta in rows:
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        if delta is None:
            delta_str = "-"
        else:
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}"
        print(f"  {method:<30} {total:>8} {acc_str:>12} {delta_str:>10}")

    print("-" * 79)
    v3_acc_str = f"{v3_acc:.2f}%" if v3_acc is not None else "N/A"
    print(f"  {'soo_centered_v3 (this run)':<30} {v3_total:>8} {v3_acc_str:>12} {'-':>10}")
    print()


def print_cross_dataset_summary(
    datasets: List[str],
    v3_results: Dict[str, Tuple[int, int, Optional[float]]],
    baseline_results: Dict[str, Dict[str, Tuple[int, int, Optional[float]]]],
) -> None:
    # Collect all method names (baseline + v3)
    methods: List[str] = []
    for ds in datasets:
        for m in baseline_results.get(ds, {}):
            if m != "soo_centered_v3" and m not in methods:
                methods.append(m)

    # Deterministic ordering: core then dylan/soo variants alphabetically
    core_order = [
        "selforg",
        "selforg_no_debate",
        "selforg_random_graph",
        "soo",
        "soo_centered",
        "soo_centered_v2",
    ]
    ordered = [m for m in core_order if m in methods] + sorted(m for m in methods if m not in core_order)

    print("=" * 79)
    print("  Cross-dataset summary (accuracy %)")
    print("=" * 79)

    def fmt_cell(acc: Optional[float]) -> str:
        return f"{acc:>{col_w}.2f}" if acc is not None else f"{'N/A':>{col_w}}"

    col_w = 11
    header_cells = [f"{ds[:col_w]:>{col_w}}" for ds in datasets]
    header = f"{'Method':<30}" + "".join(header_cells) + f"{'Avg':>{col_w}}"
    print(header)
    print("-" * len(header))

    def row_for(label: str, per_ds: Dict[str, Tuple[int, int, Optional[float]]]) -> str:
        accs: List[float] = []
        cells: List[str] = []
        for ds in datasets:
            entry = per_ds.get(ds)
            acc = entry[2] if entry else None
            cells.append(f"{fmt_cell(acc):>9}")
            if acc is not None:
                accs.append(acc)
        avg = sum(accs) / len(accs) if accs else None
        return f"{label:<30}" + "".join(cells) + f"{fmt_cell(avg):>9}"

    for method in ordered:
        # Flatten baseline into {ds: (total, correct, acc)}
        per_ds = {ds: baseline_results.get(ds, {}).get(method) for ds in datasets}
        per_ds = {k: v for k, v in per_ds.items() if v is not None}
        print(row_for(method, per_ds))

    print("-" * len(header))
    print(row_for("soo_centered_v3 (ours)", v3_results))
    print("=" * len(header))
    print()
    print(f"  cells show accuracy in percent. lower is worse; diff columns show v3 gap.")


def rank_summary(
    datasets: List[str],
    v3_results: Dict[str, Tuple[int, int, Optional[float]]],
    baseline_results: Dict[str, Dict[str, Tuple[int, int, Optional[float]]]],
) -> None:
    print()
    print("=" * 79)
    print("  v3 ranking per dataset")
    print("=" * 79)
    for ds in datasets:
        v3_entry = v3_results.get(ds)
        v3_acc = v3_entry[2] if v3_entry else None
        if v3_acc is None:
            print(f"  {ds:<16} v3 not evaluated")
            continue

        all_accs = []
        for method, (_t, _c, acc) in baseline_results.get(ds, {}).items():
            if method == "soo_centered_v3" or acc is None:
                continue
            all_accs.append((method, acc))
        all_accs.append(("soo_centered_v3", v3_acc))
        all_accs.sort(key=lambda x: (-x[1], x[0]))
        rank = [i for i, (m, _) in enumerate(all_accs, 1) if m == "soo_centered_v3"][0]
        leader = all_accs[0]
        gap = v3_acc - leader[1] if leader[0] != "soo_centered_v3" else 0.0
        sign = "+" if gap >= 0 else ""
        print(
            f"  {ds:<16} v3 rank {rank}/{len(all_accs)}  "
            f"acc={v3_acc:.2f}%  leader={leader[0]} ({leader[1]:.2f}%)  "
            f"v3-leader={sign}{gap:.2f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v3_dir", required=True, help="results dir containing v3 eval files")
    p.add_argument("--baseline_dir", required=True, help="prior ablation results dir to compare against")
    p.add_argument("--datasets", default=None, help="space-separated; default = union auto-discovered from both dirs")
    args = p.parse_args()

    if not os.path.isdir(args.v3_dir):
        print(f"[ERROR] v3_dir not found: {args.v3_dir}")
        sys.exit(1)
    if not os.path.isdir(args.baseline_dir):
        print(f"[ERROR] baseline_dir not found: {args.baseline_dir}")
        sys.exit(1)

    datasets: List[str]
    if args.datasets:
        datasets = args.datasets.split()
    else:
        datasets = discover_datasets([args.v3_dir, args.baseline_dir])

    baseline = collect_results(args.baseline_dir, datasets)
    v3_raw = collect_results(args.v3_dir, datasets)

    v3_only: Dict[str, Tuple[int, int, Optional[float]]] = {}
    for ds in datasets:
        entry = v3_raw.get(ds, {}).get("soo_centered_v3")
        if entry is not None:
            v3_only[ds] = entry

    if not v3_only:
        print(f"[ERROR] No soo_centered_v3 eval files found under {args.v3_dir}")
        sys.exit(1)

    print()
    print(f"Baseline: {args.baseline_dir}")
    print(f"v3 run:   {args.v3_dir}")
    print()

    for ds in datasets:
        if ds not in v3_only:
            continue
        v3_total, _c, v3_acc = v3_only[ds]
        print_dataset_table(ds, v3_acc, v3_total, baseline.get(ds, {}))

    print_cross_dataset_summary(datasets, v3_only, baseline)
    rank_summary(datasets, v3_only, baseline)
    print()


if __name__ == "__main__":
    main()
