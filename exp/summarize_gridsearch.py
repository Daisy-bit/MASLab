"""
Summarize soo_centered_v3 hyperparameter grid search.

Reads the directory produced by exp/run_v3_gridsearch.sh:

    <results_dir>/
        trials.json              # list of {name, overrides}
        <trial_name>/
            overrides.yaml
            <dataset>/
                soo_centered_v3_xverify_eval.jsonl

Prints a ranked table of trials by mean accuracy across datasets and
per-dataset best configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict


def _read_eval(path):
    if not os.path.exists(path):
        return 0, 0, None
    valid, correct = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = rec.get("eval_score")
            if score is None:
                continue
            valid += 1
            if score == 1:
                correct += 1
    if valid == 0:
        return 0, 0, None
    return valid, correct, correct / valid * 100.0


def _fmt_overrides(ov):
    if not ov:
        return "(baseline)"
    return " ".join(f"{k}={v}" for k, v in ov.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--datasets", default="",
                    help="Space-separated dataset names. If empty, infer from trial subdirs.")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    root = args.results_dir
    if not os.path.isdir(root):
        print(f"[ERROR] results_dir not found: {root}")
        sys.exit(1)

    trials_path = os.path.join(root, "trials.json")
    if not os.path.isfile(trials_path):
        print(f"[ERROR] trials.json not found in {root}")
        sys.exit(1)
    with open(trials_path, "r", encoding="utf-8") as f:
        trials = json.load(f)

    if args.datasets.strip():
        datasets = args.datasets.split()
    else:
        ds = set()
        for t in trials:
            tdir = os.path.join(root, t["name"])
            if not os.path.isdir(tdir):
                continue
            for entry in os.listdir(tdir):
                if os.path.isdir(os.path.join(tdir, entry)):
                    ds.add(entry)
        datasets = sorted(ds)

    rows = []
    for t in trials:
        name = t["name"]
        ov = t.get("overrides", {})
        tdir = os.path.join(root, name)
        per_ds = OrderedDict()
        accs = []
        for d in datasets:
            eval_path = os.path.join(tdir, d, "soo_centered_v3_xverify_eval.jsonl")
            valid, correct, acc = _read_eval(eval_path)
            per_ds[d] = (valid, correct, acc)
            if acc is not None:
                accs.append(acc)
        mean_acc = sum(accs) / len(accs) if accs else None
        rows.append({"name": name, "overrides": ov, "per_ds": per_ds, "mean": mean_acc})

    rows_ranked = sorted(
        rows,
        key=lambda r: (r["mean"] is None, -(r["mean"] or 0.0)),
    )

    print()
    print("=" * 110)
    print(f"  soo_centered_v3 Grid Search Results ({len(rows)} trials, {len(datasets)} datasets)")
    print(f"  Directory: {root}")
    print("=" * 110)
    header = f"{'Rank':>4} {'Trial':<26} {'Mean':>8} " + " ".join(f"{d:>10}" for d in datasets) + "  Overrides"
    print(header)
    print("-" * len(header))
    top = args.top_k if args.top_k > 0 else len(rows_ranked)
    for i, r in enumerate(rows_ranked[:top], 1):
        mean_str = f"{r['mean']:.2f}%" if r["mean"] is not None else "N/A"
        ds_strs = []
        for d in datasets:
            _, _, acc = r["per_ds"][d]
            ds_strs.append(f"{acc:.2f}%" if acc is not None else "N/A")
        ov_str = _fmt_overrides(r["overrides"])
        print(f"{i:>4} {r['name']:<26} {mean_str:>8} " + " ".join(f"{s:>10}" for s in ds_strs) + f"  {ov_str}")
    print("=" * 110)

    # Per-dataset best
    print()
    print("Best configuration per dataset:")
    for d in datasets:
        best = None
        for r in rows:
            _, _, acc = r["per_ds"][d]
            if acc is None:
                continue
            if best is None or acc > best[1]:
                best = (r, acc)
        if best is None:
            print(f"  {d:<14} no valid results")
        else:
            r, acc = best
            print(f"  {d:<14} {acc:.2f}%   [{r['name']}]   {_fmt_overrides(r['overrides'])}")

    # Best overall
    print()
    if rows_ranked and rows_ranked[0]["mean"] is not None:
        best = rows_ranked[0]
        print(f"Best overall (mean across {len(datasets)} datasets): {best['mean']:.2f}%")
        print(f"  Trial:     {best['name']}")
        print(f"  Overrides: {_fmt_overrides(best['overrides'])}")
    else:
        print("No valid overall results.")

    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write("trial,overrides,mean_accuracy," + ",".join(datasets) + "\n")
            for r in rows_ranked:
                mean = f"{r['mean']:.4f}" if r["mean"] is not None else ""
                ds_vals = []
                for d in datasets:
                    _, _, acc = r["per_ds"][d]
                    ds_vals.append(f"{acc:.4f}" if acc is not None else "")
                ov_str = _fmt_overrides(r["overrides"]).replace(",", ";")
                f.write(f"{r['name']},{ov_str},{mean}," + ",".join(ds_vals) + "\n")
        print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
