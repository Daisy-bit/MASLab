"""Aggregate 3 runs of the same SCC paper-table CSV into mean / mean+std.

Use case: `exp/run_both_scc_1.5b_3runs.sh` saves
  paper_tables/3runs_<TS>/<prefix>_run{1,2,3}.csv

This script reads those, computes per-cell mean & std across runs (skipping
non-numeric cells like dataset / variant labels), and writes:
  paper_tables/3runs_<TS>/<prefix>_mean.csv      (just the mean)
  paper_tables/3runs_<TS>/<prefix>_meanstd.csv   (mean±std formatted)

Usage:
  python scripts/diagnostic/aggregate_3runs.py \\
    --runs_dir paper_tables/3runs_<TS> \\
    --prefixes mad_scc_A1_1.5b mad_scc_A2_1.5b soo_scc_A1_1.5b soo_scc_A2_1.5b
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import List, Optional


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _parse_first_number(cell: str) -> Optional[float]:
    s = (cell or "").strip()
    if not s or s.upper() == "N/A":
        return None
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _read_csv(path: str) -> List[List[str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def _is_numeric_column(rows: List[List[List[str]]], col_idx: int) -> bool:
    for run_rows in rows:
        for r in run_rows[1:]:
            if col_idx < len(r) and _parse_first_number(r[col_idx]) is not None:
                return True
    return False


def _aggregate(prefix: str, runs_dir: str, n_runs: int) -> None:
    paths = [
        os.path.join(runs_dir, f"{prefix}_run{i}.csv") for i in range(1, n_runs + 1)
    ]
    runs = [_read_csv(p) for p in paths]
    runs = [r for r in runs if r]
    if not runs:
        print(f"[skip] no runs found for {prefix}")
        return

    n_rows = min(len(r) for r in runs)
    if n_rows < 2:
        print(f"[skip] empty CSVs for {prefix}")
        return
    n_cols = min(len(r[0]) for r in runs)
    header = runs[0][0][:n_cols]

    mean_rows: List[List[str]] = [list(header)]
    meanstd_rows: List[List[str]] = [list(header)]
    for r in range(1, n_rows):
        mean_row, meanstd_row = [], []
        for c in range(n_cols):
            cells = [runs[s][r][c] for s in range(len(runs))]
            nums = [_parse_first_number(x) for x in cells]
            nums = [x for x in nums if x is not None]
            if nums and _is_numeric_column(runs, c):
                mean = sum(nums) / len(nums)
                if len(nums) >= 2:
                    var = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
                    std = var ** 0.5
                else:
                    std = 0.0
                first = cells[0]
                m = _NUMBER_RE.search(first)
                tail = first[m.end():] if m else ""
                mean_row.append(f"{mean:.2f}{tail}")
                meanstd_row.append(f"{mean:.2f}±{std:.2f}{tail}")
            else:
                copied = next((x for x in cells if x and x.strip()), "")
                mean_row.append(copied)
                meanstd_row.append(copied)
        mean_rows.append(mean_row)
        meanstd_rows.append(meanstd_row)

    mean_path = os.path.join(runs_dir, f"{prefix}_mean.csv")
    meanstd_path = os.path.join(runs_dir, f"{prefix}_meanstd.csv")
    with open(mean_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(mean_rows)
    with open(meanstd_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(meanstd_rows)
    print(f">> {prefix}: aggregated {len(runs)}/{n_runs} runs")
    print(f"   mean    -> {mean_path}")
    print(f"   meanstd -> {meanstd_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--prefixes", nargs="+", required=True,
                    help="Filename prefixes shared by run1/run2/run3 CSVs.")
    ap.add_argument("--n_runs", type=int, default=3)
    args = ap.parse_args()
    for p in args.prefixes:
        _aggregate(p, args.runs_dir, args.n_runs)


if __name__ == "__main__":
    main()
