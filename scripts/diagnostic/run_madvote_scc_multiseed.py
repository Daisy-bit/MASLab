"""
Run mad_vote_scc with a SHARED initial-response pool, repeated K seeds, then
aggregate results into mean ± std paper tables.

Layout produced (under --run_root):

  <run_root>/
    seed1/
      A0_vanilla/   <dataset>_infer.jsonl     (one per dataset)
        _tables/    table*.csv                (analyze_diagnostic output)
      A1_triggering/...
      ...
      _paper/
        madvote_scc_A1_<model_label>.csv      (build_madvote_scc_tables output)
        madvote_scc_A2_<model_label>.csv
    seed2/...
    seed3/...
    _aggregated/
      madvote_scc_A1_<model_label>_mean.csv
      madvote_scc_A1_<model_label>_meanstd.csv  (mean +/- std formatted)
      madvote_scc_A2_<model_label>_mean.csv
      madvote_scc_A2_<model_label>_meanstd.csv

The shared initial pool is read from --initial_pool_dir; each seed *replays*
the same round-0 responses for every variant, so bucket assignment is
identical across (seed, variant). Only debate randomness varies between seeds.

Usage example (Windows PowerShell, from MASLab/ project root):

  python scripts/diagnostic/run_madvote_scc_multiseed.py `
    --run_root results_madvote_scc/run_<TS>_multiseed `
    --initial_pool_dir results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged `
    --initial_pool_filename_pattern mad_vote_{dataset}_infer.jsonl `
    --model_name qwen25-1.5b-instruct `
    --model_label 1.5b `
    --seeds 1 2 3 `
    --datasets GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro `
    --variants A0_vanilla A1_triggering A2_routing A3_aggregation A4_all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


VARIANT_TO_CONFIG = {
    "A0_vanilla":     "config_a0_vanilla",
    "A1_triggering":  "config_a1_triggering",
    "A2_routing":     "config_a2_routing",
    "A3_aggregation": "config_a3_aggregation",
    "A4_all":         "config_a4_all",
    "A5_trig_rout":   "config_a5_trig_rout",
}

DEFAULT_DATASETS = ["GSM8K", "GSM-Hard", "AIME-2024", "AQUA-RAT", "MMLU-Pro"]
DEFAULT_VARIANTS = list(VARIANT_TO_CONFIG.keys())


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    print(f">> {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(
            f"Command failed (exit {res.returncode}): {' '.join(cmd)}"
        )


def _python_exe() -> str:
    return sys.executable or "python"


# ---------------------------------------------------------------------------
# Orchestration: one inference per (seed, variant, dataset)
# ---------------------------------------------------------------------------

def run_one_inference(
    *,
    project_root: str,
    seed: int,
    variant: str,
    dataset: str,
    model_name: str,
    model_temperature: float,
    initial_pool_dir: str,
    initial_pool_pattern: str,
    seed_dir: str,
    max_samples: Optional[int],
) -> str:
    """Run inference.py for one (seed, variant, dataset) combo. Returns the
    path of the resulting JSONL."""
    config_name = VARIANT_TO_CONFIG[variant]
    out_dir = os.path.join(seed_dir, variant)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mad_vote_scc_{dataset}_infer.jsonl")

    cmd = [
        _python_exe(), "inference.py",
        "--method_name", "mad_vote_scc",
        "--method_config_name", config_name,
        "--model_name", model_name,
        "--model_temperature", str(model_temperature),
        "--test_dataset_name", dataset,
        "--output_path", out_path,
        "--initial_pool_dir", initial_pool_dir,
        "--initial_pool_filename_pattern", initial_pool_pattern,
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    _run(cmd, cwd=project_root)
    return out_path


def run_one_analyze(
    *,
    project_root: str,
    variant_dir: str,
    datasets: List[str],
) -> None:
    """Run analyze_diagnostic.py to build _tables/ for one variant dir."""
    cmd = [
        _python_exe(),
        "scripts/diagnostic/analyze_diagnostic.py",
        "--infer_dir", variant_dir,
        "--output_dir", os.path.join(variant_dir, "_tables"),
        "--datasets", *datasets,
        "--filename_pattern", "mad_vote_scc_{dataset}_infer.jsonl",
    ]
    _run(cmd, cwd=project_root)


def run_one_paper_tables(
    *,
    project_root: str,
    seed_dir: str,
    model_label: str,
) -> Tuple[str, str]:
    """Run build_madvote_scc_tables.py for a single seed directory.
    Returns (a1_csv_path, a2_csv_path)."""
    out_dir = os.path.join(seed_dir, "_paper")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        _python_exe(),
        "scripts/diagnostic/build_madvote_scc_tables.py",
        "--run_dir", seed_dir,
        "--model_label", model_label,
        "--filename_pattern", "mad_vote_scc_{dataset}_infer.jsonl",
        "--out_dir", out_dir,
    ]
    _run(cmd, cwd=project_root)
    label_lower = model_label.lower().replace(" ", "")
    a1 = os.path.join(out_dir, f"madvote_scc_A1_{label_lower}.csv")
    a2 = os.path.join(out_dir, f"madvote_scc_A2_{label_lower}.csv")
    return a1, a2


# ---------------------------------------------------------------------------
# Aggregation: mean / std across seeds for each (row, column)
# ---------------------------------------------------------------------------

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
    """A column is treated as numeric if ANY cell across ANY seed parses as
    a number."""
    for seed_rows in rows:
        for r in seed_rows[1:]:
            if col_idx < len(r) and _parse_first_number(r[col_idx]) is not None:
                return True
    return False


def _aggregate_table(
    seed_csv_paths: List[str],
    out_mean_path: str,
    out_meanstd_path: str,
) -> None:
    """Read the same CSV across N seeds, write mean and mean+/-std variants.

    Assumes all seeds have the SAME row keys (same first column) and SAME
    column headers. Non-numeric cells (Variant labels, Dataset names) are
    copied verbatim from seed[0].
    """
    seed_rows: List[List[List[str]]] = [_read_csv(p) for p in seed_csv_paths]
    seed_rows = [s for s in seed_rows if s]
    if not seed_rows:
        print(f"[WARN] No seed CSVs found, skipping aggregation for "
              f"{out_mean_path}")
        return

    n_rows = min(len(s) for s in seed_rows)
    if n_rows < 2:
        print(f"[WARN] Empty CSVs, skipping {out_mean_path}")
        return
    n_cols = min(len(s[0]) for s in seed_rows)

    header = seed_rows[0][0][:n_cols]
    mean_rows: List[List[str]] = [list(header)]
    meanstd_rows: List[List[str]] = [list(header)]

    for r in range(1, n_rows):
        mean_row: List[str] = []
        meanstd_row: List[str] = []
        for c in range(n_cols):
            cells = [seed_rows[s][r][c] for s in range(len(seed_rows))]
            nums = [_parse_first_number(x) for x in cells]
            nums = [x for x in nums if x is not None]
            if nums and _is_numeric_column(seed_rows, c):
                mean = sum(nums) / len(nums)
                if len(nums) >= 2:
                    var = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
                    std = var ** 0.5
                else:
                    std = 0.0
                # Preserve special suffix such as " (n=...)" by reusing
                # whatever the first seed's cell formatted past the number.
                first_cell = cells[0]
                m = _NUMBER_RE.search(first_cell)
                tail = first_cell[m.end():] if m else ""
                mean_row.append(f"{mean:.2f}{tail}")
                meanstd_row.append(f"{mean:.2f}±{std:.2f}{tail}")
            else:
                # Copy the first non-empty cell (or empty string)
                copied = next((x for x in cells if x and x.strip()), "")
                mean_row.append(copied)
                meanstd_row.append(copied)
        mean_rows.append(mean_row)
        meanstd_rows.append(meanstd_row)

    os.makedirs(os.path.dirname(out_mean_path) or ".", exist_ok=True)
    with open(out_mean_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(mean_rows)
    with open(out_meanstd_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(meanstd_rows)
    print(f">> Wrote {out_mean_path}")
    print(f">> Wrote {out_meanstd_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True,
                    help="Output root (will contain seed1/ seed2/ ... and _aggregated/).")
    ap.add_argument("--initial_pool_dir", required=True,
                    help="Directory with shared initial-response JSONLs.")
    ap.add_argument("--initial_pool_filename_pattern",
                    default="mad_vote_{dataset}_infer.jsonl",
                    help="Filename pattern inside --initial_pool_dir.")
    ap.add_argument("--model_name", required=True,
                    help="Model name as registered in model_api_config.json.")
    ap.add_argument("--model_label", required=True,
                    help="Display label for the model (e.g. '1.5b').")
    ap.add_argument("--model_temperature", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    choices=DEFAULT_VARIANTS)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap samples per dataset (mainly for smoke tests).")
    ap.add_argument("--skip_inference", action="store_true",
                    help="Skip inference; only re-run analyze + aggregate.")
    ap.add_argument("--skip_analyze", action="store_true",
                    help="Skip analyze_diagnostic; only re-aggregate.")
    args = ap.parse_args()

    project_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."
    ))
    os.makedirs(args.run_root, exist_ok=True)

    seed_dirs: List[str] = []
    a1_paths: List[str] = []
    a2_paths: List[str] = []

    for seed in args.seeds:
        seed_dir = os.path.join(args.run_root, f"seed{seed}")
        seed_dirs.append(seed_dir)

        # ---------- Inference per (variant, dataset) ----------
        if not args.skip_inference:
            for variant in args.variants:
                for dataset in args.datasets:
                    run_one_inference(
                        project_root=project_root,
                        seed=seed,
                        variant=variant,
                        dataset=dataset,
                        model_name=args.model_name,
                        model_temperature=args.model_temperature,
                        initial_pool_dir=args.initial_pool_dir,
                        initial_pool_pattern=args.initial_pool_filename_pattern,
                        seed_dir=seed_dir,
                        max_samples=args.max_samples,
                    )

        # ---------- Analyze per variant ----------
        if not args.skip_analyze:
            for variant in args.variants:
                variant_dir = os.path.join(seed_dir, variant)
                if os.path.isdir(variant_dir):
                    run_one_analyze(
                        project_root=project_root,
                        variant_dir=variant_dir,
                        datasets=args.datasets,
                    )

        # ---------- Per-seed paper tables ----------
        a1_path, a2_path = run_one_paper_tables(
            project_root=project_root,
            seed_dir=seed_dir,
            model_label=args.model_label,
        )
        a1_paths.append(a1_path)
        a2_paths.append(a2_path)

    # ---------- Cross-seed aggregation ----------
    agg_dir = os.path.join(args.run_root, "_aggregated")
    label_lower = args.model_label.lower().replace(" ", "")
    _aggregate_table(
        seed_csv_paths=a1_paths,
        out_mean_path=os.path.join(
            agg_dir, f"madvote_scc_A1_{label_lower}_mean.csv"
        ),
        out_meanstd_path=os.path.join(
            agg_dir, f"madvote_scc_A1_{label_lower}_meanstd.csv"
        ),
    )
    _aggregate_table(
        seed_csv_paths=a2_paths,
        out_mean_path=os.path.join(
            agg_dir, f"madvote_scc_A2_{label_lower}_mean.csv"
        ),
        out_meanstd_path=os.path.join(
            agg_dir, f"madvote_scc_A2_{label_lower}_meanstd.csv"
        ),
    )

    print("\n>> All done.")
    print(f"   Per-seed paper tables: {[os.path.dirname(p) for p in a1_paths]}")
    print(f"   Aggregated tables:    {agg_dir}")


if __name__ == "__main__":
    main()
