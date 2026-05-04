"""
Assemble Cluster A paper tables (A1 + A2) from a single
results_madvote_scc/run_<TS>/<model>/ run with five variant subdirs:
    A0_vanilla / A1_triggering / A2_routing / A3_aggregation / A4_all

Each variant subdir is expected to already contain `_tables/table*.csv`
produced by `scripts/diagnostic/analyze_diagnostic.py`.

Outputs (under --out_dir, default `paper_tables/`):
    <prefix>_A1_<label>.csv / .md   — 5-row main table
        Model | Variant | Acc | Already-flip | Recoverable Recovery |
        Unrecoverable Success | Coverage Drop | Tokens
    <prefix>_A2_<label>.csv / .md   — per-dataset detail table for one model
        Dataset | A0 Acc | A1 Acc | A2 Acc | A3 Acc | A4 Acc |
        Δ A4 vs A0 | A0 Tokens | A4 Tokens | Token saving %

All percentages use the per-dataset macro average reported by
analyze_diagnostic's "Avg." row in table2 / table6, so the numbers are
directly comparable to results_diagnostic.

Usage (from MASLab/ project root):
  python scripts/diagnostic/build_madvote_scc_tables.py \\
    --run_dir results_madvote_scc/run_<TS>/qwen25-1.5b-instruct \\
    --model_label 1.5B \\
    --out_dir paper_tables \\
    --prefix madvote_scc
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


VARIANT_DEFS = [
    # (subdir_name,      table-row label)
    ("A0_vanilla",       "A0 vanilla"),
    ("A1_triggering",    "A1 +Trig"),
    ("A2_routing",       "A2 +Rout"),
    ("A3_aggregation",   "A3 +Aggr"),
    ("A4_all",           "A4 +All"),
]

DEFAULT_DATASETS = ["GSM8K", "GSM-Hard", "AIME-2024", "AQUA-RAT", "MMLU-Pro"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _parse_first_float(cell: str) -> Optional[float]:
    cell = (cell or "").strip()
    if not cell:
        return None
    head = cell.split()[0].split("[")[0]
    try:
        return float(head)
    except ValueError:
        return None


def _read_csv_rows(path: str) -> List[List[str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _read_avg_row(path: str) -> Optional[List[str]]:
    rows = _read_csv_rows(path)
    for row in reversed(rows):
        if row and row[0].strip().lower().startswith("avg"):
            return row
    return None


def _read_dataset_row(path: str, dataset: str) -> Optional[List[str]]:
    rows = _read_csv_rows(path)
    for row in rows[1:]:
        if row and row[0].strip() == dataset:
            return row
    return None


# ---------------------------------------------------------------------------
# JSONL token aggregation
# ---------------------------------------------------------------------------

def _per_sample_total_tokens(
    infer_dir: str, datasets: List[str], filename_pattern: str
) -> Dict[str, Optional[float]]:
    """Per-dataset MEAN of `diagnostic.total_tokens` (skip records with error)."""
    out: Dict[str, Optional[float]] = {}
    for ds in datasets:
        path = os.path.join(infer_dir, filename_pattern.format(dataset=ds))
        if not os.path.exists(path):
            out[ds] = None
            continue
        toks: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("error"):
                    continue
                d = rec.get("diagnostic")
                if not d:
                    continue
                tt = d.get("total_tokens")
                if isinstance(tt, (int, float)):
                    toks.append(int(tt))
        out[ds] = (sum(toks) / len(toks)) if toks else None
    return out


def _macro_mean(per_ds: Dict[str, Optional[float]]) -> Optional[float]:
    vs = [v for v in per_ds.values() if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


# ---------------------------------------------------------------------------
# Loading per-variant cells
# ---------------------------------------------------------------------------

def _load_variant_cell(
    infer_dir: str, datasets: List[str], filename_pattern: str
) -> Dict[str, Any]:
    t2_path = os.path.join(infer_dir, "_tables", "table2_regime_analysis.csv")
    t6_path = os.path.join(infer_dir, "_tables", "table6_coverage_survival.csv")
    t2_avg = _read_avg_row(t2_path)
    t6_avg = _read_avg_row(t6_path)

    flip = _parse_first_float(t2_avg[2]) if t2_avg and len(t2_avg) > 2 else None
    recovery = _parse_first_float(t2_avg[3]) if t2_avg and len(t2_avg) > 3 else None
    unrec = _parse_first_float(t2_avg[4]) if t2_avg and len(t2_avg) > 4 else None
    final_acc = _parse_first_float(t2_avg[5]) if t2_avg and len(t2_avg) > 5 else None

    init_cov = _parse_first_float(t6_avg[2]) if t6_avg and len(t6_avg) > 2 else None
    final_cov = _parse_first_float(t6_avg[3]) if t6_avg and len(t6_avg) > 3 else None
    cov_drop = (
        (init_cov - final_cov)
        if (init_cov is not None and final_cov is not None)
        else None
    )

    per_ds_acc: Dict[str, Optional[float]] = {}
    for ds in datasets:
        row = _read_dataset_row(t2_path, ds)
        per_ds_acc[ds] = (
            _parse_first_float(row[5]) if row and len(row) > 5 else None
        )

    per_ds_tokens = _per_sample_total_tokens(infer_dir, datasets, filename_pattern)
    macro_tokens = _macro_mean(per_ds_tokens)

    return {
        "flip": flip,
        "recovery": recovery,
        "unrec": unrec,
        "final_acc": final_acc,
        "cov_drop": cov_drop,
        "tokens": macro_tokens,
        "per_ds_acc": per_ds_acc,
        "per_ds_tokens": per_ds_tokens,
    }


# ---------------------------------------------------------------------------
# Formatters / writers
# ---------------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "N/A"


def _fmt_int(x: Optional[float]) -> str:
    return f"{x:.0f}" if x is not None else "N/A"


def _write_csv_md(path_csv: str, path_md: str, rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    os.makedirs(os.path.dirname(path_md) or ".", exist_ok=True)
    with open(path_md, "w", encoding="utf-8") as f:
        if not rows:
            return
        header, *body = rows
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in body:
            f.write("| " + " | ".join(row) + " |\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir", required=True,
        help="Directory containing the four variant subdirs "
             "(A0_vanilla / A1_triggering / A2_routing / A3_both).",
    )
    ap.add_argument("--model_label", default="1.5B")
    ap.add_argument(
        "--filename_pattern", default="mad_vote_scc_{dataset}_infer.jsonl",
    )
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--out_dir", default="paper_tables")
    ap.add_argument("--prefix", default="madvote_scc")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------- per-variant cells --------
    variant_data: Dict[str, Dict[str, Any]] = {}
    for var_dir, _label in VARIANT_DEFS:
        infer_dir = os.path.join(args.run_dir, var_dir)
        if not os.path.isdir(infer_dir):
            print(f"[WARN] variant dir missing: {infer_dir}")
            variant_data[var_dir] = {
                "flip": None, "recovery": None, "unrec": None,
                "final_acc": None, "cov_drop": None, "tokens": None,
                "per_ds_acc": {ds: None for ds in args.datasets},
                "per_ds_tokens": {ds: None for ds in args.datasets},
            }
            continue
        variant_data[var_dir] = _load_variant_cell(
            infer_dir, args.datasets, args.filename_pattern,
        )

    # -------- Table A1 --------
    a1_header = [
        "Model", "Variant",
        "Acc.",
        "Already-flip (lower=better)",
        "Recoverable Recovery (higher=better)",
        "Unrecoverable Success (higher=better)",
        "Coverage Drop (lower=better)",
        "Tokens",
    ]
    a1_rows: List[List[str]] = [a1_header]
    for var_dir, label in VARIANT_DEFS:
        d = variant_data[var_dir]
        a1_rows.append([
            args.model_label, label,
            _fmt_pct(d["final_acc"]),
            _fmt_pct(d["flip"]),
            _fmt_pct(d["recovery"]),
            _fmt_pct(d["unrec"]),
            _fmt_pct(d["cov_drop"]),
            _fmt_int(d["tokens"]),
        ])

    # -------- Table A2 (per-dataset, fixed model) --------
    a2_header = (
        ["Dataset"]
        + [f"{label} Acc" for _, label in VARIANT_DEFS]
        + ["Δ A4 vs A0", "A0 Tokens", "A4 Tokens", "Token saving %"]
    )
    a2_rows: List[List[str]] = [a2_header]
    a0 = variant_data["A0_vanilla"]
    a4 = variant_data["A4_all"]
    for ds in args.datasets:
        row: List[str] = [ds]
        for var_dir, _label in VARIANT_DEFS:
            row.append(_fmt_pct(variant_data[var_dir]["per_ds_acc"].get(ds)))
        a0_acc = a0["per_ds_acc"].get(ds)
        a4_acc = a4["per_ds_acc"].get(ds)
        delta = (
            (a4_acc - a0_acc)
            if (a0_acc is not None and a4_acc is not None)
            else None
        )
        a0_tok = a0["per_ds_tokens"].get(ds)
        a4_tok = a4["per_ds_tokens"].get(ds)
        saving = (
            (1.0 - a4_tok / a0_tok) * 100.0
            if (a0_tok is not None and a4_tok is not None and a0_tok > 0)
            else None
        )
        row += [
            _fmt_pct(delta),
            _fmt_int(a0_tok),
            _fmt_int(a4_tok),
            _fmt_pct(saving),
        ]
        a2_rows.append(row)

    label_lower = args.model_label.lower().replace(" ", "")
    a1_csv = os.path.join(args.out_dir, f"{args.prefix}_A1_{label_lower}.csv")
    a1_md = os.path.join(args.out_dir, f"{args.prefix}_A1_{label_lower}.md")
    a2_csv = os.path.join(args.out_dir, f"{args.prefix}_A2_{label_lower}.csv")
    a2_md = os.path.join(args.out_dir, f"{args.prefix}_A2_{label_lower}.md")
    _write_csv_md(a1_csv, a1_md, a1_rows)
    _write_csv_md(a2_csv, a2_md, a2_rows)

    print(">> Wrote:")
    for p in (a1_csv, a1_md, a2_csv, a2_md):
        print(f"   {p}")


if __name__ == "__main__":
    main()
