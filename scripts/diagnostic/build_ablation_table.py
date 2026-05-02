"""
Assemble the Experiment 2 ablation table (6 variants on a single model) from:
  * the existing 3B mad_vote diagnostic run (Vanilla MAD + Initial Vote rows)
  * the SCC ablation run produced by `exp/run_scc_ablation.sh`
    (scc_full / no_triggering / no_routing / no_aggregation)

Output: a 7-column, 6-row table:
  Variant | Triggering | Routing | Aggregation | Accuracy | Tokens | Flip | Recovery

Row order matches the user's paper spec:
  Initial Vote, Vanilla MAD, SCC w/o Triggering, SCC w/o Routing,
  SCC w/o Aggregation, SCC-full

Usage (from MASLab/ project root):
  python scripts/diagnostic/build_ablation_table.py \\
    --mad_run results_diagnostic/run_20260501_130540/qwen25-3b-instruct \\
    --scc_run results_ablation_scc/run_<TS> \\
    --output_csv paper_tables/ablation_3b.csv \\
    --output_md  paper_tables/ablation_3b.md
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CSV Avg-row reading (shared with build_regime_comparison.py)
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


def _read_avg_row(csv_path: str) -> Optional[List[str]]:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in reversed(rows):
        if row and row[0].strip().lower().startswith("avg"):
            return row
    return None


def _load_table2_avg(infer_dir: str) -> Dict[str, Optional[float]]:
    t2 = _read_avg_row(os.path.join(infer_dir, "_tables", "table2_regime_analysis.csv"))
    if t2 is None:
        print(f"[WARN] table2 missing in {infer_dir}", file=sys.stderr)
        return {"flip": None, "recovery": None, "unrecoverable": None, "final_acc": None}
    return {
        "flip": _parse_first_float(t2[2]) if len(t2) > 2 else None,
        "recovery": _parse_first_float(t2[3]) if len(t2) > 3 else None,
        "unrecoverable": _parse_first_float(t2[4]) if len(t2) > 4 else None,
        "final_acc": _parse_first_float(t2[5]) if len(t2) > 5 else None,
    }


# ---------------------------------------------------------------------------
# JSONL token aggregation
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _per_sample_total_tokens(infer_dir: str, filename_pattern: str, datasets: List[str]) -> List[int]:
    """Walk every per-dataset JSONL and return a flat list of per-sample
    `diagnostic.total_tokens` values (across all datasets)."""
    out: List[int] = []
    for ds in datasets:
        path = os.path.join(infer_dir, filename_pattern.format(dataset=ds))
        for rec in _iter_jsonl(path):
            d = rec.get("diagnostic")
            if not d or rec.get("error"):
                continue
            tt = d.get("total_tokens")
            if isinstance(tt, (int, float)):
                out.append(int(tt))
    return out


def _per_sample_initial_tokens_and_correct(infer_dir: str, filename_pattern: str,
                                           datasets: List[str]) -> Tuple[List[int], List[bool]]:
    """For the Initial-Vote row: walk mad_vote JSONL and compute, per sample,
    the initial-only token sum and the initial_vote_correct flag."""
    tokens: List[int] = []
    correct: List[bool] = []
    for ds in datasets:
        path = os.path.join(infer_dir, filename_pattern.format(dataset=ds))
        for rec in _iter_jsonl(path):
            d = rec.get("diagnostic")
            if not d or rec.get("error"):
                continue
            ir = d.get("initial_responses") or []
            tok = sum(int(r.get("tokens", 0)) for r in ir)
            tokens.append(tok)
            correct.append(bool(d.get("initial_vote_correct", False)))
    return tokens, correct


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANT_ROW_DEFS = [
    # (label,                    triggering, routing,    aggregation,    source-key)
    ("Initial Vote",             "X",        "X",        "plurality",    "initial_vote"),
    ("Vanilla MAD",              "fixed",    "full",     "plurality",    "mad_vote"),
    ("SCC w/o Triggering",       "fixed",    "spectral", "SCC agg",      "no_triggering"),
    ("SCC w/o Routing",          "adaptive", "full",     "SCC agg",      "no_routing"),
    ("SCC w/o Aggregation",      "adaptive", "spectral", "plurality",    "no_aggregation"),
    ("SCC-full",                 "adaptive", "spectral", "conservative", "scc_full"),
]


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "N/A"


def _fmt_int(x: Optional[float]) -> str:
    return f"{x:.0f}" if x is not None else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mad_run", required=True,
        help="Path to the mad_vote diagnostic dir (e.g. results_diagnostic/run_20260501_130540/qwen25-3b-instruct).",
    )
    parser.add_argument(
        "--scc_run", required=True,
        help="Path to the SCC ablation run root (containing scc_full/, no_triggering/, ...).",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["GSM8K", "GSM-Hard", "AIME-2024", "AQUA-RAT", "MMLU-Pro"],
    )
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    # Vanilla MAD: Table 2 + per-sample tokens from mad_vote_<ds>_infer.jsonl.
    mad_table2 = _load_table2_avg(args.mad_run)
    mad_tokens = _mean(_per_sample_total_tokens(
        args.mad_run, "mad_vote_{dataset}_infer.jsonl", args.datasets,
    ))

    # Initial Vote: derive from the same mad_vote JSONL (initial-only tokens
    # and initial_vote_correct accuracy). Flip = Recovery = 0 by construction.
    init_tokens, init_correct = _per_sample_initial_tokens_and_correct(
        args.mad_run, "mad_vote_{dataset}_infer.jsonl", args.datasets,
    )
    init_acc = _mean([1.0 if c else 0.0 for c in init_correct])
    init_acc_pct = init_acc * 100.0 if init_acc is not None else None
    init_tok_mean = _mean(init_tokens)

    # SCC variants: Table 2 + per-sample tokens for each.
    scc_results: Dict[str, Dict[str, Optional[float]]] = {}
    for variant_dir in ("scc_full", "no_triggering", "no_routing", "no_aggregation"):
        infer_dir = os.path.join(args.scc_run, variant_dir)
        scc_results[variant_dir] = _load_table2_avg(infer_dir)
        scc_results[variant_dir]["tokens"] = _mean(_per_sample_total_tokens(
            infer_dir, "soo_centered_v3_{dataset}_infer.jsonl", args.datasets,
        ))

    # Build rows.
    header = [
        "Variant", "Triggering", "Routing", "Aggregation",
        "Accuracy", "Tokens",
        "Flip (lower=better)", "Recovery (higher=better)",
    ]
    rows: List[List[str]] = [header]

    for label, trig, route, agg, key in VARIANT_ROW_DEFS:
        if key == "initial_vote":
            rows.append([
                label, trig, route, agg,
                _fmt_pct(init_acc_pct),
                _fmt_int(init_tok_mean),
                "0.00", "0.00",
            ])
        elif key == "mad_vote":
            rows.append([
                label, trig, route, agg,
                _fmt_pct(mad_table2["final_acc"]),
                _fmt_int(mad_tokens),
                _fmt_pct(mad_table2["flip"]),
                _fmt_pct(mad_table2["recovery"]),
            ])
        else:
            cell = scc_results.get(key, {})
            rows.append([
                label, trig, route, agg,
                _fmt_pct(cell.get("final_acc")),
                _fmt_int(cell.get("tokens")),
                _fmt_pct(cell.get("flip")),
                _fmt_pct(cell.get("recovery")),
            ])

    # Write outputs.
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows[1:]:
            f.write("| " + " | ".join(row) + " |\n")

    for row in rows:
        print(",".join(row))
    print(f">> Wrote {args.output_csv}")
    print(f">> Wrote {args.output_md}")


if __name__ == "__main__":
    main()
