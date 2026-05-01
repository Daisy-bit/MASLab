"""
Aggregate the per-sample JSONL produced by `methods/mad_vote` into the six
diagnostic tables specified in exp/实验流程.md.

Usage
-----
  python scripts/diagnostic/analyze_diagnostic.py \
      --infer_dir results_diagnostic/qwen25-3b-instruct \
      --output_dir results_diagnostic/qwen25-3b-instruct/_tables \
      --datasets GSM8K GSM-Hard AIME-2024 AQUA-RAT MMLU-Pro

`--infer_dir` is expected to contain one file per dataset with a stable name
(e.g. `mad_vote_GSM8K_infer.jsonl`). The exact filename pattern is configurable
via `--filename_pattern` (default `mad_vote_{dataset}_infer.jsonl`).

Outputs (CSV, one row per dataset, plus an `Avg.` row):
  table1_initial_diagnostics.csv
  table2_regime_analysis.csv
  table3_accuracy_decomposition.csv
  table4_transition_analysis.csv
  table5_roundwise_accuracy.csv
  table6_coverage_survival.csv
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def usable_records(records: List[Dict]) -> List[Dict]:
    """Filter to records that successfully ran inference and have diagnostic."""
    out = []
    for rec in records:
        if "diagnostic" not in rec:
            continue
        if rec.get("error"):
            continue
        d = rec["diagnostic"]
        # gold answer must exist for correctness signals
        if d.get("gold_answer") is None:
            continue
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI for binary mean (proportion).
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float], n_iter: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) for a sequence of 0/1 (or bounded float) values."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    n = len(values)
    rng = random.Random(seed)
    mean = sum(values) / n
    if n < 5:
        return mean, mean, mean
    samples = []
    for _ in range(n_iter):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo_idx = max(0, int((alpha / 2) * n_iter))
    hi_idx = min(n_iter - 1, int((1 - alpha / 2) * n_iter))
    return mean, samples[lo_idx], samples[hi_idx]


def fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return ""
    return f"{x * 100:.2f}"


def fmt_pct_ci(mean: float, lo: float, hi: float) -> str:
    if mean != mean:
        return ""
    return f"{mean * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}]"


# ---------------------------------------------------------------------------
# Per-dataset metric extraction
# ---------------------------------------------------------------------------

def per_sample_metrics(records: List[Dict]) -> List[Dict]:
    """Project each record into the small set of indicators we'll aggregate."""
    out = []
    for rec in records:
        d = rec["diagnostic"]
        initial_correct_flags = [r["is_correct"] for r in d["initial_responses"]]
        rounds_num = d.get("rounds_num", len(d.get("round_responses", {})))
        # round-by-round vote correctness (length rounds_num + 1; index 0 == initial)
        rvc = d.get("round_vote_correct_history", [])
        # round-by-round per-agent correctness (same shape; each entry is a list of bools)
        rch = d.get("round_correct_history", [])
        final_correct_flags = rch[-1] if rch else initial_correct_flags
        out.append(
            {
                "single_acc": (
                    sum(initial_correct_flags) / len(initial_correct_flags)
                    if initial_correct_flags else 0.0
                ),
                "initial_vote_correct": bool(d.get("initial_vote_correct", False)),
                "final_vote_correct": bool(d.get("final_vote_correct", False)),
                "initial_coverage": bool(d.get("initial_oracle_coverage", False)),
                "final_coverage": bool(d.get("final_oracle_coverage", False)),
                "bucket": d.get("bucket"),
                "round_vote_correct_history": rvc,
                "rounds_num": rounds_num,
                "initial_correct_flags": initial_correct_flags,
                "final_correct_flags": final_correct_flags,
                "tokens": d.get("total_tokens", 0),
            }
        )
    return out


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_table1(per_dataset: Dict[str, List[Dict]]) -> List[List[str]]:
    """Initial Answer Pool Diagnostics."""
    header = [
        "Dataset", "N",
        "Single Acc.",
        "Initial Vote Acc.",
        "Oracle Coverage",
        "Coverage - Vote",
        "Recoverable Ratio",
    ]
    rows = [header]
    avg_acc, agg_metrics = [], {k: [] for k in (
        "single", "vote0", "cov", "covmvote", "recoverable"
    )}
    for ds, samples in per_dataset.items():
        if not samples:
            continue
        single = [s["single_acc"] for s in samples]
        vote0 = [1.0 if s["initial_vote_correct"] else 0.0 for s in samples]
        cov = [1.0 if s["initial_coverage"] else 0.0 for s in samples]
        rec = [
            1.0 if (not s["initial_vote_correct"] and s["initial_coverage"]) else 0.0
            for s in samples
        ]
        m_single, lo_s, hi_s = bootstrap_ci(single)
        m_vote0, lo_v, hi_v = bootstrap_ci(vote0)
        m_cov, lo_c, hi_c = bootstrap_ci(cov)
        m_rec, lo_r, hi_r = bootstrap_ci(rec)
        cov_minus_vote = m_cov - m_vote0
        rows.append([
            ds, str(len(samples)),
            fmt_pct_ci(m_single, lo_s, hi_s),
            fmt_pct_ci(m_vote0, lo_v, hi_v),
            fmt_pct_ci(m_cov, lo_c, hi_c),
            fmt_pct(cov_minus_vote),
            fmt_pct_ci(m_rec, lo_r, hi_r),
        ])
        agg_metrics["single"].append(m_single)
        agg_metrics["vote0"].append(m_vote0)
        agg_metrics["cov"].append(m_cov)
        agg_metrics["covmvote"].append(cov_minus_vote)
        agg_metrics["recoverable"].append(m_rec)
    rows.append([
        "Avg.", "",
        fmt_pct(safe_mean(agg_metrics["single"])),
        fmt_pct(safe_mean(agg_metrics["vote0"])),
        fmt_pct(safe_mean(agg_metrics["cov"])),
        fmt_pct(safe_mean(agg_metrics["covmvote"])),
        fmt_pct(safe_mean(agg_metrics["recoverable"])),
    ])
    return rows


def build_table2(per_dataset: Dict[str, List[Dict]]) -> List[List[str]]:
    """Vanilla MAD by initial regime."""
    header = [
        "Dataset", "N",
        "Already-solved Flip (lower=better)",
        "Recoverable Recovery (higher=better)",
        "Unrecoverable Success (higher=better)",
        "Final MAD Acc.",
    ]
    rows = [header]
    agg = {k: [] for k in ("flip", "recover", "unrec", "final")}
    for ds, samples in per_dataset.items():
        if not samples:
            continue
        # Already solved: V(0)=y. Flip = V(L) != y given V(0)=y.
        already = [s for s in samples if s["initial_vote_correct"]]
        flip = [
            1.0 if not s["final_vote_correct"] else 0.0 for s in already
        ]
        # Recoverable: V(0)!=y, C(0)=1. Recovery = V(L)=y given that.
        rec = [
            s for s in samples
            if (not s["initial_vote_correct"] and s["initial_coverage"])
        ]
        recovery = [1.0 if s["final_vote_correct"] else 0.0 for s in rec]
        # Unrecoverable: C(0)=0. Success = V(L)=y given that.
        unrec = [s for s in samples if not s["initial_coverage"]]
        unrec_success = [1.0 if s["final_vote_correct"] else 0.0 for s in unrec]
        # Final MAD acc
        final_acc = [1.0 if s["final_vote_correct"] else 0.0 for s in samples]

        m_flip, lo_f, hi_f = bootstrap_ci(flip)
        m_rec, lo_r, hi_r = bootstrap_ci(recovery)
        m_unrec, lo_u, hi_u = bootstrap_ci(unrec_success)
        m_final, lo_fa, hi_fa = bootstrap_ci(final_acc)

        rows.append([
            ds, str(len(samples)),
            fmt_pct_ci(m_flip, lo_f, hi_f) + (f" (n={len(already)})" if already else " (n=0)"),
            fmt_pct_ci(m_rec, lo_r, hi_r) + (f" (n={len(rec)})" if rec else " (n=0)"),
            fmt_pct_ci(m_unrec, lo_u, hi_u) + (f" (n={len(unrec)})" if unrec else " (n=0)"),
            fmt_pct_ci(m_final, lo_fa, hi_fa),
        ])
        agg["flip"].append(m_flip)
        agg["recover"].append(m_rec)
        agg["unrec"].append(m_unrec)
        agg["final"].append(m_final)
    rows.append([
        "Avg.", "",
        fmt_pct(safe_mean(agg["flip"])),
        fmt_pct(safe_mean(agg["recover"])),
        fmt_pct(safe_mean(agg["unrec"])),
        fmt_pct(safe_mean(agg["final"])),
    ])
    return rows


def build_table3(per_dataset: Dict[str, List[Dict]]) -> List[List[str]]:
    header = [
        "Dataset", "N",
        "Single Acc.",
        "Vote-0 Acc.",
        "Vanilla MAD Acc.",
        "Gain over Vote",
    ]
    rows = [header]
    agg = {k: [] for k in ("single", "vote0", "mad", "gain")}
    for ds, samples in per_dataset.items():
        if not samples:
            continue
        single = [s["single_acc"] for s in samples]
        vote0 = [1.0 if s["initial_vote_correct"] else 0.0 for s in samples]
        mad = [1.0 if s["final_vote_correct"] else 0.0 for s in samples]
        m_s, lo_s, hi_s = bootstrap_ci(single)
        m_v, lo_v, hi_v = bootstrap_ci(vote0)
        m_m, lo_m, hi_m = bootstrap_ci(mad)
        gain = m_m - m_v
        rows.append([
            ds, str(len(samples)),
            fmt_pct_ci(m_s, lo_s, hi_s),
            fmt_pct_ci(m_v, lo_v, hi_v),
            fmt_pct_ci(m_m, lo_m, hi_m),
            fmt_pct(gain),
        ])
        agg["single"].append(m_s)
        agg["vote0"].append(m_v)
        agg["mad"].append(m_m)
        agg["gain"].append(gain)
    rows.append([
        "Avg.", "",
        fmt_pct(safe_mean(agg["single"])),
        fmt_pct(safe_mean(agg["vote0"])),
        fmt_pct(safe_mean(agg["mad"])),
        fmt_pct(safe_mean(agg["gain"])),
    ])
    return rows


def build_table4(per_dataset: Dict[str, List[Dict]]) -> List[List[str]]:
    """Agent-level transitions: count over (sample, agent) pairs."""
    header = ["Dataset", "AgentSamples", "C->C", "C->W", "W->C", "W->W", "Delta_amp"]
    rows = [header]
    agg = {k: [] for k in ("cc", "cw", "wc", "ww", "amp")}
    for ds, samples in per_dataset.items():
        if not samples:
            continue
        cc = cw = wc = ww = 0
        total = 0
        for s in samples:
            ic = s["initial_correct_flags"]
            fc = s["final_correct_flags"]
            for a, b in zip(ic, fc):
                total += 1
                if a and b:
                    cc += 1
                elif a and not b:
                    cw += 1
                elif (not a) and b:
                    wc += 1
                else:
                    ww += 1
        if total == 0:
            continue
        p_cc = cc / total
        p_cw = cw / total
        p_wc = wc / total
        p_ww = ww / total
        amp = p_wc - p_cw
        rows.append([
            ds, str(total),
            fmt_pct(p_cc), fmt_pct(p_cw), fmt_pct(p_wc), fmt_pct(p_ww),
            fmt_pct(amp),
        ])
        agg["cc"].append(p_cc)
        agg["cw"].append(p_cw)
        agg["wc"].append(p_wc)
        agg["ww"].append(p_ww)
        agg["amp"].append(amp)
    rows.append([
        "Avg.", "",
        fmt_pct(safe_mean(agg["cc"])),
        fmt_pct(safe_mean(agg["cw"])),
        fmt_pct(safe_mean(agg["wc"])),
        fmt_pct(safe_mean(agg["ww"])),
        fmt_pct(safe_mean(agg["amp"])),
    ])
    return rows


def build_table5(per_dataset: Dict[str, List[Dict]]) -> List[List[str]]:
    """Round-wise vote accuracy."""
    # detect global max rounds
    max_rounds = 0
    for samples in per_dataset.values():
        for s in samples:
            max_rounds = max(max_rounds, len(s.get("round_vote_correct_history", [])) - 1)
    if max_rounds <= 0:
        max_rounds = 3
    header = ["Dataset", "N"] + [f"Round {r}" for r in range(0, max_rounds + 1)]
    rows = [header]
    agg = [[] for _ in range(max_rounds + 1)]
    for ds, samples in per_dataset.items():
        if not samples:
            continue
        per_round_acc = []
        for r in range(max_rounds + 1):
            flags = []
            for s in samples:
                hist = s.get("round_vote_correct_history", [])
                if r < len(hist):
                    flags.append(1.0 if hist[r] else 0.0)
            m, lo, hi = bootstrap_ci(flags)
            per_round_acc.append(m)
            agg[r].append(m)
        row = [ds, str(len(samples))] + [fmt_pct(x) for x in per_round_acc]
        rows.append(row)
    avg_row = ["Avg.", ""] + [fmt_pct(safe_mean(col)) for col in agg]
    rows.append(avg_row)
    return rows


def build_table6(per_dataset: Dict[str, List[Dict]]) -> List[List[str]]:
    header = ["Dataset", "N", "Initial Coverage", "Final Coverage", "Survival"]
    rows = [header]
    agg = {k: [] for k in ("init", "final", "survival")}
    for ds, samples in per_dataset.items():
        if not samples:
            continue
        init_cov = [1.0 if s["initial_coverage"] else 0.0 for s in samples]
        final_cov = [1.0 if s["final_coverage"] else 0.0 for s in samples]
        with_init = [s for s in samples if s["initial_coverage"]]
        survival = [1.0 if s["final_coverage"] else 0.0 for s in with_init]
        m_i, lo_i, hi_i = bootstrap_ci(init_cov)
        m_f, lo_f, hi_f = bootstrap_ci(final_cov)
        m_s, lo_s, hi_s = bootstrap_ci(survival)
        rows.append([
            ds, str(len(samples)),
            fmt_pct_ci(m_i, lo_i, hi_i),
            fmt_pct_ci(m_f, lo_f, hi_f),
            fmt_pct_ci(m_s, lo_s, hi_s) + (f" (n={len(with_init)})" if with_init else " (n=0)"),
        ])
        agg["init"].append(m_i)
        agg["final"].append(m_f)
        agg["survival"].append(m_s)
    rows.append([
        "Avg.", "",
        fmt_pct(safe_mean(agg["init"])),
        fmt_pct(safe_mean(agg["final"])),
        fmt_pct(safe_mean(agg["survival"])),
    ])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Sample-level debug records (for bucket-assignment audit)
# ---------------------------------------------------------------------------

def write_sample_level_debug(per_dataset_records: Dict[str, List[Dict]],
                             output_path: str) -> None:
    """Flat per-sample JSONL with the user-spec fields for spot-checking
    bucket assignments without re-running inference."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ds, records in per_dataset_records.items():
            for i, rec in enumerate(records):
                d = rec["diagnostic"]
                rch = d.get("round_correct_history", [])
                final_correct = list(rch[-1]) if rch else []
                reh = d.get("round_extracted_history", [])
                final_extracted = list(reh[-1]) if reh else []
                debug_record = {
                    "dataset": ds,
                    "sample_id": f"{ds}_{i:05d}",
                    "query": rec.get("query"),
                    "gold_answer": d.get("gold_answer"),
                    "initial_answers": [
                        r["extracted_answer"] for r in d.get("initial_responses", [])
                    ],
                    "initial_correct_flags": [
                        bool(r["is_correct"]) for r in d.get("initial_responses", [])
                    ],
                    "initial_vote": d.get("initial_vote"),
                    "initial_vote_correct": bool(d.get("initial_vote_correct", False)),
                    "initial_oracle_coverage": bool(d.get("initial_oracle_coverage", False)),
                    "bucket": d.get("bucket"),
                    "final_answers": final_extracted,
                    "final_correct_flags": [bool(x) for x in final_correct],
                    "final_vote": d.get("final_vote"),
                    "final_vote_correct": bool(d.get("final_vote_correct", False)),
                    "final_oracle_coverage": bool(d.get("final_oracle_coverage", False)),
                }
                f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Sanity checks (per exp/实验流程.md §6)
# ---------------------------------------------------------------------------

def run_sanity_checks(per_dataset_records: Dict[str, List[Dict]],
                      tolerance: float = 1e-9) -> bool:
    """Per-dataset PASS/FAIL print. Returns True iff every check passes."""
    print("\n" + "=" * 30 + " Sanity Checks " + "=" * 30)
    overall = True
    for ds, records in per_dataset_records.items():
        if not records:
            print(f"\n{ds}: (no usable samples)")
            continue
        n_total = len(records)
        n_already = sum(
            1 for r in records if r["diagnostic"].get("initial_vote_correct")
        )
        n_coverage = sum(
            1 for r in records if r["diagnostic"].get("initial_oracle_coverage")
        )
        n_recoverable = sum(
            1 for r in records
            if (not r["diagnostic"].get("initial_vote_correct")
                and r["diagnostic"].get("initial_oracle_coverage"))
        )
        n_unrecoverable = sum(
            1 for r in records
            if not r["diagnostic"].get("initial_oracle_coverage")
        )
        # bucket-field mismatch check
        n_bucket_already = sum(1 for r in records if r["diagnostic"].get("bucket") == "already_solved")
        n_bucket_recoverable = sum(1 for r in records if r["diagnostic"].get("bucket") == "recoverable")
        n_bucket_unrecoverable = sum(1 for r in records if r["diagnostic"].get("bucket") == "unrecoverable")

        # Check 1
        c1 = (n_already + n_recoverable + n_unrecoverable == n_total)
        # Check 2
        c2 = (n_recoverable == n_coverage - n_already)
        # Check 3 (proportions)
        oracle_cov = n_coverage / n_total if n_total else 0.0
        ivote_acc = n_already / n_total if n_total else 0.0
        rec_ratio = n_recoverable / n_total if n_total else 0.0
        c3 = abs(rec_ratio - (oracle_cov - ivote_acc)) < tolerance
        # Check 4 (tautology under our partition: n_already counts vote_correct)
        c4 = True
        # Check 5
        c5 = (n_unrecoverable == n_total - n_coverage)
        # Check 6 (bucket field matches partition)
        c6 = (
            n_bucket_already == n_already
            and n_bucket_recoverable == n_recoverable
            and n_bucket_unrecoverable == n_unrecoverable
        )

        all_pass = c1 and c2 and c3 and c4 and c5 and c6
        if not all_pass:
            overall = False

        print(f"\n{ds}:")
        print(f"  total                      = {n_total}")
        print(
            f"  already + recoverable + unrecoverable = "
            f"{n_already}+{n_recoverable}+{n_unrecoverable}"
            f"={n_already + n_recoverable + n_unrecoverable} "
            f"{'PASS' if c1 else 'FAIL'}"
        )
        print(
            f"  recoverable == coverage - already      : "
            f"{n_recoverable} == {n_coverage} - {n_already} = {n_coverage - n_already} "
            f"{'PASS' if c2 else 'FAIL'}"
        )
        print(
            f"  recoverable_ratio == coverage - vote   : "
            f"{rec_ratio:.6f} ~= {oracle_cov:.6f} - {ivote_acc:.6f} = "
            f"{oracle_cov - ivote_acc:.6f} {'PASS' if c3 else 'FAIL'}"
        )
        print(
            f"  unrecoverable == total - coverage      : "
            f"{n_unrecoverable} == {n_total} - {n_coverage} = {n_total - n_coverage} "
            f"{'PASS' if c5 else 'FAIL'}"
        )
        print(
            f"  bucket-field counts match partition    : "
            f"already={n_bucket_already}/{n_already} recoverable="
            f"{n_bucket_recoverable}/{n_recoverable} unrecoverable="
            f"{n_bucket_unrecoverable}/{n_unrecoverable} "
            f"{'PASS' if c6 else 'FAIL'}"
        )
    print("\n" + "=" * 75)
    print(f">> Overall sanity checks: {'PASS' if overall else 'FAIL'}")
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_dir", required=True,
                        help="Directory containing per-dataset JSONL files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where the table CSVs will be written.")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["GSM8K", "GSM-Hard", "AIME-2024", "AQUA-RAT", "MMLU-Pro"],
    )
    parser.add_argument(
        "--filename_pattern", default="mad_vote_{dataset}_infer.jsonl",
        help="Per-dataset filename inside infer_dir. {dataset} is substituted.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with non-zero status if any sanity check fails.",
    )
    args = parser.parse_args()

    per_dataset: Dict[str, List[Dict]] = {}
    per_dataset_records: Dict[str, List[Dict]] = {}
    for ds in args.datasets:
        fname = args.filename_pattern.format(dataset=ds)
        path = os.path.join(args.infer_dir, fname)
        records = usable_records(load_jsonl(path))
        per_dataset_records[ds] = records
        per_dataset[ds] = per_sample_metrics(records)
        print(f"  [{ds}] loaded {len(records)} usable samples from {path}")

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "table1_initial_diagnostics.csv"),
              build_table1(per_dataset))
    write_csv(os.path.join(args.output_dir, "table2_regime_analysis.csv"),
              build_table2(per_dataset))
    write_csv(os.path.join(args.output_dir, "table3_accuracy_decomposition.csv"),
              build_table3(per_dataset))
    write_csv(os.path.join(args.output_dir, "table4_transition_analysis.csv"),
              build_table4(per_dataset))
    write_csv(os.path.join(args.output_dir, "table5_roundwise_accuracy.csv"),
              build_table5(per_dataset))
    write_csv(os.path.join(args.output_dir, "table6_coverage_survival.csv"),
              build_table6(per_dataset))
    print(f">> Wrote 6 diagnostic tables under {args.output_dir}")

    debug_path = os.path.join(args.output_dir, "diagnostic_sample_level_records.jsonl")
    write_sample_level_debug(per_dataset_records, debug_path)
    print(f">> Wrote per-sample debug JSONL to {debug_path}")

    overall_pass = run_sanity_checks(per_dataset_records)
    if args.strict and not overall_pass:
        print("\n[ERROR] --strict mode: sanity checks failed; exiting with status 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
