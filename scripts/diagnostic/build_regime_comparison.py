"""
Assemble the Experiment 1 regime-comparison table (Vanilla MAD vs SCC) from
existing per-(model x method) `_tables/` directories produced by
`scripts/diagnostic/analyze_diagnostic.py`.

For each (model, method) cell we read:
  * `_tables/table2_regime_analysis.csv` Avg row    -> Already Flip,
                                                       Recoverable Recovery,
                                                       Unrecoverable Success
  * `_tables/table6_coverage_survival.csv` Avg row  -> Initial / Final coverage
                                                       (Coverage Drop = I - F)

Output: a 6-row CSV (and matching markdown) with columns
  Model | Method | Already Flip | Recoverable Recovery |
                  Unrecoverable Success | Coverage Drop

Usage (from MASLab/ project root):
  python scripts/diagnostic/build_regime_comparison.py \\
    --mad_runs 1.5B:results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged,\\
3B:results_diagnostic/run_20260501_130540/qwen25-3b-instruct,\\
7B:results_diagnostic/run_20260501_185633/qwen25-7b-instruct \\
    --scc_runs 1.5B:results_diagnostic_scc/run_<TS>/qwen25-1.5b-instruct,\\
3B:results_diagnostic_scc/run_<TS>/qwen25-3b-instruct,\\
7B:results_diagnostic_scc/run_<TS>/qwen25-7b-instruct \\
    --output_csv paper_tables/regime_comparison.csv \\
    --output_md  paper_tables/regime_comparison.md

Each `--mad_runs` / `--scc_runs` entry is `<label>:<dir>`; <dir> must contain
`_tables/table2_regime_analysis.csv` and `_tables/table6_coverage_survival.csv`.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple


def _parse_first_float(cell: str) -> Optional[float]:
    """Parse '7.11' or '7.11 [3.20, 9.50] (n=42)' -> 7.11."""
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
    if not rows:
        return None
    for row in reversed(rows):
        if row and row[0].strip().lower().startswith("avg"):
            return row
    return None


def _load_cell(infer_dir: str) -> Dict[str, Optional[float]]:
    """Return a dict with the four metrics for a single (model, method) cell."""
    t2 = _read_avg_row(os.path.join(infer_dir, "_tables", "table2_regime_analysis.csv"))
    t6 = _read_avg_row(os.path.join(infer_dir, "_tables", "table6_coverage_survival.csv"))

    if t2 is None:
        print(f"[WARN] table2_regime_analysis.csv missing in {infer_dir}/_tables", file=sys.stderr)
    if t6 is None:
        print(f"[WARN] table6_coverage_survival.csv missing in {infer_dir}/_tables", file=sys.stderr)

    flip = _parse_first_float(t2[2]) if t2 and len(t2) > 2 else None
    recovery = _parse_first_float(t2[3]) if t2 and len(t2) > 3 else None
    unrec = _parse_first_float(t2[4]) if t2 and len(t2) > 4 else None

    init_cov = _parse_first_float(t6[2]) if t6 and len(t6) > 2 else None
    final_cov = _parse_first_float(t6[3]) if t6 and len(t6) > 3 else None
    cov_drop = (init_cov - final_cov) if (init_cov is not None and final_cov is not None) else None

    return {
        "flip": flip,
        "recovery": recovery,
        "unrecoverable_success": unrec,
        "coverage_drop": cov_drop,
    }


def _parse_runs_arg(arg: str) -> List[Tuple[str, str]]:
    """`label:dir,label:dir` -> [(label, dir), ...]."""
    out: List[Tuple[str, str]] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Expected `label:dir`, got '{piece}'")
        label, path = piece.split(":", 1)
        out.append((label.strip(), path.strip()))
    return out


def _fmt(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "N/A"


def _build_rows(mad: List[Tuple[str, str]], scc: List[Tuple[str, str]]) -> List[List[str]]:
    mad_map = {label: _load_cell(path) for label, path in mad}
    scc_map = {label: _load_cell(path) for label, path in scc}

    # Row order: model order from the mad_runs argument; for each model,
    # Vanilla MAD then SCC.
    header = [
        "Model", "Method",
        "Already Flip (lower=better)",
        "Recoverable Recovery (higher=better)",
        "Unrecoverable Success (higher=better)",
        "Coverage Drop (lower=better)",
    ]
    rows = [header]
    for label, _ in mad:
        for method, cell_map in (("Vanilla MAD", mad_map), ("SCC", scc_map)):
            cell = cell_map.get(label)
            if cell is None:
                rows.append([label, method, "N/A", "N/A", "N/A", "N/A"])
                continue
            rows.append([
                label, method,
                _fmt(cell["flip"]),
                _fmt(cell["recovery"]),
                _fmt(cell["unrecoverable_success"]),
                _fmt(cell["coverage_drop"]),
            ])
    return rows


def _write_csv(path: str, rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _write_md(path: str, rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if not rows:
            return
        header, *body = rows
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in body:
            f.write("| " + " | ".join(row) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mad_runs", required=True,
        help="Comma-separated `label:dir` for Vanilla MAD diagnostic runs.",
    )
    parser.add_argument(
        "--scc_runs", required=True,
        help="Comma-separated `label:dir` for SCC diagnostic runs.",
    )
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    mad = _parse_runs_arg(args.mad_runs)
    scc = _parse_runs_arg(args.scc_runs)

    rows = _build_rows(mad, scc)
    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows)

    # Echo to stdout for quick scanning.
    for row in rows:
        print(",".join(row))
    print(f">> Wrote {args.output_csv}")
    print(f">> Wrote {args.output_md}")


if __name__ == "__main__":
    main()
