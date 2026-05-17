"""Generic 3-seed aggregator for any (method, variant) combination.

Reads <method_root>/run_1.5B_run{1,2,3}/<variant>/_tables/*.csv,
drops AIME-2024, recomputes Avg per run, and writes mean ± std into
<method_root>/avg_<variant>/_tables/. Mirrors results_mad_scc/aggregate_a0_vanilla.py
in numeric handling but is parameterized.

Usage:
    python scripts/diagnostic/aggregate_variant.py \
        --method_root results_soo_scc --variant A1_triggering
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

TABLE_FILES = [
    "table1_initial_diagnostics.csv",
    "table2_regime_analysis.csv",
    "table3_accuracy_decomposition.csv",
    "table4_transition_analysis.csv",
    "table5_roundwise_accuracy.csv",
    "table6_coverage_survival.csv",
]

EXCLUDE_DATASETS = {"AIME-2024"}

VAL_N_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*\(n=(\d+)\)\s*$")
NUM_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f) if row]


def _parse_numeric(cell: str) -> float | None:
    cell = cell.strip()
    m = VAL_N_RE.match(cell)
    if m:
        return float(m.group(1))
    if NUM_RE.match(cell):
        return float(cell)
    return None


def drop_excluded(rows: list[list[str]]) -> list[list[str]]:
    return [r for r in rows if r[0].strip() not in EXCLUDE_DATASETS]


def recompute_avg_row(rows: list[list[str]]) -> None:
    header = rows[0]
    dataset_rows = rows[1:-1]
    avg_row = rows[-1][:]
    for c in range(2, len(header)):
        vals: list[float] = []
        for r in dataset_rows:
            v = _parse_numeric(r[c])
            if v is None:
                vals = []
                break
            vals.append(v)
        if vals:
            avg_row[c] = f"{sum(vals) / len(vals):.2f}"
    rows[-1] = avg_row


def fmt_mean_std(values: list[float], decimals: int = 2) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def aggregate_cell(cells: list[str]) -> str:
    cells = [c.strip() for c in cells]
    if len(set(cells)) == 1:
        return cells[0]

    parsed = [VAL_N_RE.match(c) for c in cells]
    if all(parsed):
        vals = [float(m.group(1)) for m in parsed]
        ns = [m.group(2) for m in parsed]
        n_part = ns[0] if len(set(ns)) == 1 else "/".join(ns)
        return f"{fmt_mean_std(vals)} (n={n_part})"

    if all(NUM_RE.match(c) for c in cells):
        vals = [float(c) for c in cells]
        if all(float(c).is_integer() for c in cells):
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if std == 0:
                return f"{int(mean)}"
            return f"{mean:.2f} ± {std:.2f}"
        return fmt_mean_std(vals)

    return " / ".join(cells)


def aggregate_table(filename: str, run_dirs: list[Path], variant: str) -> list[list[str]]:
    tables = [read_csv(rd / variant / "_tables" / filename) for rd in run_dirs]

    for i, t in enumerate(tables):
        tables[i] = drop_excluded(t)
        recompute_avg_row(tables[i])

    n_rows = len(tables[0])
    n_cols = len(tables[0][0])
    for t in tables[1:]:
        assert len(t) == n_rows, f"row count mismatch in {filename}"
        assert len(t[0]) == n_cols, f"col count mismatch in {filename}"

    out: list[list[str]] = [tables[0][0]]
    for r in range(1, n_rows):
        new_row: list[str] = []
        for c in range(n_cols):
            cells = [tables[i][r][c] for i in range(len(tables))]
            new_row.append(aggregate_cell(cells))
        out.append(new_row)
    return out


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_root", required=True,
                    help="e.g. results_mad_scc or results_soo_scc")
    ap.add_argument("--variant", required=True,
                    help="e.g. A1_triggering / A2_routing / A5_trig_rout")
    ap.add_argument("--runs", nargs="+",
                    default=["run_1.5B_run1", "run_1.5B_run2", "run_1.5B_run3"])
    args = ap.parse_args()

    method_root = Path(args.method_root).resolve()
    run_dirs = [method_root / r for r in args.runs]
    out_dir = method_root / f"avg_{args.variant}" / "_tables"

    for fname in TABLE_FILES:
        rows = aggregate_table(fname, run_dirs, args.variant)
        write_csv(out_dir / fname, rows)
        print(f"wrote {out_dir / fname}")


if __name__ == "__main__":
    main()
