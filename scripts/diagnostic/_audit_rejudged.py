"""Ad-hoc audit: full re-validation of the rejudged tables and JSONL."""
import csv, json, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    'results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged'
TABLES = os.path.join(ROOT, '_tables')
DATASETS = ['GSM8K', 'GSM-Hard', 'AIME-2024', 'AQUA-RAT', 'MMLU-Pro']


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def usable(records):
    out = []
    for r in records:
        if 'diagnostic' not in r:
            continue
        if r.get('error'):
            continue
        if r['diagnostic'].get('gold_answer') is None:
            continue
        out.append(r)
    return out


def parse_pct(cell):
    m = re.match(r'\s*([0-9.+-]+)', cell)
    return float(m.group(1)) if m else None


def parse_n(cell):
    m = re.search(r'\(n=(\d+)\)', cell)
    return int(m.group(1)) if m else None


def read_csv_rows(path):
    with open(path, encoding='utf-8') as f:
        return [r for r in csv.reader(f)]


def per_dataset_metrics(records):
    n_total = len(records)
    n_already = n_rec = n_unrec = n_cov = 0
    n_final_correct = 0
    flips, recoveries, unrec_succ = [], [], []
    single_acc_per_sample = []
    bucket_counter = {'already_solved': 0, 'recoverable': 0, 'unrecoverable': 0}
    init_cov, final_cov, survival = [], [], []
    for r in records:
        d = r['diagnostic']
        vc = bool(d['initial_vote_correct'])
        cov = bool(d['initial_oracle_coverage'])
        fcov = bool(d.get('final_oracle_coverage'))
        fvc = bool(d['final_vote_correct'])
        ic = [bool(x['is_correct']) for x in d['initial_responses']]
        single_acc_per_sample.append(sum(ic) / len(ic))
        if cov:
            n_cov += 1
        if fvc:
            n_final_correct += 1
        if vc:
            n_already += 1
            flips.append(0 if fvc else 1)
        elif cov:
            n_rec += 1
            recoveries.append(1 if fvc else 0)
        else:
            n_unrec += 1
            unrec_succ.append(1 if fvc else 0)
        bk = d.get('bucket')
        if bk in bucket_counter:
            bucket_counter[bk] += 1
        init_cov.append(1 if cov else 0)
        final_cov.append(1 if fcov else 0)
        if cov:
            survival.append(1 if fcov else 0)

    def avg(xs):
        return (sum(xs) / len(xs) * 100) if xs else float('nan')

    return dict(
        n_total=n_total, n_already=n_already, n_rec=n_rec, n_unrec=n_unrec,
        n_cov=n_cov, n_final_correct=n_final_correct,
        single_acc=avg(single_acc_per_sample),
        vote0_acc=n_already / n_total * 100 if n_total else 0,
        coverage=n_cov / n_total * 100 if n_total else 0,
        recoverable_ratio=n_rec / n_total * 100 if n_total else 0,
        final_acc=n_final_correct / n_total * 100 if n_total else 0,
        flip_rate=avg(flips),
        recovery_rate=avg(recoveries),
        unrec_success=avg(unrec_succ),
        bucket_counter=bucket_counter,
        init_cov=avg(init_cov),
        final_cov=avg(final_cov),
        survival=avg(survival),
    )


def relclose(a, b, tol=0.02):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


print('=' * 30, 'REVALIDATION OF REJUDGED TABLES', '=' * 30)
print('ROOT =', ROOT)

all_ok = True
per_ds_metrics = {}
for ds in DATASETS:
    p = os.path.join(ROOT, f'mad_vote_{ds}_infer.jsonl')
    if not os.path.exists(p):
        print(f'\n[{ds}] missing JSONL')
        all_ok = False
        continue
    recs = usable(load_jsonl(p))
    m = per_dataset_metrics(recs)
    per_ds_metrics[ds] = m
    print(f'\n--- {ds} ---')
    print(f"  N={m['n_total']}  cov={m['n_cov']}  "
          f"already={m['n_already']}  rec={m['n_rec']}  unrec={m['n_unrec']}  "
          f"sum={m['n_already'] + m['n_rec'] + m['n_unrec']}")
    c1 = m['n_already'] + m['n_rec'] + m['n_unrec'] == m['n_total']
    c2 = m['n_rec'] == m['n_cov'] - m['n_already']
    rr = m['recoverable_ratio'] / 100
    diff = (m['coverage'] - m['vote0_acc']) / 100
    c3 = abs(rr - diff) < 1e-9
    c5 = m['n_unrec'] == m['n_total'] - m['n_cov']
    c6 = (m['bucket_counter']['already_solved'] == m['n_already']
          and m['bucket_counter']['recoverable'] == m['n_rec']
          and m['bucket_counter']['unrecoverable'] == m['n_unrec'])
    for name, ok in [('Check1 sum=N', c1), ('Check2 rec=cov-already', c2),
                     ('Check3 ratios', c3), ('Check5 unrec=N-cov', c5),
                     ('Check6 saved bucket==partition', c6)]:
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
        if not ok:
            all_ok = False

print('\n' + '=' * 30, 'CSV vs JSONL cross-check', '=' * 30)

# Table 1
rows = read_csv_rows(os.path.join(TABLES, 'table1_initial_diagnostics.csv'))
mm = []
for row in rows[1:-1]:
    ds = row[0]
    if ds not in per_ds_metrics:
        continue
    m = per_ds_metrics[ds]
    pairs = [
        ('Single Acc', parse_pct(row[2]), m['single_acc']),
        ('Vote0 Acc', parse_pct(row[3]), m['vote0_acc']),
        ('Coverage', parse_pct(row[4]), m['coverage']),
        ('Cov-Vote', parse_pct(row[5]), m['coverage'] - m['vote0_acc']),
        ('RecRatio', parse_pct(row[6]), m['recoverable_ratio']),
    ]
    for name, c, expected in pairs:
        if not relclose(c, expected, 0.5):
            mm.append((ds, 'table1', name, c, expected))
print('Table1 mismatches:', len(mm))
for x in mm: print(' ', x)

# Table 2
rows = read_csv_rows(os.path.join(TABLES, 'table2_regime_analysis.csv'))
mm = []
for row in rows[1:-1]:
    ds = row[0]
    if ds not in per_ds_metrics:
        continue
    m = per_ds_metrics[ds]
    csv_N = int(row[1])
    if csv_N != m['n_total']:
        mm.append((ds, 'table2', 'N', csv_N, m['n_total']))
    n_a = parse_n(row[2]); n_r = parse_n(row[3]); n_u = parse_n(row[4])
    s = (n_a or 0) + (n_r or 0) + (n_u or 0)
    if s != m['n_total']:
        mm.append((ds, 'table2', f'partition n_a+n_r+n_u={s}', None, m['n_total']))
    pairs = [
        ('flip', parse_pct(row[2]), m['flip_rate']),
        ('recovery', parse_pct(row[3]), m['recovery_rate']),
        ('unrec_s', parse_pct(row[4]), m['unrec_success']),
        ('final', parse_pct(row[5]), m['final_acc']),
    ]
    for name, c, expected in pairs:
        if not relclose(c, expected, 0.5):
            mm.append((ds, 'table2', name, c, expected))
print('Table2 mismatches:', len(mm))
for x in mm: print(' ', x)

# Table 3
rows = read_csv_rows(os.path.join(TABLES, 'table3_accuracy_decomposition.csv'))
mm = []
for row in rows[1:-1]:
    ds = row[0]
    if ds not in per_ds_metrics:
        continue
    m = per_ds_metrics[ds]
    pairs = [
        ('Single', parse_pct(row[2]), m['single_acc']),
        ('Vote0', parse_pct(row[3]), m['vote0_acc']),
        ('MAD', parse_pct(row[4]), m['final_acc']),
        ('Gain', parse_pct(row[5]), m['final_acc'] - m['vote0_acc']),
    ]
    for name, c, expected in pairs:
        if not relclose(c, expected, 0.5):
            mm.append((ds, 'table3', name, c, expected))
print('Table3 mismatches:', len(mm))
for x in mm: print(' ', x)

# Table 6
rows = read_csv_rows(os.path.join(TABLES, 'table6_coverage_survival.csv'))
mm = []
for row in rows[1:-1]:
    ds = row[0]
    if ds not in per_ds_metrics:
        continue
    m = per_ds_metrics[ds]
    pairs = [
        ('init_cov', parse_pct(row[2]), m['init_cov']),
        ('final_cov', parse_pct(row[3]), m['final_cov']),
        ('survival', parse_pct(row[4]), m['survival']),
    ]
    for name, c, expected in pairs:
        if not relclose(c, expected, 0.5):
            mm.append((ds, 'table6', name, c, expected))
print('Table6 mismatches:', len(mm))
for x in mm: print(' ', x)

print()
print('=' * 30, ('OVERALL: PASS' if all_ok else 'OVERALL: FAIL'), '=' * 30)
