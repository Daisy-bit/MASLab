"""
Re-derive bucket-consistent correctness signals on EXISTING mad_vote JSONL
files, without re-running inference.

Why this exists
---------------
Older mad_vote runs (judge_protocol="xverify") judged each agent on its
**raw response** but judged the per-round vote winner on a synthesised
"The answer is X" sentence. xverify can return different verdicts on
those two inputs, breaking the bucket implication
    vote_correct(X) ==> exists agent i with extracted_i == X AND is_correct_i
which the diagnostic-table math relies on.

This script reads such a JSONL, **re-judges every correctness signal via
xverify on the canonical extracted answer** (cached per sample), and
writes a new JSONL with judge_protocol="xverify-cached-on-canonical".
The new file is fully consistent with bucket invariants and can be fed
straight into scripts/diagnostic/analyze_diagnostic.py.

Inference responses (raw_response, prompt/completion tokens, etc.) are
preserved verbatim -- only correctness fields are recomputed.

Usage
-----
  # rejudge one model's worth of JSONLs
  python scripts/diagnostic/rejudge_diagnostic.py \
      --input_dir  results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct \
      --output_dir results_diagnostic/run_20260501_113709/qwen25-1.5b-instruct_rejudged

  # rejudge a single file
  python scripts/diagnostic/rejudge_diagnostic.py \
      --input_dir  path/to/mad_vote_GSM8K_infer.jsonl \
      --output_dir path/to/rejudged/

After rejudging, run the analyzer with --strict to verify all sanity
checks pass:
  python scripts/diagnostic/analyze_diagnostic.py \
      --infer_dir  <output_dir> \
      --output_dir <output_dir>/_tables \
      --strict
"""

import argparse
import concurrent.futures
import glob
import json
import os
import random
import sys
import threading
from typing import Dict, List, Optional, Tuple

import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Allow `from evaluations....` regardless of cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluations.evaluate_xverify import format_prompt as _xverify_format_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
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


def write_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Canonical answer-only response synthesiser (must match mad_vote_main.py).
# ---------------------------------------------------------------------------

def canonical_response_text(canonical: Optional[str], task_type: str) -> Optional[str]:
    if canonical is None:
        return None
    if task_type == "mcq":
        return f"The answer is ({canonical})."
    return f"The answer is \\boxed{{{canonical}}}."


# ---------------------------------------------------------------------------
# xverify client (thread-safe)
# ---------------------------------------------------------------------------

class XverifyJudge:
    def __init__(
        self,
        model_api_config_path: str,
        model_name: str = "xverify-9b-c",
        max_tokens: int = 64,
        temperature: float = 0.0,
        timeout: int = 600,
    ):
        with open(model_api_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if model_name not in cfg:
            raise RuntimeError(
                f"xverify model '{model_name}' not in {model_api_config_path}"
            )
        self.model_name = model_name
        self.model_list = cfg[model_name]["model_list"]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.lock = threading.Lock()
        self.token_stats = {
            "num_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
    )
    def _call(self, prompt: str) -> str:
        model_dict = random.choice(self.model_list)
        client = openai.OpenAI(
            base_url=model_dict["model_url"], api_key=model_dict["api_key"]
        )
        try:
            comp = client.chat.completions.create(
                model=model_dict["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            out = comp.choices[0].message.content
            n_p = comp.usage.prompt_tokens
            n_c = comp.usage.completion_tokens
        finally:
            client.close()
        with self.lock:
            self.token_stats["num_llm_calls"] += 1
            self.token_stats["prompt_tokens"] += n_p
            self.token_stats["completion_tokens"] += n_c
        return out

    def judge(self, query, response_text, gt) -> Tuple[bool, str]:
        if response_text is None or str(response_text).strip() == "":
            return False, "empty-response"
        prompt = _xverify_format_prompt(query, str(response_text), gt)
        try:
            out = self._call(prompt)
        except Exception as e:  # noqa: BLE001
            return False, f"judge-error:{type(e).__name__}"
        label = (out or "").strip().lower() if isinstance(out, str) else ""
        if label == "correct":
            return True, "ok"
        if label == "incorrect":
            return False, "ok"
        return False, f"unparseable:{label[:60]}"


# ---------------------------------------------------------------------------
# Per-record rejudge
# ---------------------------------------------------------------------------

def rejudge_record(rec: Dict, judge: XverifyJudge) -> Dict:
    """Return a new record with bucket-consistent correctness fields."""
    if "diagnostic" not in rec:
        return rec  # inference-error sample, leave alone

    d = rec["diagnostic"]
    query = rec.get("query")
    gt = rec.get("gt")
    task_type = d.get("task_type", "math")

    # per-sample canonical cache
    cache: Dict[Optional[str], Tuple[bool, str]] = {None: (False, "no-extraction")}

    def grade(canon: Optional[str]) -> Tuple[bool, str]:
        if canon in cache:
            return cache[canon]
        verdict = judge.judge(query, canonical_response_text(canon, task_type), gt)
        cache[canon] = verdict
        return verdict

    # Round 0 -- re-judge per agent on canonical answer
    new_initial = []
    for r in d.get("initial_responses", []):
        ext = r.get("extracted_answer")
        ok, status = grade(ext)
        nr = dict(r)
        nr["is_correct"] = ok
        nr["judge_status"] = status
        new_initial.append(nr)
    initial_extracted = [r["extracted_answer"] for r in new_initial]
    initial_correct = [r["is_correct"] for r in new_initial]
    initial_vote = d.get("initial_vote")
    initial_vote_correct, initial_vote_status = grade(initial_vote)
    initial_oracle = any(initial_correct)

    # Rounds 1..L -- re-judge per agent and per vote winner
    saved_round_vote_history = d.get("round_vote_history", [])
    new_round_responses: Dict[str, List[Dict]] = {}
    round_vote_hist: List[Optional[str]] = [initial_vote]
    round_vote_correct_hist: List[bool] = [initial_vote_correct]
    round_vote_status_hist: List[str] = [initial_vote_status]
    round_extracted_hist: List[List[Optional[str]]] = [list(initial_extracted)]
    round_correct_hist: List[List[bool]] = [list(initial_correct)]

    for r_key in sorted(
        d.get("round_responses", {}).keys(), key=lambda k: int(k)
    ):
        round_recs = d["round_responses"][r_key]
        new_round = []
        for rr in round_recs:
            ext = rr.get("extracted_answer")
            ok, status = grade(ext)
            nrr = dict(rr)
            nrr["is_correct"] = ok
            nrr["judge_status"] = status
            new_round.append(nrr)
        new_round_responses[r_key] = new_round
        ext_r = [x["extracted_answer"] for x in new_round]
        cor_r = [x["is_correct"] for x in new_round]
        # vote winner: trust the saved plurality result (it's a deterministic
        # function of the extracted answers, which we are not modifying).
        try:
            winner = saved_round_vote_history[int(r_key)]
        except (IndexError, ValueError):
            winner = None
        winner_ok, winner_status = grade(winner)
        round_vote_hist.append(winner)
        round_vote_correct_hist.append(winner_ok)
        round_vote_status_hist.append(winner_status)
        round_extracted_hist.append(ext_r)
        round_correct_hist.append(cor_r)

    final_vote = round_vote_hist[-1]
    final_vote_correct = round_vote_correct_hist[-1]
    final_oracle = any(round_correct_hist[-1])

    if initial_vote_correct:
        bucket = "already_solved"
    elif initial_oracle:
        bucket = "recoverable"
    else:
        bucket = "unrecoverable"

    new_d = dict(d)
    new_d["judge_protocol"] = "xverify-cached-on-canonical"
    new_d["judge_model"] = judge.model_name
    new_d["initial_responses"] = new_initial
    new_d["round_responses"] = new_round_responses
    new_d["initial_vote_correct"] = initial_vote_correct
    new_d["initial_vote_judge_status"] = initial_vote_status
    new_d["initial_oracle_coverage"] = initial_oracle
    new_d["final_vote"] = final_vote
    new_d["final_vote_correct"] = final_vote_correct
    new_d["final_oracle_coverage"] = final_oracle
    new_d["round_vote_history"] = round_vote_hist
    new_d["round_vote_correct_history"] = round_vote_correct_hist
    new_d["round_vote_judge_status"] = round_vote_status_hist
    new_d["round_extracted_history"] = round_extracted_hist
    new_d["round_correct_history"] = round_correct_hist
    new_d["bucket"] = bucket
    new_d["judge_cache_keys"] = [
        ("__none__" if k is None else str(k)) for k in cache.keys()
    ]

    new_rec = dict(rec)
    new_rec["diagnostic"] = new_d
    new_rec["response"] = final_vote if final_vote is not None else ""
    return new_rec


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def rejudge_file(in_path: str, out_path: str, judge: XverifyJudge,
                 max_workers: int = 8) -> None:
    records = load_jsonl(in_path)
    n = len(records)
    print(f"  [{os.path.basename(in_path)}] {n} records to rejudge")
    new_records: List[Optional[Dict]] = [None] * n

    if max_workers <= 1:
        for i, r in enumerate(tqdm(records, desc="rejudge")):
            new_records[i] = rejudge_record(r, judge)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(rejudge_record, r, judge): i
                       for i, r in enumerate(records)}
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc="rejudge"):
                idx = futures[fut]
                new_records[idx] = fut.result()

    write_jsonl(out_path, new_records)
    print(f"  -> wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", required=True,
        help="Directory with mad_vote_<dataset>_infer.jsonl files, or a single file path.",
    )
    ap.add_argument(
        "--output_dir", required=True,
        help="Where to write the re-judged JSONL file(s). Filenames preserved.",
    )
    ap.add_argument(
        "--model_api_config",
        default=os.path.join("model_api_configs", "model_api_config.json"),
    )
    ap.add_argument("--xverify_model_name", default="xverify-9b-c")
    ap.add_argument("--xverify_max_tokens", type=int, default=64)
    ap.add_argument("--xverify_temperature", type=float, default=0.0)
    ap.add_argument(
        "--max_workers", type=int, default=8,
        help="Parallelism over samples (each sample serialises its own xverify calls).",
    )
    ap.add_argument(
        "--filename_pattern", default="mad_vote_*_infer.jsonl",
        help="Glob pattern within --input_dir when it is a directory.",
    )
    args = ap.parse_args()

    judge = XverifyJudge(
        args.model_api_config,
        model_name=args.xverify_model_name,
        max_tokens=args.xverify_max_tokens,
        temperature=args.xverify_temperature,
    )

    if os.path.isfile(args.input_dir):
        in_files = [args.input_dir]
    else:
        in_files = sorted(glob.glob(os.path.join(args.input_dir, args.filename_pattern)))
    if not in_files:
        print(f"[ERROR] no files matched in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    for inf in in_files:
        outf = os.path.join(args.output_dir, os.path.basename(inf))
        rejudge_file(inf, outf, judge, max_workers=args.max_workers)

    print(f"\n>> Rejudge done. xverify token stats: {judge.token_stats}")
    print(f">> Next step: run analyzer on {args.output_dir} with --strict.")


if __name__ == "__main__":
    main()
