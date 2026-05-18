"""Code-task diagnostic path: grade_code_sample contracts + analyze_diagnostic
compatibility.

Two test groups:

1. grade_code_sample edge cases (the subprocess-based code grader that
   replaces xverify when task_type == "code" and emit_diagnostic == True).

2. analyze_diagnostic.py end-to-end on a synthesised code-task diagnostic
   JSONL — verifies the math/MCQ analyzer accepts code-task records
   without modification (no task_type-specific branches needed).

Run:
  pytest tests/test_code_diagnostic.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluations.evaluate_code import grade_code_sample


# ---------------------------------------------------------------------------
# Group 1: grade_code_sample contracts
# ---------------------------------------------------------------------------

def test_humaneval_correct():
    ok, status = grade_code_sample(
        "def add(a,b):\n    return a+b",
        entry_point="add",
        test="def check(candidate):\n    assert candidate(1,2)==3\n",
    )
    assert ok is True
    assert status == "ok"


def test_humaneval_wrong():
    ok, status = grade_code_sample(
        "def add(a,b):\n    return a-b",
        entry_point="add",
        test="def check(candidate):\n    assert candidate(1,2)==3\n",
    )
    assert ok is False
    assert status.startswith("fail:")


def test_mbpp_correct():
    ok, status = grade_code_sample(
        "def add(a,b):\n    return a+b",
        test_list=["assert add(1,2)==3", "assert add(5,5)==10"],
    )
    assert ok is True
    assert status == "ok"


def test_mbpp_one_failing_assert():
    ok, status = grade_code_sample(
        "def add(a,b):\n    return a+b",
        test_list=["assert add(1,2)==3", "assert add(1,2)==99"],
    )
    assert ok is False
    assert status.startswith("fail:")


def test_syntax_invalid():
    ok, status = grade_code_sample(
        "def f(:\n    pass",
        test_list=["assert True"],
    )
    assert ok is False
    assert status == "syntax-invalid"


def test_empty_response():
    for empty in ("", None, "   "):
        ok, status = grade_code_sample(empty, test_list=["assert True"])
        assert ok is False
        assert status == "empty-response"


def test_no_test_provided():
    ok, status = grade_code_sample("def f(): return 1")
    assert ok is False
    assert status == "no-test"


def test_mbpp_with_setup():
    ok, status = grade_code_sample(
        "def double(x):\n    return CONST * x",
        test_list=["assert double(3) == 6"],
        test_setup_code="CONST = 2",
    )
    assert ok is True
    assert status == "ok"


# ---------------------------------------------------------------------------
# Group 2: analyze_diagnostic.py end-to-end on synthetic code diagnostic
# ---------------------------------------------------------------------------

def _synth_diagnostic_record(
    sample_idx: int,
    *,
    initial_correct: list,
    final_correct: bool,
    initial_vote_correct: bool,
):
    """Build a single record matching the mad_vote diagnostic schema for code."""
    n_agents = len(initial_correct)
    initial_records = [
        {
            "agent_id": i,
            "role": "Algebraic-Reasoner",
            "raw_response": f"```python\ndef f_{sample_idx}_{i}():\n    return {i}\n```",
            "extracted_answer": f"def f_{sample_idx}_{i}():\n    return {i}",
            "is_correct": bool(initial_correct[i]),
            "judge_status": "ok",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "tokens": 150,
        }
        for i in range(n_agents)
    ]
    initial_vote = initial_records[0]["extracted_answer"]
    round_responses = {
        "1": [
            {
                **rec,
                "raw_response": rec["raw_response"] + " r1",
                "is_correct": bool(final_correct),
                "judge_status": "ok",
            }
            for rec in initial_records
        ]
    }
    final_vote = initial_vote

    if initial_vote_correct:
        bucket = "already_solved"
    elif any(initial_correct):
        bucket = "recoverable"
    else:
        bucket = "unrecoverable"

    return {
        "query": f"task_{sample_idx}",
        "gt": "code-canonical-placeholder",
        "entry_point": f"f_{sample_idx}",
        "test_list": [f"assert f_{sample_idx}()=={sample_idx}"],
        "source": "HumanEval",
        "tag": ["HumanEval", "code"],
        "response": initial_records[0]["raw_response"],
        "diagnostic": {
            "dataset": "HumanEval",
            "task_type": "code",
            "agents_num": n_agents,
            "rounds_num": 1,
            "judge_protocol": "xverify-cached-on-canonical",
            "judge_model": "xverify-9b-c",
            "gold_answer": "code-canonical-placeholder",
            "initial_responses": initial_records,
            "round_responses": round_responses,
            "initial_vote": initial_vote,
            "initial_vote_correct": initial_vote_correct,
            "initial_vote_judge_status": "ok",
            "initial_vote_counts": {initial_vote: 1, "__none__": 0},
            "final_vote": final_vote,
            "final_vote_correct": final_correct,
            "initial_oracle_coverage": any(initial_correct),
            "final_oracle_coverage": final_correct,
            "round_vote_history": [initial_vote, final_vote],
            "round_vote_correct_history": [initial_vote_correct, final_correct],
            "round_vote_judge_status": ["ok", "ok"],
            "round_extracted_history": [
                [r["extracted_answer"] for r in initial_records],
                [r["extracted_answer"] for r in initial_records],
            ],
            "round_correct_history": [
                list(initial_correct),
                [final_correct] * n_agents,
            ],
            "bucket": bucket,
            "total_tokens": 1500,
            "judge_cache_keys": [initial_vote, "__none__"],
        },
        "token_stats": {"qwen": {"num_llm_calls": 6, "prompt_tokens": 600, "completion_tokens": 300}},
    }


def test_analyze_diagnostic_accepts_code_schema(tmp_path):
    """Run analyze_diagnostic.py on a 5-sample code-task JSONL; verify all
    6 expected tables are emitted and parse the strict-mode sanity check.

    Sample mix:
      idx=0: already_solved (initial_vote_correct=True, all agents correct)
      idx=1: recoverable + fixed (initial wrong but coverage, final correct)
      idx=2: recoverable + not fixed
      idx=3: unrecoverable (no agent ever correct)
      idx=4: harmful flip (initial correct, final wrong)
    """
    infer_dir = tmp_path / "A4_all"
    infer_dir.mkdir()
    jsonl = infer_dir / "mad_scc_HumanEval_infer.jsonl"

    records = [
        _synth_diagnostic_record(
            0, initial_correct=[True, True, True], final_correct=True, initial_vote_correct=True
        ),
        _synth_diagnostic_record(
            1, initial_correct=[True, False, False], final_correct=True, initial_vote_correct=False
        ),
        _synth_diagnostic_record(
            2, initial_correct=[True, False, False], final_correct=False, initial_vote_correct=False
        ),
        _synth_diagnostic_record(
            3, initial_correct=[False, False, False], final_correct=False, initial_vote_correct=False
        ),
        _synth_diagnostic_record(
            4, initial_correct=[True, True, True], final_correct=False, initial_vote_correct=True
        ),
    ]
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    tables_dir = infer_dir / "_tables"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "diagnostic" / "analyze_diagnostic.py"),
        "--infer_dir", str(infer_dir),
        "--output_dir", str(tables_dir),
        "--datasets", "HumanEval",
        "--filename_pattern", "mad_scc_{dataset}_infer.jsonl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"analyze_diagnostic.py failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Verify 6 expected diagnostic tables emitted.
    expected = [
        "table1_initial_diagnostics.csv",
        "table2_regime_analysis.csv",
        "table3_accuracy_decomposition.csv",
        "table4_transition_analysis.csv",
        "table5_roundwise_accuracy.csv",
        "table6_coverage_survival.csv",
    ]
    for name in expected:
        path = tables_dir / name
        assert path.exists(), (
            f"missing {name}; only got: {sorted(p.name for p in tables_dir.iterdir())}"
        )
        # Smoke: file is non-empty and parses as CSV with a header row.
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        assert len(lines) >= 2, f"{name} should have at least header + 1 data row"

    # Sample-level diagnostic JSONL should also be produced.
    assert (tables_dir / "diagnostic_sample_level_records.jsonl").exists()
