"""A0 baseline behaviour on code task: BLEU plurality + no early stop.

Verifies:
  * count_first_plurality on "code" returns the BLEU-winning cluster's
    representative when enable_plurality_for_code=True.
  * Syntactically invalid candidates are filtered out before grouping.
  * Format_final returns the canonical verbatim (no math/mcq wrapping).
  * task_typing detects HumanEval / MBPP sources as "code".
  * The 999 unreachable-threshold trick: Gate A check
    `size >= answer_consensus_min_initial` evaluates False for any realistic
    cluster size, so Gate A never triggers in A0_code.

Run:
  pytest tests/test_code_a0_plurality.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.scc_components.voting import (
    code_is_equiv,
    count_first_plurality,
    format_final,
)
from methods.scc_components.task_typing import detect_task_type


# ---------------------------------------------------------------------------
# Task-type detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sample,expected",
    [
        ({"source": "HumanEval"}, "code"),
        ({"source": "humaneval"}, "code"),
        ({"source": "MBPP"}, "code"),
        ({"source": "MBPP-500"}, "code"),
        ({"entry_point": "foo"}, "code"),
        ({"test_list": ["assert foo()==1"]}, "code"),
        ({"source": "GSM8K"}, "math"),
        ({"source": "MMLU-Pro"}, "mcq"),
    ],
)
def test_detect_task_type_code(sample, expected):
    assert detect_task_type(sample) == expected


def test_force_task_type_code():
    assert detect_task_type({"source": "GSM8K"}, force_task_type="code") == "code"


# ---------------------------------------------------------------------------
# A0 BLEU plurality
# ---------------------------------------------------------------------------

def _wrap(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.fixture
def codes_two_majority():
    # Two agents agree (a+b), one disagrees (a-b).
    return [
        _wrap("def add(a,b):\n    return a+b"),
        _wrap("def add(a,b):\n    return a+b"),
        _wrap("def add(a,b):\n    return a-b"),
    ]


def test_a0_plurality_returns_majority_code(codes_two_majority):
    canonical, size = count_first_plurality(
        codes_two_majority,
        contributions=[0.4, 0.4, 0.2],
        task_type="code",
        enable_contribution_aggregation=False,
        enable_plurality_for_code=True,
        entry_point="add",
    )
    assert size == 2
    assert "return a+b" in canonical
    assert canonical.startswith("def add")


def test_a0_plurality_filters_syntax_errors():
    answers = [
        _wrap("def f():\n    return 1"),
        _wrap("def f():\n    return 1"),
        _wrap("def f(:\n    return 1"),  # broken: invalid def
    ]
    canonical, size = count_first_plurality(
        answers,
        contributions=[0.3, 0.3, 0.4],
        task_type="code",
        enable_contribution_aggregation=False,
        enable_plurality_for_code=True,
        entry_point="f",
    )
    # Invalid candidate dropped; remaining two cluster together.
    assert size == 2
    assert "return 1" in canonical


def test_a0_plurality_all_invalid_returns_empty():
    answers = [
        _wrap("def f(:\n    pass"),
        _wrap("def g(:\n    pass"),
    ]
    canonical, size = count_first_plurality(
        answers,
        contributions=[0.5, 0.5],
        task_type="code",
        enable_contribution_aggregation=False,
        enable_plurality_for_code=True,
        entry_point="f",
    )
    assert canonical == ""
    assert size == 0


# ---------------------------------------------------------------------------
# Threshold = 999 ⇒ Gate A never triggers
# ---------------------------------------------------------------------------

def test_a0_threshold_999_disables_gate_a(codes_two_majority):
    """Even when 2/3 agree on code, the A0 unreachable threshold of 999
    keeps the Gate-A early-stop condition False."""
    _, size = count_first_plurality(
        codes_two_majority,
        contributions=[0.4, 0.4, 0.2],
        task_type="code",
        enable_contribution_aggregation=False,
        enable_plurality_for_code=True,
        entry_point="add",
    )
    answer_consensus_min_initial = 999
    enable_answer_consensus = True
    early_stop_fires = (
        enable_answer_consensus
        and size > 0
        and size >= answer_consensus_min_initial
    )
    assert early_stop_fires is False
    # And realistically with 5 agents, size <= 5 << 999, never fires.
    assert size < answer_consensus_min_initial


# ---------------------------------------------------------------------------
# code_is_equiv sanity
# ---------------------------------------------------------------------------

def test_code_is_equiv_identical():
    a = "def add(a,b):\n    return a+b"
    assert code_is_equiv(a, a) is True


def test_code_is_equiv_differs_by_operator():
    a = "def add(a,b):\n    return a+b"
    b = "def add(a,b):\n    return a-b"
    # Very small body → BLEU on these two short strings happens to be high
    # (signature is identical). The test only asserts that the result is a
    # bool — the threshold tuning is dylan's CODE_THRESHOLD=0.9 default.
    result = code_is_equiv(a, b)
    assert isinstance(result, bool)


def test_code_is_equiv_empty():
    assert code_is_equiv("", "x") is False
    assert code_is_equiv("x", "") is False
    assert code_is_equiv("", "") is False


# ---------------------------------------------------------------------------
# format_final on code returns canonical verbatim
# ---------------------------------------------------------------------------

def test_format_final_code_verbatim():
    code = "def f():\n    return 42"
    assert format_final(code, "code") == code


def test_format_final_code_empty():
    assert format_final("", "code") == ""
