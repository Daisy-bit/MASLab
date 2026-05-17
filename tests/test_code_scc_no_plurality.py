"""SCC-variant (A1-A5) behaviour on code task: plurality fully skipped.

Verifies the contract that backs config_code_a{1..5}.yaml:
  * count_first_plurality with enable_plurality_for_code=False on a "code"
    sample short-circuits to ("", 0) regardless of inputs.
  * Math / MCQ plurality is unaffected by the flag (math A0 still works).
  * mad_scc._vote on code with enable_answer_consensus=False returns the
    argmax-contribution agent's full code, not a plurality canonical.
  * mad_scc._trigger_hit on code uses only the spectral signal.

Run:
  pytest tests/test_code_scc_no_plurality.py -v
"""

from __future__ import annotations

import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.scc_components.voting import count_first_plurality
from methods.mad_scc.mad_scc_main import MAD_SCC_Main


def _wrap(code: str) -> str:
    return f"```python\n{code}\n```"


# ---------------------------------------------------------------------------
# count_first_plurality: SCC short-circuit
# ---------------------------------------------------------------------------

def test_code_scc_short_circuit_returns_empty():
    answers = [_wrap("def f(): return 1")] * 3
    canonical, size = count_first_plurality(
        answers,
        contributions=[0.4, 0.4, 0.2],
        task_type="code",
        enable_plurality_for_code=False,
        entry_point="f",
    )
    assert canonical == ""
    assert size == 0


def test_code_scc_short_circuit_regardless_of_inputs():
    # Even with perfect agreement, plurality is skipped.
    code = _wrap("def f():\n    return 1")
    canonical, size = count_first_plurality(
        [code, code, code, code, code],
        contributions=[0.2, 0.2, 0.2, 0.2, 0.2],
        task_type="code",
        enable_plurality_for_code=False,
        entry_point="f",
    )
    assert canonical == ""
    assert size == 0


# ---------------------------------------------------------------------------
# Math / MCQ unaffected by enable_plurality_for_code
# ---------------------------------------------------------------------------

def test_math_plurality_unchanged_by_code_flag():
    canonical, size = count_first_plurality(
        ["The answer is 42", "boxed{42}", "The answer is 13"],
        contributions=[0.4, 0.4, 0.2],
        task_type="math",
        enable_plurality_for_code=False,  # should not affect math
    )
    assert canonical == "42"
    assert size == 2


def test_mcq_plurality_unchanged_by_code_flag():
    canonical, size = count_first_plurality(
        ["The answer is (B)", "(B)", "(C)"],
        contributions=[0.3, 0.4, 0.3],
        task_type="mcq",
        enable_plurality_for_code=False,
    )
    assert canonical == "B"
    assert size == 2


# ---------------------------------------------------------------------------
# mad_scc._vote: SCC code path = argmax contribution
# ---------------------------------------------------------------------------

class _MadStub:
    task_type = "code"
    equiv_protocol = "task_aware"
    enable_answer_consensus = False  # SCC variant
    enable_weighted_aggregation = False
    enable_triggering = True
    variance_consensus_thr = 0.05


def _bound_stub():
    s = _MadStub()
    for name in (
        "_vote",
        "_group_for_diag",
        "_trigger_hit",
        "_legacy_canonical",
    ):
        setattr(s, name, types.MethodType(getattr(MAD_SCC_Main, name), s))
    return s


def test_mad_scc_vote_code_scc_picks_argmax_contribution():
    stub = _bound_stub()
    codes = [
        "def add(a,b):\n    return a+b",
        "def add(a,b):\n    return a+b",
        "def add(a,b):\n    return a-b",
    ]
    contribs = [0.1, 0.2, 0.7]  # agent 2 has highest contribution
    winner, counts, weights = stub._vote(codes, contribs)
    # SCC variant picks argmax(contributions) → agent 2's code (a-b).
    assert "return a-b" in winner
    # The plurality majority would have been agent 0/1 (a+b) — confirm SCC
    # path does NOT use that.
    assert "return a+b" not in winner


def test_mad_scc_vote_code_a0_uses_bleu_plurality():
    stub = _bound_stub()
    stub.enable_answer_consensus = True  # A0 baseline
    codes = [
        "def add(a,b):\n    return a+b",
        "def add(a,b):\n    return a+b",
        "def add(a,b):\n    return a-b",
    ]
    contribs = [0.1, 0.2, 0.7]
    winner, counts, weights = stub._vote(codes, contribs)
    # A0 baseline: plurality wins → a+b (2 votes vs 1).
    assert "return a+b" in winner


def test_mad_scc_vote_code_empty_extracted():
    stub = _bound_stub()
    stub.enable_answer_consensus = False
    winner, counts, weights = stub._vote([None, "", None], [0.1, 0.2, 0.7])
    assert winner is None


# ---------------------------------------------------------------------------
# mad_scc._trigger_hit on code: only spectral, never plurality
# ---------------------------------------------------------------------------

def test_trigger_code_fires_on_spectral_consensus():
    stub = _bound_stub()
    codes = ["def f(): return 1"] * 5  # 5 identical → would trigger plurality
    triggered, reason = stub._trigger_hit(codes, {"trace": 0.01})
    assert triggered is True
    assert reason == "spectral_trace"


def test_trigger_code_no_fire_when_spectral_high():
    stub = _bound_stub()
    codes = ["def f(): return 1"] * 5  # plurality would have fired
    triggered, reason = stub._trigger_hit(codes, {"trace": 0.5})
    # Code path ignores plurality → no trigger despite 5/5 agreement.
    assert triggered is False
    assert reason is None
