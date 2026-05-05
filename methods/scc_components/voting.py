"""Symbolic plurality voting + answer extraction + final formatting.

Mirrors soo_centered_v3_main.py:263-368 so any baseline can vote with the
same task-aware equivalence relation and the same count-first lex score.

Always count-first lex: winner = max(group, key=(size, sum_contrib)).
Contribution acts only as a tiebreaker — never overrides numerical
majority. This is the contract that distinguishes the reference v3
behaviour from the failed mad_vote_scc weight-first variant.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from methods.soo_math.math_answer_utils import (
    extract_math_answer,
    is_equiv as math_is_equiv,
    strip_string,
)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_ANCHOR = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer[\s:]*|correct\s+answer[\s:]*)"
    r"\s*\(?\s*([A-Ja-j])\s*\)?",
    flags=re.IGNORECASE,
)
_PAREN_LETTER = re.compile(r"\(\s*([A-Ja-j])\s*\)")
_LINE_START_LETTER = re.compile(r"(?:^|\n)\s*([A-Ja-j])\)")


def _extract_mcq_letter(reply: str) -> str:
    if not reply:
        return ""
    last = None
    for m in _ANSWER_ANCHOR.finditer(reply):
        last = m.group(1)
    if last:
        return last.upper()
    for m in _PAREN_LETTER.finditer(reply):
        last = m.group(1)
    if last:
        return last.upper()
    for m in _LINE_START_LETTER.finditer(reply):
        last = m.group(1)
    return last.upper() if last else ""


def extract_answer(reply: str, task_type: str) -> str:
    """Task-aware answer extraction.

    math → soo_math.extract_math_answer
    mcq  → anchored "the answer is X" / "(X)" / line-start letter chain
    open → "" (no extraction)
    """
    if task_type == "math":
        return extract_math_answer(reply) or ""
    if task_type == "mcq":
        return _extract_mcq_letter(reply)
    return ""


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def mcq_is_equiv(a: str, b: str) -> bool:
    """Case-insensitive letter equality for MCQ answers."""
    return bool(a) and bool(b) and a.strip().upper() == b.strip().upper()


# ---------------------------------------------------------------------------
# Plurality voting (count-first lex)
# ---------------------------------------------------------------------------

def count_first_plurality(
    answers: List[str],
    contributions: Sequence[float],
    task_type: str,
    *,
    enable_contribution_aggregation: bool = True,
) -> Tuple[str, int]:
    """Group equivalent extracted answers and return (canonical_winner, size).

    Groups are formed by linear scan + transitive equivalence under
      - math_is_equiv (math),
      - mcq_is_equiv  (mcq),
      - no grouping at all (open) → returns ("", 0).

    Score per group:
      enable_contribution_aggregation=True  → (size, sum_contrib)
      enable_contribution_aggregation=False → (size, -group_index)  # deterministic
    Always lexicographic max with size as primary key, so a numerical
    majority always wins; contribution is only a tiebreaker.

    Returns:
      canonical: the first-seen extracted answer of the winning group, with
                 strip_string() applied for math (so "0.5" / "1/2" collapse
                 to one canonical form). Empty string when nothing extracts.
      size:      the agent count in the winning group.
    """
    n = len(answers)
    if n == 0 or task_type == "open":
        return "", 0

    extracted = [extract_answer(a, task_type) for a in answers]
    pairs = [(i, x) for i, x in enumerate(extracted) if x]
    if not pairs:
        return "", 0

    eq = math_is_equiv if task_type == "math" else mcq_is_equiv

    groups: List[List[Tuple[int, str]]] = []
    for i, x in pairs:
        placed = False
        for g in groups:
            if eq(x, g[0][1]):
                g.append((i, x))
                placed = True
                break
        if not placed:
            groups.append([(i, x)])

    def score(g):
        size = len(g)
        if not enable_contribution_aggregation:
            return (size, -groups.index(g))
        w = float(sum(contributions[i] for i, _ in g))
        return (size, w)

    best = max(groups, key=score)
    canonical = best[0][1]
    if task_type == "math":
        canonical = strip_string(canonical)
    return canonical, len(best)


# ---------------------------------------------------------------------------
# Final formatting
# ---------------------------------------------------------------------------

def format_final(canonical: str, task_type: str) -> str:
    """Format a canonical answer for downstream xverify / display.

    math → "The answer is <c>. \\boxed{<c>}"
    mcq  → "The answer is (<C>)"
    open → canonical (verbatim)
    """
    if not canonical:
        return ""
    if task_type == "math":
        return f"The answer is {canonical}. \\boxed{{{canonical}}}"
    if task_type == "mcq":
        letter = canonical.strip().upper()
        return f"The answer is ({letter})"
    return canonical
