"""Symbolic plurality voting + answer extraction + final formatting.

Mirrors soo_centered_v3_main.py:263-368 so any baseline can vote with the
same task-aware equivalence relation and the same count-first lex score.

Always count-first lex: winner = max(group, key=(size, sum_contrib)).
Contribution acts only as a tiebreaker — never overrides numerical
majority. This is the contract that distinguishes the reference v3
behaviour from the failed mad_vote_scc weight-first variant.

Task types and their equivalence relations:
  math → soo_math.is_equiv (normalisation + numeric fallback)
  mcq  → exact letter equality (case-insensitive)
  code → BLEU >= CODE_THRESHOLD on the function body (signature stripped via
         the function's AST; syntactically invalid candidates filtered out).
         Gated by `enable_plurality_for_code` so SCC variants that opt out
         of plurality on code short-circuit at zero cost.
  open → no plurality (returns ("", 0))
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Sequence, Tuple

from methods.soo_math.math_answer_utils import (
    extract_math_answer,
    is_equiv as math_is_equiv,
    strip_string,
)

# Reuse dylan's verified helpers rather than re-implementing.
from methods.dylan.utils_humaneval import (
    CODE_THRESHOLD,
    extract_last_python_code_block,
    parse_code_completion,
    py_is_syntax_valid,
)

# sacrebleu is already a dylan dependency.
from sacrebleu import sentence_bleu


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


def _extract_code(reply: str) -> str:
    """Extract the assistant's code as a single executable string.

    Falls back through dylan's parse_code_completion so that responses
    without a ```python``` fence but with an inline `def` are still
    captured. Returns "" when nothing extractable is found.
    """
    if not reply:
        return ""
    # parse_code_completion's second arg is the *question* and is only used
    # to prepend a signature when the reply contains zero `def` lines. We
    # pass "" because reconstructing the signature is unsafe here (we don't
    # have the prompt) — agents that omit the signature will simply be
    # filtered out by py_is_syntax_valid at the plurality stage.
    code = parse_code_completion(reply, "")
    return code if code else ""


def extract_answer(reply: str, task_type: str) -> str:
    """Task-aware answer extraction.

    math → soo_math.extract_math_answer
    mcq  → anchored "the answer is X" / "(X)" / line-start letter chain
    code → last ```python``` block (with parse_code_completion fallback)
    open → "" (no extraction)
    """
    if task_type == "math":
        return extract_math_answer(reply) or ""
    if task_type == "mcq":
        return _extract_mcq_letter(reply)
    if task_type == "code":
        return _extract_code(reply)
    return ""


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def mcq_is_equiv(a: str, b: str) -> bool:
    """Case-insensitive letter equality for MCQ answers."""
    return bool(a) and bool(b) and a.strip().upper() == b.strip().upper()


def _strip_def_to_body(code: str, entry_point: Optional[str]) -> str:
    """Return the *body* of `entry_point` (signature + docstring stripped).

    Used only to make BLEU comparison robust against agents who restate the
    function signature verbatim (which would inflate similarity scores).
    Falls back to the original code when AST parsing fails or the named
    function is not found — the caller (`_code_bleu_plurality`) separately
    filters syntactically invalid candidates via `py_is_syntax_valid`.
    """
    if not code or not entry_point:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            if not node.body:
                return code
            lines = code.splitlines()
            # body[0].lineno is 1-indexed; AST docstrings are ast.Expr at
            # body[0], which we want to skip so two functions with the same
            # logic but different docstrings still collide.
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
                and len(node.body) > 1
            ):
                first = node.body[1]
            start = first.lineno - 1
            end = node.end_lineno or len(lines)
            body = "\n".join(lines[start:end])
            return body if body.strip() else code
    return code


def code_is_equiv(
    a: str, b: str, *, threshold: float = CODE_THRESHOLD
) -> bool:
    """BLEU-based equivalence on already-normalised function bodies."""
    if not a or not b:
        return False
    try:
        score = sentence_bleu(a, [b], lowercase=True).score
    except Exception:
        return False
    return score >= threshold * 100.0


# ---------------------------------------------------------------------------
# Plurality voting (count-first lex)
# ---------------------------------------------------------------------------

def bleu_cluster_groups(
    extracted_codes: Sequence[Optional[str]],
    *,
    entry_point: Optional[str] = None,
) -> List[List[Tuple[int, str, str]]]:
    """Group already-extracted code candidates by BLEU equivalence.

    Shared between scc_components.count_first_plurality (SCC code path,
    with contribution weighting) and mad_vote.extractor.plurality_vote
    (baseline code path, simple count). Eliminates the previous duplicate
    clustering logic — see also `methods/mad_vote/extractor.py`.

    Each input may be None or empty (failed extraction); those slots are
    skipped. `py_is_syntax_valid` filters parse-failing candidates so they
    cannot pollute the BLEU grouping.

    Args:
        extracted_codes: Per-agent extracted code strings (output of
            extract_answer(..., "code")). None / empty / syntactically
            invalid entries are dropped.
        entry_point: Function name to strip to body before BLEU. When None,
            BLEU is computed on the full code (signature included). The
            mad_vote baseline path passes None because the sample's
            entry_point isn't currently plumbed to plurality_vote.

    Returns:
        List of groups; each group is a list of (agent_idx, body_for_bleu,
        full_code) tuples. Groups are formed via linear scan + transitive
        equivalence under `code_is_equiv`.
    """
    valid: List[Tuple[int, str, str]] = []
    for i, code in enumerate(extracted_codes):
        if not code or not py_is_syntax_valid(code):
            continue
        body = _strip_def_to_body(code, entry_point) if entry_point else code
        if not body:
            continue
        valid.append((i, body, code))

    groups: List[List[Tuple[int, str, str]]] = []
    for item in valid:
        body = item[1]
        placed = False
        for g in groups:
            if code_is_equiv(body, g[0][1]):
                g.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def _code_bleu_plurality(
    extracted_codes: Sequence[Optional[str]],
    contributions: Sequence[float],
    *,
    enable_contribution_aggregation: bool,
    entry_point: Optional[str],
) -> Tuple[str, int]:
    """A0 baseline / SCC final-aggregation path: BLEU plurality on
    syntactically valid candidates with optional contribution weighting.

    `extracted_codes` must be already-extracted code (post extract_answer).
    Returns the winning group's representative (full executable code with
    signature) and its size. When `enable_contribution_aggregation=False`
    (A0 baseline default), the representative is the first agent in the
    group (deterministic). When True, the highest-contribution member is
    used. Returns ("", 0) when no valid candidate exists.
    """
    groups = bleu_cluster_groups(extracted_codes, entry_point=entry_point)
    if not groups:
        return "", 0

    def score(g):
        size = len(g)
        if not enable_contribution_aggregation:
            return (size, -groups.index(g))
        w = float(sum(contributions[i] for i, _, _ in g))
        return (size, w)

    best = max(groups, key=score)
    if enable_contribution_aggregation:
        rep = max(best, key=lambda t: contributions[t[0]])
    else:
        rep = best[0]
    return rep[2], len(best)


def count_first_plurality(
    answers: List[str],
    contributions: Sequence[float],
    task_type: str,
    *,
    enable_contribution_aggregation: bool = True,
    enable_plurality_for_code: bool = True,
    entry_point: Optional[str] = None,
) -> Tuple[str, int]:
    """Group equivalent extracted answers and return (canonical_winner, size).

    Groups are formed by linear scan + transitive equivalence under
      - math_is_equiv (math),
      - mcq_is_equiv  (mcq),
      - code_is_equiv on stripped function body (code) — only when
        `enable_plurality_for_code=True`; SCC variants pass False to skip
        plurality entirely on code,
      - no grouping at all (open) → returns ("", 0).

    Score per group:
      enable_contribution_aggregation=True  → (size, sum_contrib)
      enable_contribution_aggregation=False → (size, -group_index)  # deterministic
    Always lexicographic max with size as primary key, so a numerical
    majority always wins; contribution is only a tiebreaker.

    Returns:
      canonical: the first-seen extracted answer of the winning group, with
                 strip_string() applied for math (so "0.5" / "1/2" collapse
                 to one canonical form). For code, the full executable
                 function (signature preserved) is returned verbatim.
                 Empty string when nothing extracts.
      size:      the agent count in the winning group.
    """
    n = len(answers)
    if n == 0 or task_type == "open":
        return "", 0

    if task_type == "code":
        if not enable_plurality_for_code:
            return "", 0
        # Extract code from raw responses first so the shared clustering
        # helper takes already-extracted strings (same contract as the
        # math/mcq branch below). _extract_code returns None on empty
        # input but possibly the raw text on no-fence input; the cluster
        # helper filters by py_is_syntax_valid downstream.
        extracted_codes = [_extract_code(a) for a in answers]
        return _code_bleu_plurality(
            extracted_codes,
            contributions,
            enable_contribution_aggregation=enable_contribution_aggregation,
            entry_point=entry_point,
        )

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
    code → canonical (verbatim; already an executable function)
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
