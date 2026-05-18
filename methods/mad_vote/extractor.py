"""
Deterministic answer extraction & equivalence checking.

Handles two task types:
  * math : free-form numeric answer (GSM8K, GSM-Hard, AIME-2024, MATH-style).
  * mcq  : single-letter option (AQUA-RAT, MMLU-Pro, etc.).

For diagnostic experiments we cannot afford an LLM judge per agent per round,
so all extraction & comparison is rule-based. The extractor is biased toward
*explicit* answer markers (\\boxed{...}, "the answer is ...", "####") and
falls back to the last number / option letter only if no marker is found.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset -> task type mapping
# ---------------------------------------------------------------------------

DATASET_TASK_TYPE = {
    "GSM8K": "math",
    "GSM-Hard": "math",
    "AIME-2024": "math",
    "AQUA-RAT": "mcq",
    "MMLU-Pro": "mcq",
    "MATH": "math",
    "MedMCQA": "mcq",
    "MMLU": "mcq",
    "SciBench": "math",
    "HumanEval": "code",
    "MBPP": "code",
    "MBPP-500": "code",
}


def get_task_type(dataset_name: Optional[str], explicit: Optional[str] = None) -> str:
    if explicit and explicit != "auto":
        return explicit
    if dataset_name is None:
        return "math"
    return DATASET_TASK_TYPE.get(dataset_name, "math")


# ---------------------------------------------------------------------------
# Math extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content inside the LAST \\boxed{...} (handling balanced braces)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None


def _normalize_number_str(s: str) -> Optional[str]:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    s = s.replace(" ", "")
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f):
        return str(int(f))
    # canonicalize: strip trailing zeros, but keep decimals
    return f"{f:.10f}".rstrip("0").rstrip(".")


def extract_math_answer(text: str) -> Optional[str]:
    """Return a normalized string form of the math answer in `text`, or None."""
    if not text:
        return None

    # 1. \boxed{...} (most reliable)
    boxed = _extract_boxed(text)
    if boxed is not None:
        # boxed content may itself be like "440" or "x=440" -- pick the last number
        nums = _NUMBER_RE.findall(boxed)
        if nums:
            n = _normalize_number_str(nums[-1])
            if n is not None:
                return n
        # otherwise fall through with just the boxed text
        bs = boxed.strip()
        if bs:
            return bs

    # 2. "#### X" (GSM8K-style gold marker)
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        n = _normalize_number_str(m.group(1))
        if n is not None:
            return n

    # 3. "the answer is X" / "answer: X" / "final answer: X"
    for pat in (
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\\?boxed?\{?\s*(-?[\d,\.]+)",
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\(?\s*(-?[\d,\.]+)\s*\)?",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            n = _normalize_number_str(m.group(1))
            if n is not None:
                return n

    # 4. Last number in the LAST non-empty line (trailing-answer convention).
    last_line = ""
    for line in reversed(text.splitlines()):
        if line.strip():
            last_line = line
            break
    nums = _NUMBER_RE.findall(last_line)
    if nums:
        n = _normalize_number_str(nums[-1])
        if n is not None:
            return n

    # 5. Last number anywhere.
    nums = _NUMBER_RE.findall(text)
    if nums:
        n = _normalize_number_str(nums[-1])
        if n is not None:
            return n

    return None


# ---------------------------------------------------------------------------
# MCQ extraction
# ---------------------------------------------------------------------------

def extract_mcq_answer(text: str) -> Optional[str]:
    """Return a single capital letter option (A-Z) or None."""
    if not text:
        return None

    # 1. \boxed{X} or \boxed{(X)}
    boxed = _extract_boxed(text)
    if boxed is not None:
        m = re.search(r"\(?([A-Za-z])\)?", boxed.strip())
        if m:
            return m.group(1).upper()

    # 2. "the answer is (X)" / "answer is X" / "answer: X"
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-Za-z])\)?(?:[\s\.,!?\)]|$)",
        r"answer\s*\.\s*([A-Za-z])\b",
        r"\boption\s*\(?([A-Za-z])\)?\b",
        r"\bchoice\s*\(?([A-Za-z])\)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 3. Last "(X)" in last non-empty line
    last_line = ""
    for line in reversed(text.splitlines()):
        if line.strip():
            last_line = line
            break
    matches = re.findall(r"\(([A-Za-z])\)", last_line)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([A-Za-z])\b", last_line)
    # filter out common non-option words
    cand = [c for c in matches if len(c) == 1 and c.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    if cand:
        return cand[-1].upper()

    return None


# ---------------------------------------------------------------------------
# Code extraction (HumanEval / MBPP)
# ---------------------------------------------------------------------------

def extract_code_answer(text: str) -> Optional[str]:
    """Extract the assistant's Python implementation as a single string.

    Wraps dylan's `parse_code_completion` so we share its ```python``` fence
    detection + signature-recovery fallback. Returns None when nothing
    extractable is found.
    """
    if not text:
        return None
    # Late import to keep mad_vote's import-time dependency on dylan minimal.
    from methods.dylan.utils_humaneval import parse_code_completion
    code = parse_code_completion(text, "") or ""
    return code if code else None


# ---------------------------------------------------------------------------
# Unified extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str, task_type: str) -> Optional[str]:
    if task_type == "mcq":
        return extract_mcq_answer(text)
    if task_type == "code":
        return extract_code_answer(text)
    return extract_math_answer(text)


def extract_gold(gt, task_type: str) -> Optional[str]:
    """Normalize the dataset's gold answer (`gt` field) into a canonical form."""
    if gt is None:
        return None
    if task_type == "code":
        # Code tasks have no single-string gold (correctness comes from
        # executing test_list / test). We return str(gt) — the canonical
        # solution from the HF dataset — as a *placeholder* gold_answer so
        # downstream `analyze_diagnostic.py`'s "is None" filter doesn't
        # drop code samples. The string is never used for grading; grading
        # goes through evaluations.evaluate_code.grade_code_sample.
        return str(gt) if str(gt).strip() else "code-no-string-gold"
    if isinstance(gt, (int, float)):
        if task_type == "mcq":
            return str(gt).strip().upper()
        return _normalize_number_str(str(gt))
    gt_s = str(gt)
    return extract_answer(gt_s, task_type)


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def answers_equivalent(pred: Optional[str], gold: Optional[str], task_type: str) -> bool:
    if pred is None or gold is None:
        return False
    if task_type == "mcq":
        return pred.strip().upper() == gold.strip().upper()
    if task_type == "code":
        # Not used: code correctness is decided by executing test_list, not
        # by string equivalence. Returning False is the safe default so any
        # accidental caller can't silently mark code samples as "correct".
        return False
    # math: try numeric, fall back to string
    pn = _normalize_number_str(pred)
    gn = _normalize_number_str(gold)
    if pn is not None and gn is not None:
        try:
            return abs(float(pn) - float(gn)) < 1e-6
        except ValueError:
            return pn == gn
    return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Plurality voting
# ---------------------------------------------------------------------------

def plurality_vote(answers, task_type: str):
    """
    Plurality vote on a list of (possibly-None) extracted answers.

    Returns (winning_answer, vote_count_dict).
    Ties broken by first-appearance (stable).
    None / empty answers are still counted as a "no-extraction" bucket but never win
    if any real answer received >= 1 vote -- they only win if everyone failed extraction.

    For task_type=="code", grouping uses BLEU-similarity (>= CODE_THRESHOLD)
    on the full extracted code (signature included). The winning group's
    first member's code is returned as the canonical winner. `counts` is
    keyed by the winning representative, not by an abstract key.
    """
    if task_type == "code":
        return _code_plurality_vote(answers)

    counts = {}
    order = []
    for a in answers:
        if a is None:
            key = None
        elif task_type == "mcq":
            key = a.strip().upper()
        else:
            n = _normalize_number_str(a)
            key = n if n is not None else a.strip()
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    real_keys = [k for k in order if k is not None]
    if real_keys:
        # winner: highest count; ties -> earliest in `order`
        best = real_keys[0]
        for k in real_keys[1:]:
            if counts[k] > counts[best]:
                best = k
        return best, counts
    return None, counts


def _code_plurality_vote(extracted_codes):
    """BLEU-based plurality for code candidates (mad_vote baseline path).

    Thin wrapper around `scc_components.voting.bleu_cluster_groups` —
    shares the same clustering logic as soo_scc / mad_scc to prevent the
    two implementations from drifting.

    `extracted_codes` is the output of `extract_answer(reply, "code")` for
    each agent — already-extracted code strings or None. We don't have
    sample.entry_point here, so BLEU is computed on the full extracted
    code (no body stripping). For tighter clustering, callers that DO
    have entry_point should use scc_components.count_first_plurality
    directly.

    Returns (winner_full_code_or_None, counts_dict). counts_dict maps
    each cluster's representative -> cluster size, plus a {None: n} entry
    for failed-extraction / syntactically invalid agents.
    """
    # Late imports keep this module's top-level dependencies small for the
    # math / mcq fast path.
    from methods.scc_components.voting import bleu_cluster_groups
    from methods.dylan.utils_humaneval import py_is_syntax_valid

    groups = bleu_cluster_groups(extracted_codes, entry_point=None)
    none_count = sum(
        1 for a in extracted_codes if not a or not py_is_syntax_valid(a)
    )

    if not groups:
        if none_count:
            return None, {None: none_count}
        return None, {}

    # Winner: largest cluster, ties → earliest seen.
    best = groups[0]
    for g in groups[1:]:
        if len(g) > len(best):
            best = g

    # Counts dict: group representative (full code) → cluster size.
    counts = {g[0][2]: len(g) for g in groups}
    if none_count:
        counts[None] = none_count

    return best[0][2], counts
