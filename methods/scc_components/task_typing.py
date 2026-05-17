"""Task-type detection (math / mcq / open).

Lifted verbatim from soo_centered_v3_main.py:64-174 so any baseline can
detect the task type the same way.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# Source tags we treat as math (numeric / symbolic answer).
_MATH_SOURCES = {
    "gsm8k",
    "gsm-hard",
    "gsmhard",
    "gsm_hard",
    "aime-2024",
    "aime",
    "math",
    "math-500",
}

# Source tags that are always multiple choice.
_MCQ_SOURCES = {
    "mmlu-pro",
    "mmlu",
    "gpqa",
}

# Source tags that produce code (HumanEval, MBPP). All routed through the
# `code` task type so plurality uses BLEU + body normalisation and the final
# canonical is returned verbatim (executable function).
_CODE_SOURCES = {
    "humaneval",
    "human-eval",
    "mbpp",
    "mbpp-500",
}

# Regex: detect in-query MCQ options like "(A)...(B)..." or "A)... B)..."
_MCQ_OPTION_RE = re.compile(r"(?:^|\n|\s)\(?([A-J])\)\s*\S")


def _query_looks_like_mcq(query: str) -> bool:
    if not query:
        return False
    letters = set()
    for m in _MCQ_OPTION_RE.finditer(query):
        letters.add(m.group(1).upper())
    return len(letters) >= 3


def detect_task_type(
    sample: Dict, *, force_task_type: Optional[str] = None
) -> str:
    """Return one of {"math", "mcq", "code", "open"} for the given sample.

    Order of evidence:
      1. force_task_type override (if a recognised value)
      2. explicit num_choices on the sample
      3. source tag membership in _MATH_SOURCES / _MCQ_SOURCES / _CODE_SOURCES
         (for math sources, an MCQ-shaped query body promotes to mcq;
          AQUA-RAT is tagged math in MASLab but is multiple-choice)
      4. presence of `test_list` / `entry_point` on the sample (HumanEval /
         MBPP shape) when source is unset
      5. fallback: sniff query body for option labels
    """
    if isinstance(force_task_type, str) and force_task_type.lower() in (
        "math",
        "mcq",
        "code",
        "open",
    ):
        return force_task_type.lower()

    if "num_choices" in sample and sample.get("num_choices"):
        return "mcq"

    source = str(sample.get("source", "")).strip().lower()
    if source in _MCQ_SOURCES:
        return "mcq"
    if source in _CODE_SOURCES:
        return "code"
    if source in _MATH_SOURCES:
        if _query_looks_like_mcq(sample.get("query", "")):
            return "mcq"
        return "math"

    # Shape-based fallback for code samples without an explicit source tag.
    if sample.get("test_list") or sample.get("entry_point"):
        return "code"

    if _query_looks_like_mcq(sample.get("query", "")):
        return "mcq"
    return "open"
