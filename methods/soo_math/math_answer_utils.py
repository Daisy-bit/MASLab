"""
Math answer normalization / extraction (aligned with DyLAN-Math for fair voting).
"""

import math
import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

import numpy as np


def strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string

    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)
    return string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]

    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b

    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string

    a = string.split("/")[0]
    b = string.split("/")[1]

    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) >= 2
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string

    splits = string.split("\\sqrt")
    new_string = splits[0]

    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr

    return new_string


def extract_math_answer(pred_str: str) -> str:
    if "The answer is " in pred_str:
        pred = pred_str.split("The answer is ")[-1].strip()
    elif "the answer is " in pred_str:
        pred = pred_str.split("the answer is ")[-1].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""

        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()

        a = strip_string(a)
        pred = a
    else:
        pattern = r"-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str)
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""

    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]

    pred = strip_string(pred)

    if "boxed" in pred:
        ans = pred.split("boxed")[-1]
        if ans and ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()

        a = strip_string(a)
        pred = a

    return pred


def _numeric_equiv(str1: str, str2: str) -> bool:
    """
    在 strip 之后尝试按数值比较，对齐 GSM 类数据集中常见标注（如 9.0 vs 9）及多数 LLM 评测可接受情形。
    无法安全解析为数字时返回 False。
    """
    try:
        a = strip_string(str1)
        b = strip_string(str2)
    except Exception:
        return False
    if not a or not b:
        return False
    try:
        da = Decimal(a.replace(",", ""))
        db = Decimal(b.replace(",", ""))
        if da == db:
            return True
    except (InvalidOperation, ValueError):
        pass
    try:
        fa = float(a.replace(",", ""))
        fb = float(b.replace(",", ""))
        return bool(math.isclose(fa, fb, rel_tol=0.0, abs_tol=1e-9))
    except ValueError:
        return False


def is_equiv(str1: Optional[str], str2: Optional[str]) -> bool:
    """
    两答案是否等价：先 strip_string 严格相等，再尝试数值相等（与常见 xverify 对纯数答案的判定更一致）。
    用于 plurality 聚类及 trace 中与 gold 比对。
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        if strip_string(str1) == strip_string(str2):
            return True
    except Exception:
        if str1 == str2:
            return True
        return False
    try:
        if _numeric_equiv(str1, str2):
            return True
    except Exception:
        pass
    return False


def min_offdiag_similarity(similarity_matrix: np.ndarray) -> float:
    """Minimum pairwise similarity over distinct agents (diagonal ignored)."""
    s = np.asarray(similarity_matrix, dtype=np.float64)
    n = s.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    return float(np.min(s[iu]))


def plurality_answer_by_contribution(
    responses: List[str],
    contributions: np.ndarray,
) -> Tuple[str, int]:
    """
    Cluster extracted answers by DyLAN-style equivalence; pick the cluster with
    largest size, tie-break by sum of contribution weights in that cluster.
    Returns (canonical_extracted_answer, best_cluster_size).
    """
    n = len(responses)
    pairs: List[Tuple[int, str]] = []
    for i in range(n):
        a = extract_math_answer(responses[i])
        if a:
            pairs.append((i, a))

    if not pairs:
        return "", 0

    groups: List[List[Tuple[int, str]]] = []
    for i, a in pairs:
        placed = False
        for g in groups:
            if is_equiv(a, g[0][1]):
                g.append((i, a))
                placed = True
                break
        if not placed:
            groups.append([(i, a)])

    def group_score(g: List[Tuple[int, str]]) -> Tuple[int, float]:
        size = len(g)
        w = float(sum(contributions[i] for i, _ in g))
        return size, w

    best = max(groups, key=group_score)
    return strip_string(best[0][1]), len(best)
