"""Code evaluator for HumanEval / MBPP samples.

Runs the model's `response` against the sample's test cases in a subprocess
sandbox (via `methods.mapcoder.func_evaluate.func_exec`). The subprocess
isolation matches dylan / mapcoder's existing protocol — broken or
infinite-loop code times out without poisoning the worker.

Sample schema (one of):
  HumanEval-style:
    sample["query"]       = function signature + docstring
    sample["entry_point"] = function name (e.g. "has_close_elements")
    sample["test"]        = string containing a `def check(candidate): ...`
                            block and a trailing `check(<entry_point>)` call
  MBPP-style:
    sample["query"]       = problem statement (often includes signature)
    sample["entry_point"] = function name
    sample["test_list"]   = list of assert statements
    sample["test_setup_code"] = optional setup string

Returns (eval_content, eval_score) so the existing evaluate.py worker
can write it next to the response. eval_score is 1 (all tests pass), 0
(any test fails or timeout), or None (eval-level error).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Tuple

from methods.dylan.utils_humaneval import parse_code_completion


_HUMANEVAL_DEFAULT_TIMEOUT = 10
_MBPP_DEFAULT_TIMEOUT = 10


def _run_in_subprocess(directory: str, timeout: int) -> Tuple[bool, str]:
    """Cross-platform replacement for mapcoder.func_evaluate.func_exec.

    mapcoder's func_exec has a `test_file` UnboundLocalError on Windows
    (`command = "cd {} && dir && python {}".format(directory, test_file)`
    references an undefined variable). We replicate the Linux branch's
    behaviour — write main.py into `directory`, exec it, time-bound the
    subprocess — using `sys.executable` so we don't depend on `python3`
    being on PATH on Windows.
    """
    main_py = os.path.join(directory, "main.py")
    try:
        process = subprocess.Popen(
            [sys.executable, main_py],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
        )
        try:
            _out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.communicate(timeout=2)
            except Exception:
                pass
            return False, f"Timeout: exceeded {timeout}s"
        if process.returncode == 0:
            return True, "pass"
        err_text = err.decode("utf-8", errors="replace") if err else ""
        # AssertionError on a failing assert is the common path — keep the
        # last non-empty line of the traceback for diagnostic logging.
        last_line = ""
        for line in reversed(err_text.splitlines()):
            if line.strip():
                last_line = line.strip()
                break
        return False, last_line or f"non-zero exit code {process.returncode}"
    except FileNotFoundError as e:
        return False, f"Eval Error: subprocess setup failed: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"Eval Error: {type(e).__name__}: {e}"


def _normalise_response_code(item: dict) -> str:
    """Strip markdown fences from the response and recover the function.

    Reuses dylan's parse_code_completion so the extraction path here
    matches what mad_scc / soo_scc see during inference.
    """
    response = item.get("response") or ""
    if not response:
        return ""
    code = parse_code_completion(response, "") or ""
    return code


def _run_humaneval(code: str, test_block: str, entry_point: str, timeout: int) -> Tuple[bool, str]:
    """HumanEval format: a `def check(candidate)` block + a trailing call."""
    if not entry_point:
        return False, "Eval Error: missing entry_point"
    # The HF HumanEval test field ends with a call like `check(candidate)` or
    # `check(<entry_point>)`. We append both forms for safety.
    runner = (
        f"{code}\n\n"
        f"{test_block}\n\n"
        f"try:\n"
        f"    check({entry_point})\n"
        f"except NameError:\n"
        f"    check(candidate)\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(runner)
        ok, msg = _run_in_subprocess(tmp, timeout=timeout)
    return ok, msg


def _run_mbpp(code: str, test_list, test_setup_code: str, timeout: int) -> Tuple[bool, str]:
    """MBPP format: a list of `assert ...` statements, optional setup."""
    if not isinstance(test_list, (list, tuple)) or not test_list:
        return False, "Eval Error: empty test_list"
    setup = test_setup_code or ""
    runner_parts = [code, setup]
    for t in test_list:
        runner_parts.append(str(t))
    runner = "\n\n".join(p for p in runner_parts if p)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(runner)
        ok, msg = _run_in_subprocess(tmp, timeout=timeout)
    return ok, msg


def eval_func_code(item, llm=None):
    """Evaluate one code sample. `llm` is unused (kept for API symmetry)."""
    try:
        code = _normalise_response_code(item)
        if not code:
            return "Eval Error: empty extracted code", 0

        entry_point = item.get("entry_point", "")
        # HumanEval style takes precedence: if there's a `test` block we run it.
        test_block = item.get("test")
        if isinstance(test_block, str) and test_block.strip():
            ok, msg = _run_humaneval(
                code, test_block, entry_point, timeout=_HUMANEVAL_DEFAULT_TIMEOUT
            )
        else:
            ok, msg = _run_mbpp(
                code,
                item.get("test_list", []),
                item.get("test_setup_code", ""),
                timeout=_MBPP_DEFAULT_TIMEOUT,
            )
        return ("pass" if ok else f"fail: {msg}", 1 if ok else 0)
    except Exception as e:  # noqa: BLE001
        return f"Eval Error: {type(e).__name__}: {e}", None
