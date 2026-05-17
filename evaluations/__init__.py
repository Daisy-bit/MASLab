from .evaluate_xverify import eval_func_xverify
from .evaluate_code import eval_func_code

# Datasets that always require code execution regardless of --eval_protocol.
_CODE_DATASETS = {"HumanEval", "MBPP", "MBPP-500"}


def get_eval_func(eval_protocol, tested_dataset_name):
    """Return the evaluator function for (protocol, dataset).

    Code datasets (HumanEval / MBPP / MBPP-500) always route through
    eval_func_code (pass@1 via subprocess test execution), even when the
    user passes --eval_protocol xverify — xverify cannot grade code.
    """
    if tested_dataset_name in _CODE_DATASETS:
        return eval_func_code
    if eval_protocol == "code":
        return eval_func_code
    if eval_protocol == "xverify":
        return eval_func_xverify
    raise ValueError(f"Unsupported evaluation protocol: {eval_protocol}")