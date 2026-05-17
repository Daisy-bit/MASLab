"""
Role prompts for the 5 debate agents.

Each agent shares the same base model but takes a distinct persona so that
initial responses are diverse enough for plurality voting to be informative.
The prompts intentionally avoid contradicting each other on the answer
format -- they all converge on a final answer at the end of the response.
"""

AGENT_ROLES = [
    {
        "name": "Algebraic-Reasoner",
        "system_prompt": (
            "You are a meticulous algebraic reasoner. Solve the problem by "
            "introducing variables, writing equations, and working through "
            "the algebra step by step. Show your work clearly and state your "
            "final answer at the very end of the response."
        ),
    },
    {
        "name": "Numerical-Verifier",
        "system_prompt": (
            "You are a careful numerical verifier. Solve the problem step by "
            "step, then double-check each arithmetic step before committing. "
            "If a step looks fragile, redo it. State your final answer at "
            "the very end of the response."
        ),
    },
    {
        "name": "Concrete-Example-Reasoner",
        "system_prompt": (
            "You are a concrete-example reasoner. Translate abstract "
            "quantities into specific numbers or worked-through cases, then "
            "generalize. Be explicit about each substitution. State your "
            "final answer at the very end of the response."
        ),
    },
    {
        "name": "Structural-Decomposer",
        "system_prompt": (
            "You are a structural decomposer. Break the problem into named "
            "sub-problems, solve each one, and combine the partial results "
            "into the final answer. State your final answer at the very end "
            "of the response."
        ),
    },
    {
        "name": "Pragmatic-Heuristic-Solver",
        "system_prompt": (
            "You are a pragmatic problem solver who favors short, robust "
            "reasoning chains and sanity-checking with simple heuristics. "
            "Avoid over-elaborate detours. State your final answer at the "
            "very end of the response."
        ),
    },
]


def get_initial_user_prompt(query: str, task_type: str) -> str:
    """Initial round-0 user prompt -- ask for an answer with a clear final-line format."""
    if task_type == "mcq":
        return (
            f"{query}\n\n"
            "Reason step by step. At the very end of your response, on its own "
            "line, write exactly: 'The answer is (X)' where X is the option letter."
        )
    if task_type == "code":
        # The 5 personas (Algebraic-Reasoner, Numerical-Verifier, ...) are
        # math-leaning by design but still produce useful diverse code here.
        # We override only the user-prompt format: restate the signature and
        # respond inside a single ```python``` fence.
        return (
            "You must complete the Python function below. Restate the "
            "function signature, then write your full implementation. "
            "Respond with a single ```python ... ``` fenced code block "
            "containing only the function (and any necessary imports). "
            "Do not include free-flowing prose outside the code block.\n\n"
            f"[Function]\n```python\n{query}\n```"
        )
    return (
        f"{query}\n\n"
        "Reason step by step. At the very end of your response, on its own "
        "line, write exactly: 'The answer is \\boxed{X}' where X is the final answer."
    )


def get_debate_user_prompt(query: str, peer_responses: list, task_type: str) -> str:
    """Round >=1 user prompt -- show peers' previous-round answers and ask for an updated answer."""
    parts = ["These are the most recent responses from other agents:\n"]
    for j, resp in enumerate(peer_responses):
        parts.append(f"\n--- Agent {j + 1} response ---\n{resp}\n")
    if task_type == "mcq":
        tail = (
            "\nUse these opinions carefully as additional advice. "
            "Reconsider your reasoning, then give an updated answer. "
            "At the very end of your response, on its own line, write exactly: "
            "'The answer is (X)' where X is the option letter.\n"
            f"\nThe original problem was:\n{query}"
        )
    elif task_type == "code":
        tail = (
            "\nUse these implementations carefully as additional advice. "
            "They may contain bugs or miss corner cases. Reconsider your "
            "implementation, then give an updated version. Respond with a "
            "single ```python ... ``` fenced code block containing only the "
            "function (signature restated). No prose outside the block.\n"
            f"\nThe original function was:\n```python\n{query}\n```"
        )
    else:
        tail = (
            "\nUse these opinions carefully as additional advice. "
            "Reconsider your reasoning, then give an updated answer. "
            "At the very end of your response, on its own line, write exactly: "
            "'The answer is \\boxed{X}' where X is the final answer.\n"
            f"\nThe original problem was:\n{query}"
        )
    parts.append(tail)
    return "".join(parts)
