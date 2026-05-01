"""
Vanilla multi-agent debate with **plurality voting** aggregation, plus the
diagnostic instrumentation required by exp/实验流程.md.

Differences from `methods/llm_debate/llm_debate_main.py`:
  * Aggregation is a per-extracted-answer plurality vote (not an LLM aggregator).
  * Each agent uses a distinct role/system prompt (5 personas).
  * Every initial response and every per-round response is saved with its
    extracted answer, correctness flag, and token count.
  * The sample is bucketed into {already_solved, recoverable, unrecoverable}.

The class still exposes a `response` field on the output dict (the final voted
answer, as a string) so the existing xverify-based evaluator can be used on
the same output file if desired.
"""

import os
from typing import Dict, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..mas_base import MAS
from ..utils import handle_retry_error
from .extractor import (
    answers_equivalent,
    extract_answer,
    extract_gold,
    get_task_type,
    plurality_vote,
)
from .prompts import (
    AGENT_ROLES,
    get_debate_user_prompt,
    get_initial_user_prompt,
)


class MAD_Vote_Main(MAS):
    """5-agent vanilla MAD with majority voting + diagnostic logging."""

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = int(self.method_config.get("agents_num", 5))
        self.rounds_num = int(self.method_config.get("rounds_num", 3))

        # task-type resolution: explicit override -> dataset-name lookup
        explicit = self.method_config.get("task_type", "auto")
        self.dataset_name = general_config.get("test_dataset_name")
        self.task_type = get_task_type(self.dataset_name, explicit)

        # roles: take the first `agents_num` from the role list, cycling if necessary
        if self.agents_num <= len(AGENT_ROLES):
            self.roles = AGENT_ROLES[: self.agents_num]
        else:
            self.roles = [AGENT_ROLES[i % len(AGENT_ROLES)] for i in range(self.agents_num)]

    # ------------------------------------------------------------------
    # LLM call: same as parent but also returns per-call token usage so we
    # can attribute tokens to each (agent, round) cell of the diagnostic.
    # ------------------------------------------------------------------
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def _call_llm_with_usage(self, messages, temperature=None):
        import random

        model_name = self.model_name
        model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        model_name_, model_url, api_key = (
            model_dict["model_name"],
            model_dict["model_url"],
            model_dict["api_key"],
        )

        request_dict = {
            "model": model_name_,
            "messages": messages,
            "max_tokens": self.model_max_tokens,
            "timeout": self.model_timeout,
        }
        if "o1" not in model_name_:
            request_dict["temperature"] = (
                temperature if temperature is not None else self.model_temperature
            )

        llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        try:
            completion = llm.chat.completions.create(**request_dict)
            response = completion.choices[0].message.content
            n_prompt = completion.usage.prompt_tokens
            n_completion = completion.usage.completion_tokens
        finally:
            llm.close()

        if not isinstance(response, str):
            raise ValueError(f"Invalid response from LLM: {response}")

        # update aggregate token stats (mirrors parent)
        if model_name_ not in self.token_stats:
            self.token_stats[model_name_] = {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        self.token_stats[model_name_]["num_llm_calls"] += 1
        self.token_stats[model_name_]["prompt_tokens"] += n_prompt
        self.token_stats[model_name_]["completion_tokens"] += n_completion

        return response, n_prompt, n_completion

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, sample: Dict) -> Dict:
        query = sample["query"]
        gt = sample.get("gt")
        gold = extract_gold(gt, self.task_type)

        # Per-agent conversation histories. Each entry is a list of
        # {role, content} dicts that we append to as rounds progress.
        agent_contexts: List[List[Dict[str, str]]] = []
        for role_def in self.roles:
            ctx = [
                {"role": "system", "content": role_def["system_prompt"]},
                {"role": "user", "content": get_initial_user_prompt(query, self.task_type)},
            ]
            agent_contexts.append(ctx)

        # ------------- Round 0 (initial independent generation) -------------
        initial_records = []
        for i, ctx in enumerate(agent_contexts):
            try:
                resp, n_p, n_c = self._call_llm_with_usage(ctx)
            except Exception as e:
                resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
            ctx.append({"role": "assistant", "content": resp})
            ext = extract_answer(resp, self.task_type)
            initial_records.append(
                {
                    "agent_id": i,
                    "role": self.roles[i]["name"],
                    "raw_response": resp,
                    "extracted_answer": ext,
                    "is_correct": answers_equivalent(ext, gold, self.task_type),
                    "prompt_tokens": n_p,
                    "completion_tokens": n_c,
                    "tokens": n_p + n_c,
                }
            )

        initial_extracted = [r["extracted_answer"] for r in initial_records]
        initial_correct_flags = [r["is_correct"] for r in initial_records]
        initial_vote, initial_vote_counts = plurality_vote(
            initial_extracted, self.task_type
        )
        initial_vote_correct = answers_equivalent(initial_vote, gold, self.task_type)
        initial_oracle_coverage = any(initial_correct_flags)

        # ------------- Rounds 1..rounds_num (vanilla MAD, fully connected) ------
        round_responses: Dict[str, List[Dict]] = {}
        round_vote_history = [initial_vote]
        round_vote_correct_history = [initial_vote_correct]
        # round_extracted[r] holds the round-r extracted answer for each agent
        round_extracted_history: List[List[Optional[str]]] = [list(initial_extracted)]
        round_correct_history: List[List[bool]] = [list(initial_correct_flags)]

        for r in range(1, self.rounds_num + 1):
            # snapshot peers' last-round responses BEFORE updating any context
            last_round_responses = [ctx[-1]["content"] for ctx in agent_contexts]

            this_round_records = []
            for i, ctx in enumerate(agent_contexts):
                peer_responses = [
                    last_round_responses[j] for j in range(len(agent_contexts)) if j != i
                ]
                user_prompt = get_debate_user_prompt(query, peer_responses, self.task_type)
                ctx.append({"role": "user", "content": user_prompt})
                try:
                    resp, n_p, n_c = self._call_llm_with_usage(ctx)
                except Exception as e:
                    resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
                ctx.append({"role": "assistant", "content": resp})
                ext = extract_answer(resp, self.task_type)
                this_round_records.append(
                    {
                        "agent_id": i,
                        "role": self.roles[i]["name"],
                        "raw_response": resp,
                        "extracted_answer": ext,
                        "is_correct": answers_equivalent(ext, gold, self.task_type),
                        "prompt_tokens": n_p,
                        "completion_tokens": n_c,
                        "tokens": n_p + n_c,
                    }
                )

            round_responses[str(r)] = this_round_records
            extracted_r = [rec["extracted_answer"] for rec in this_round_records]
            correct_r = [rec["is_correct"] for rec in this_round_records]
            vote_r, _ = plurality_vote(extracted_r, self.task_type)
            round_vote_history.append(vote_r)
            round_vote_correct_history.append(
                answers_equivalent(vote_r, gold, self.task_type)
            )
            round_extracted_history.append(extracted_r)
            round_correct_history.append(correct_r)

        # ------------- Final vote / final coverage / bucket ----------------
        final_vote = round_vote_history[-1]
        final_vote_correct = round_vote_correct_history[-1]
        final_oracle_coverage = any(round_correct_history[-1])

        if initial_vote_correct:
            bucket = "already_solved"
        elif initial_oracle_coverage:
            bucket = "recoverable"
        else:
            bucket = "unrecoverable"

        total_tokens = sum(
            rec["tokens"] for rec in initial_records
        ) + sum(
            rec["tokens"]
            for rec_list in round_responses.values()
            for rec in rec_list
        )

        diagnostic = {
            "dataset": self.dataset_name,
            "task_type": self.task_type,
            "agents_num": self.agents_num,
            "rounds_num": self.rounds_num,
            "gold_answer": gold,
            "initial_responses": initial_records,
            "round_responses": round_responses,
            "initial_vote": initial_vote,
            "initial_vote_correct": initial_vote_correct,
            "initial_vote_counts": {
                (k if k is not None else "__none__"): v
                for k, v in initial_vote_counts.items()
            },
            "final_vote": final_vote,
            "final_vote_correct": final_vote_correct,
            "initial_oracle_coverage": initial_oracle_coverage,
            "final_oracle_coverage": final_oracle_coverage,
            "round_vote_history": round_vote_history,  # length = rounds_num + 1
            "round_vote_correct_history": round_vote_correct_history,
            "round_extracted_history": round_extracted_history,
            "round_correct_history": round_correct_history,
            "bucket": bucket,
            "total_tokens": total_tokens,
        }

        # `response` is the canonical final answer string -- kept simple for
        # downstream evaluators / logging.
        response_str = final_vote if final_vote is not None else ""

        return {"response": response_str, "diagnostic": diagnostic}
