"""
Vanilla multi-agent debate with **plurality voting** aggregation, plus the
diagnostic instrumentation required by exp/实验流程.md.

Differences from `methods/llm_debate/llm_debate_main.py`:
  * Aggregation is a per-extracted-answer plurality vote (not an LLM aggregator).
  * Each agent uses a distinct role/system prompt (5 personas).
  * Every initial response and every per-round response is saved with its
    extracted answer, **xverify-LLM-judged** correctness flag, and token count.
  * Per-round vote winners are also judged by xverify (not by deterministic
    string equivalence) so all correctness signals come from the same judge.
  * The sample is bucketed into {already_solved, recoverable, unrecoverable}.

The class still exposes a `response` field on the output dict (the final voted
answer, as a string) for compatibility with the standard evaluator.
"""

import random
from typing import Dict, List, Optional, Tuple

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..mas_base import MAS
from ..utils import handle_retry_error
from .extractor import extract_answer, extract_gold, get_task_type, plurality_vote
from .prompts import AGENT_ROLES, get_debate_user_prompt, get_initial_user_prompt

# Reuse the official xverify prompt format so judgments are identical to the
# standard `evaluate.py --eval_protocol xverify` pipeline.
from evaluations.evaluate_xverify import format_prompt as _xverify_format_prompt


class MAD_Vote_Main(MAS):
    """5-agent vanilla MAD with majority voting + xverify-judged diagnostic logging."""

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = int(self.method_config.get("agents_num", 5))
        self.rounds_num = int(self.method_config.get("rounds_num", 3))

        # task-type resolution: explicit override -> dataset-name lookup
        explicit = self.method_config.get("task_type", "auto")
        self.dataset_name = general_config.get("test_dataset_name")
        self.task_type = get_task_type(self.dataset_name, explicit)

        # xverify (LLM judge) endpoint -- must exist in model_api_config
        self.xverify_model_name = self.method_config.get(
            "xverify_model_name", "xverify-9b-c"
        )
        # judge call budget
        self.xverify_max_tokens = int(self.method_config.get("xverify_max_tokens", 64))
        self.xverify_temperature = float(
            self.method_config.get("xverify_temperature", 0.0)
        )

        # roles: take the first `agents_num` from the role list, cycling if necessary
        if self.agents_num <= len(AGENT_ROLES):
            self.roles = AGENT_ROLES[: self.agents_num]
        else:
            self.roles = [AGENT_ROLES[i % len(AGENT_ROLES)] for i in range(self.agents_num)]

    # ------------------------------------------------------------------
    # Generation LLM call: returns (response, prompt_tokens, completion_tokens)
    # so we can attribute tokens to each (agent, round) cell of the diagnostic.
    # ------------------------------------------------------------------
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def _call_llm_with_usage(self, messages, temperature=None):
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
    # xverify LLM judge: returns (is_correct: bool, status: str)
    # `status` is "ok" on a clean correct/incorrect verdict, otherwise an
    # error tag. The retry decorator handles transient endpoint failures;
    # exhausted retries fall through to status="judge-error" and is_correct=False.
    # ------------------------------------------------------------------
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def _call_xverify(self, query, response_text, gt) -> Tuple[bool, str]:
        if response_text is None or str(response_text).strip() == "":
            return False, "empty-response"
        if self.xverify_model_name not in self.model_api_config:
            raise RuntimeError(
                f"xverify model '{self.xverify_model_name}' not found in model_api_config; "
                "configure it in model_api_configs/model_api_config.json"
            )

        prompt = _xverify_format_prompt(query, str(response_text), gt)
        model_dict = random.choice(
            self.model_api_config[self.xverify_model_name]["model_list"]
        )
        request_dict = {
            "model": model_dict["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.xverify_max_tokens,
            "temperature": self.xverify_temperature,
            "timeout": self.model_timeout,
        }
        llm = openai.OpenAI(base_url=model_dict["model_url"], api_key=model_dict["api_key"])
        try:
            completion = llm.chat.completions.create(**request_dict)
            out = completion.choices[0].message.content
            n_p = completion.usage.prompt_tokens
            n_c = completion.usage.completion_tokens
        finally:
            llm.close()

        # update token stats for the judge separately
        if self.xverify_model_name not in self.token_stats:
            self.token_stats[self.xverify_model_name] = {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        self.token_stats[self.xverify_model_name]["num_llm_calls"] += 1
        self.token_stats[self.xverify_model_name]["prompt_tokens"] += n_p
        self.token_stats[self.xverify_model_name]["completion_tokens"] += n_c

        label = out.strip().lower() if isinstance(out, str) else ""
        if label == "correct":
            return True, "ok"
        if label == "incorrect":
            return False, "ok"
        return False, f"unparseable:{label[:60]}"

    def _judge_or_default(self, query, response_text, gt) -> Tuple[bool, str]:
        """Wrap _call_xverify so an exhausted-retry / unexpected error becomes
        (False, 'judge-error:...') instead of crashing the whole sample."""
        try:
            result = self._call_xverify(query, response_text, gt)
        except Exception as e:  # noqa: BLE001
            return False, f"judge-error:{type(e).__name__}"
        if result is None:  # retry exhausted -> handle_retry_error returns None
            return False, "judge-error:retry-exhausted"
        return result

    # ------------------------------------------------------------------
    # Vote-winner correctness: judge a synthesised "answer-only" response
    # with xverify, so the canonical winner is graded the same way as
    # individual agents.
    # ------------------------------------------------------------------
    def _vote_winner_response_text(self, winner: Optional[str]) -> Optional[str]:
        if winner is None:
            return None
        if self.task_type == "mcq":
            return f"The answer is ({winner})."
        return f"The answer is \\boxed{{{winner}}}."

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, sample: Dict) -> Dict:
        query = sample["query"]
        gt = sample.get("gt")
        # `gold_answer` is informational (canonical-form gold extracted by the
        # deterministic extractor) -- correctness itself is judged by xverify.
        gold = extract_gold(gt, self.task_type)

        # Per-agent conversation histories (independent contexts).
        agent_contexts: List[List[Dict[str, str]]] = []
        for role_def in self.roles:
            ctx = [
                {"role": "system", "content": role_def["system_prompt"]},
                {"role": "user", "content": get_initial_user_prompt(query, self.task_type)},
            ]
            agent_contexts.append(ctx)

        # ---------- Round 0: independent generation + per-agent xverify ----------
        initial_records = []
        for i, ctx in enumerate(agent_contexts):
            try:
                resp, n_p, n_c = self._call_llm_with_usage(ctx)
            except Exception as e:
                resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
            ctx.append({"role": "assistant", "content": resp})
            ext = extract_answer(resp, self.task_type)
            is_correct, judge_status = self._judge_or_default(query, resp, gt)
            initial_records.append(
                {
                    "agent_id": i,
                    "role": self.roles[i]["name"],
                    "raw_response": resp,
                    "extracted_answer": ext,
                    "is_correct": is_correct,
                    "judge_status": judge_status,
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
        initial_vote_correct, initial_vote_judge_status = self._judge_or_default(
            query, self._vote_winner_response_text(initial_vote), gt
        )
        initial_oracle_coverage = any(initial_correct_flags)

        # ---------- Rounds 1..rounds_num: vanilla MAD, fully connected ----------
        round_responses: Dict[str, List[Dict]] = {}
        round_vote_history: List[Optional[str]] = [initial_vote]
        round_vote_correct_history: List[bool] = [initial_vote_correct]
        round_vote_judge_status: List[str] = [initial_vote_judge_status]
        round_extracted_history: List[List[Optional[str]]] = [list(initial_extracted)]
        round_correct_history: List[List[bool]] = [list(initial_correct_flags)]

        for r in range(1, self.rounds_num + 1):
            # Snapshot peers' last-round responses BEFORE updating any context
            # -- otherwise within a round agent i would see agent (i-1)'s newly
            # generated round-r response, not its round-(r-1) response.
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
                is_correct, judge_status = self._judge_or_default(query, resp, gt)
                this_round_records.append(
                    {
                        "agent_id": i,
                        "role": self.roles[i]["name"],
                        "raw_response": resp,
                        "extracted_answer": ext,
                        "is_correct": is_correct,
                        "judge_status": judge_status,
                        "prompt_tokens": n_p,
                        "completion_tokens": n_c,
                        "tokens": n_p + n_c,
                    }
                )

            round_responses[str(r)] = this_round_records
            extracted_r = [rec["extracted_answer"] for rec in this_round_records]
            correct_r = [rec["is_correct"] for rec in this_round_records]
            vote_r, _ = plurality_vote(extracted_r, self.task_type)
            vote_r_correct, vote_r_status = self._judge_or_default(
                query, self._vote_winner_response_text(vote_r), gt
            )
            round_vote_history.append(vote_r)
            round_vote_correct_history.append(vote_r_correct)
            round_vote_judge_status.append(vote_r_status)
            round_extracted_history.append(extracted_r)
            round_correct_history.append(correct_r)

        # ---------- Final vote / coverage / bucket ----------
        final_vote = round_vote_history[-1]
        final_vote_correct = round_vote_correct_history[-1]
        final_oracle_coverage = any(round_correct_history[-1])

        if initial_vote_correct:
            bucket = "already_solved"
        elif initial_oracle_coverage:
            bucket = "recoverable"
        else:
            bucket = "unrecoverable"

        total_tokens = sum(rec["tokens"] for rec in initial_records) + sum(
            rec["tokens"]
            for rec_list in round_responses.values()
            for rec in rec_list
        )

        diagnostic = {
            "dataset": self.dataset_name,
            "task_type": self.task_type,
            "agents_num": self.agents_num,
            "rounds_num": self.rounds_num,
            "judge_protocol": "xverify",
            "judge_model": self.xverify_model_name,
            "gold_answer": gold,
            "initial_responses": initial_records,
            "round_responses": round_responses,
            "initial_vote": initial_vote,
            "initial_vote_correct": initial_vote_correct,
            "initial_vote_judge_status": initial_vote_judge_status,
            "initial_vote_counts": {
                (k if k is not None else "__none__"): v
                for k, v in initial_vote_counts.items()
            },
            "final_vote": final_vote,
            "final_vote_correct": final_vote_correct,
            "initial_oracle_coverage": initial_oracle_coverage,
            "final_oracle_coverage": final_oracle_coverage,
            "round_vote_history": round_vote_history,
            "round_vote_correct_history": round_vote_correct_history,
            "round_vote_judge_status": round_vote_judge_status,
            "round_extracted_history": round_extracted_history,
            "round_correct_history": round_correct_history,
            "bucket": bucket,
            "total_tokens": total_tokens,
        }

        response_str = final_vote if final_vote is not None else ""
        return {"response": response_str, "diagnostic": diagnostic}
