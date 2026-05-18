"""
SOO-Centered-v3: Task-adaptive spectral orchestration with symbolic plurality.

Keeps the spectral core of v2 (double-centered cosine -> PC1 contribution,
tr(S_c) variance consensus) and fixes the four empirical gaps observed in the
ablation study under results_ablation/20260416_152034:

    Gap 1 (GSM8K/GSM-Hard, -6.8 / -2.8 vs dylan_math, soo_math)
        Cause: v2 aggregates by embedding centroid. A single verbose wrong
        answer can sit closest to the centroid even when 3 of 5 agents produced
        the same correct number. Fix: symbolic answer extraction + contribution-
        weighted plurality vote (borrowed from soo_math / dylan_math).

    Gap 2 (AQUA-RAT, -4.7 vs selforg_no_debate)
        Cause: debate injects confusion on multiple-choice where the initial
        majority is already correct. Fix: answer-level consensus check at
        round 0 — if >= `answer_consensus_min_initial` agents agree on the
        extracted (X), return immediately, skipping debate entirely (matches
        selforg_no_debate's winning behaviour on easy-majority MCQ items).

    Gap 3 (MMLU-Pro, -3.4 vs selforg_random_graph)
        Cause: contribution-driven DAG concentrates edges between similar
        agents -> echo chamber on diverse-topic questions. Fix: with
        probability `diversity_p`, swap each contribution-picked peer for a
        random peer. Injects exactly the randomness that made random_graph
        win, while preserving the contribution signal for the majority of
        edges.

    Gap 4 (all tasks, format fragility for xverify)
        Cause: free-form prose sometimes confuses the verifier (e.g. "11
        minutes" with secondary numbers in the explanation). Fix: task-typed
        prompts request explicit `\\boxed{}` (math) / `(X)` (MCQ) format, and
        the final response is post-formatted to that canonical form so
        xverify has an unambiguous target.

Task type is auto-detected from `sample["source"]` / MCQ patterns in the
query, with an optional manual override via `force_task_type` in config.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from methods.dylan.utils_math import get_examples
from methods.soo_centered_v2 import SOO_Centered_v2_Main
from methods.soo_math.math_answer_utils import (
    extract_math_answer,
    is_equiv as math_is_equiv,
    plurality_answer_by_contribution as math_plurality,
    strip_string,
)
from methods.utils import handle_retry_error
from evaluations.evaluate_xverify import format_prompt as _xverify_format_prompt


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

# Regex: detect in-query MCQ options like "(A)...(B)..." or "A)... B)..."
_MCQ_OPTION_RE = re.compile(r"(?:^|\n|\s)\(?([A-J])\)\s*\S")

# Regex: extract the chosen letter from an assistant reply.
_MCQ_LETTER_RE = re.compile(r"\(?([A-Ja-j])\)")


class SOO_Centered_v3_Main(SOO_Centered_v2_Main):
    """Task-adaptive spectral orchestration.

    Adds three orthogonal improvements on top of v2:
      1. Symbolic answer extraction + contribution-weighted plurality vote
      2. Early answer-level consensus (round 0 and mid-debate)
      3. Diversity-augmented contribution DAG
    Everything else (PC1 contribution on S_c, tr(S_c) consensus check,
    DAG/debate plumbing) is inherited unchanged.
    """

    # Matches mad_vote so the diagnostic-record schema is identical and
    # scripts/diagnostic/analyze_diagnostic.py can consume our JSONL via
    # --filename_pattern soo_centered_v3_{dataset}_infer.jsonl.
    JUDGE_PROTOCOL = "xverify-cached-on-canonical"

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        mc = self.method_config
        self.answer_consensus_min_initial = int(mc.get("answer_consensus_min_initial", 3))
        self.answer_consensus_min_round = int(mc.get("answer_consensus_min_round", 3))
        self.enable_answer_consensus = bool(mc.get("enable_answer_consensus", True))
        self.enable_spectral_consensus = bool(mc.get("enable_spectral_consensus", True))
        self.diversity_p = float(mc.get("diversity_p", 0.2))
        self.include_math_few_shot = bool(mc.get("include_math_few_shot", True))
        self.math_mode = str(mc.get("math_mode", "complex")).lower()
        self.include_mcq_format_hint = bool(mc.get("include_mcq_format_hint", True))
        self.force_task_type = mc.get("force_task_type", None)

        # Ablation flags. Defaults preserve current SCC behavior so existing
        # configs (config_main.yaml) are unchanged.
        self.emit_diagnostic = bool(mc.get("emit_diagnostic", False))
        self.enable_contribution_routing = bool(mc.get("enable_contribution_routing", True))
        self.enable_contribution_aggregation = bool(mc.get("enable_contribution_aggregation", True))

        # xverify settings (only used when emit_diagnostic=True).
        self.xverify_model_name = mc.get("xverify_model_name", "xverify-9b-c")
        self.xverify_max_tokens = int(mc.get("xverify_max_tokens", 64))
        self.xverify_temperature = float(mc.get("xverify_temperature", 0.0))
        self.dataset_name = general_config.get("test_dataset_name")

        self._few_shot_block = get_examples(self.math_mode) if self.include_math_few_shot else ""

        # Deterministic RNG for diversity injection (independent of global RNG
        # so that other methods in the same run aren't perturbed).
        self._diversity_rng = random.Random(self.random_seed + 9157)

    # ------------------------------------------------------------------
    # Task-type detection
    # ------------------------------------------------------------------

    def _detect_task_type(self, sample: Dict) -> str:
        forced = self.force_task_type
        if isinstance(forced, str) and forced.lower() in ("math", "mcq", "open"):
            return forced.lower()

        if "num_choices" in sample and sample.get("num_choices"):
            return "mcq"

        source = str(sample.get("source", "")).strip().lower()
        if source in _MCQ_SOURCES:
            return "mcq"
        if source in _MATH_SOURCES:
            # AQUA-RAT is tagged as math in MASLab but is multiple-choice;
            # detect from the query body.
            if self._query_looks_like_mcq(sample.get("query", "")):
                return "mcq"
            return "math"

        # Fallback: sniff the query text.
        if self._query_looks_like_mcq(sample.get("query", "")):
            return "mcq"
        return "open"

    @staticmethod
    def _query_looks_like_mcq(query: str) -> bool:
        if not query:
            return False
        # At least 3 distinct option labels starting lines -> treat as MCQ.
        letters = set()
        for m in _MCQ_OPTION_RE.finditer(query):
            letters.add(m.group(1).upper())
        return len(letters) >= 3

    # ------------------------------------------------------------------
    # Task-typed prompting
    # ------------------------------------------------------------------

    def _init_prompt(self, query, task_type: str = "open") -> str:  # type: ignore[override]
        if task_type == "math":
            head = (
                "Solve the problem step by step. Show your reasoning briefly, "
                "then put the final numeric answer inside \\boxed{...} and "
                "on a new line write 'The answer is <final>'.\n"
            )
            if self._few_shot_block:
                return (
                    f"{self._few_shot_block}\n\n"
                    f"{head}[Problem]\n{query}\nAnswer: Let's think step by step\n"
                )
            return f"{head}[Problem]\n{query}\nAnswer:"

        if task_type == "mcq":
            head = (
                "Answer the multiple-choice question. Think step by step, then "
                "at the very end of your response write exactly: "
                "'The answer is (X)' where X is the letter of the correct "
                "option.\n"
            )
            return f"{head}[Question]\n{query}\n"

        if task_type == "code":
            # dylan-style: restate the signature, only python code in a single
            # ```python``` fence. We intentionally avoid the "impl]" markers
            # dylan uses internally; the extractor relies on the fenced block.
            return (
                "You must complete the Python function below. Restate the "
                "function signature, then write your full implementation. "
                "Respond with a single ```python ... ``` fenced code block "
                "containing only the function (and any necessary imports). "
                "Do not include free-flowing prose outside the code block.\n"
                f"[Function]\n```python\n{query}\n```\n"
            )

        return (
            "You will independently attempt the user's task. Let's think step "
            "by step. Be precise and complete.\n"
            f"[Task]\n{query}\n"
        )

    def _update_prompt(self, query, own_response, incoming, task_type: str = "open") -> str:  # type: ignore[override]
        block = "".join(f"\n[Peer {nid} answer]\n{txt}\n" for nid, txt in incoming)

        if task_type == "math":
            tail = (
                "Re-examine your solution and the peer answers step by step — "
                "they may be wrong. Give the updated solution, finish with "
                "\\boxed{<answer>} and then 'The answer is <answer>'."
            )
        elif task_type == "mcq":
            tail = (
                "Re-examine your reasoning and the peer answers step by step. "
                "They may be wrong. End your reply with 'The answer is (X)'."
            )
        elif task_type == "code":
            tail = (
                "Re-examine your implementation and the peer implementations "
                "above. They may contain bugs or miss corner cases. Write the "
                "improved full function in a single ```python ... ``` fenced "
                "block (restate the signature). No prose outside the block."
            )
        else:
            tail = (
                "Update your answer by critically evaluating the peer answers "
                "above. They may contain errors - do not copy blindly. Provide "
                "your improved answer with concise reasoning."
            )

        return (
            f"[Task]\n{query}\n"
            f"\n[Your previous answer]\n{own_response}\n"
            f"{block}\n"
            f"{tail}"
        )

    def _update_prompt_leader_agent(self, query, own_response, task_type: str = "open") -> str:  # type: ignore[override]
        if task_type == "math":
            tail = (
                "Review your previous answer and improve it if needed. Finish "
                "with \\boxed{<answer>} and then 'The answer is <answer>'."
            )
        elif task_type == "mcq":
            tail = (
                "Review your previous answer and improve it if needed. End "
                "with 'The answer is (X)'."
            )
        elif task_type == "code":
            tail = (
                "Review your previous implementation and improve it if needed. "
                "Return the full function in a single ```python ... ``` fenced "
                "block (restate the signature). No prose outside the block."
            )
        else:
            tail = "Review your previous answer and improve it if needed."

        return (
            "You are the current lead agent. No peer answers are available "
            "for this round.\n"
            f"[Task]\n{query}\n"
            f"\n[Your previous answer]\n{own_response}\n"
            f"{tail}"
        )

    # ------------------------------------------------------------------
    # Answer extraction / plurality (task-typed)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mcq_letter(reply: str) -> str:
        if not reply:
            return ""
        # Prefer the last "The answer is X" / "Final answer: X" form, which our
        # MCQ prompts explicitly request. Whitespace / optional parens tolerated.
        anchor = re.compile(
            r"(?:the\s+answer\s+is|final\s+answer[\s:]*|correct\s+answer[\s:]*)"
            r"\s*\(?\s*([A-Ja-j])\s*\)?",
            flags=re.IGNORECASE,
        )
        last = None
        for m in anchor.finditer(reply):
            last = m.group(1)
        if last:
            return last.upper()

        # Fallback: last "(X)" (allowing whitespace inside parens).
        paren = re.compile(r"\(\s*([A-Ja-j])\s*\)")
        for m in paren.finditer(reply):
            last = m.group(1)
        if last:
            return last.upper()

        # Last resort: "X)" at start of a line, e.g. bullet lists.
        line_start = re.compile(r"(?:^|\n)\s*([A-Ja-j])\)")
        for m in line_start.finditer(reply):
            last = m.group(1)
        return last.upper() if last else ""

    def _extract_answer(self, reply: str, task_type: str) -> str:
        if task_type == "math":
            return extract_math_answer(reply) or ""
        if task_type == "mcq":
            return self._extract_mcq_letter(reply)
        if task_type == "code":
            # Late import: parse_code_completion lives in dylan.utils_humaneval,
            # which has no class-level state. Passing "" as the question is safe
            # for our prompts — they always include a ```python``` fence with
            # the signature; the question-prepending fallback only fires when
            # no `def` is present, which our extractor pipeline filters out.
            from methods.dylan.utils_humaneval import parse_code_completion
            return parse_code_completion(reply, "") or ""
        return ""

    @staticmethod
    def _mcq_is_equiv(a: str, b: str) -> bool:
        return bool(a) and bool(b) and a.strip().upper() == b.strip().upper()

    def _plurality(
        self,
        answers: List[str],
        contributions: Sequence[float],
        task_type: str,
    ) -> Tuple[str, int]:
        """Return (canonical_extracted_answer, cluster_size).

        For math: reuses soo_math's normalizer/equivalence (strip_string +
        numeric fallback). For MCQ: exact-letter equality.  For open: no
        plurality (returns ("", 0)). For code: short-circuits — v3's
        diagnostic-emit path uses xverify which cannot grade code, so code
        runs must have emit_diagnostic=False and go through soo_scc's own
        scc_components-based inference (which calls count_first_plurality
        directly with entry_point/enable_plurality_for_code).
        """
        n = len(answers)
        if n == 0 or task_type in ("open", "code"):
            return "", 0

        extracted = [self._extract_answer(a, task_type) for a in answers]
        pairs = [(i, x) for i, x in enumerate(extracted) if x]
        if not pairs:
            return "", 0

        eq = math_is_equiv if task_type == "math" else self._mcq_is_equiv

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
            if not self.enable_contribution_aggregation:
                # Plain plurality: size only, deterministic first-cluster
                # tie-break (so the output is reproducible across runs).
                return (size, -groups.index(g))
            w = float(sum(contributions[i] for i, _ in g))
            return (size, w)

        best = max(groups, key=score)
        canonical = best[0][1]
        if task_type == "math":
            canonical = strip_string(canonical)
        return canonical, len(best)

    # ------------------------------------------------------------------
    # Output formatting (task-typed)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_final(canonical: str, task_type: str) -> str:
        if not canonical:
            return ""
        if task_type == "math":
            # xverify on math benches accepts either \boxed{} or "The answer is X"
            # -- providing both is the safest bet.
            return f"The answer is {canonical}. \\boxed{{{canonical}}}"
        if task_type == "mcq":
            letter = canonical.strip().upper()
            return f"The answer is ({letter})"
        return canonical

    # ------------------------------------------------------------------
    # Graph construction with diversity injection
    # ------------------------------------------------------------------

    def _build_diverse_graph(
        self,
        sims,
        contributions,
        n: int,
    ):
        """Contribution-based peer selection with random-swap diversity.

        Same scoring as SelfOrg (sim * (1 + n * (c_j - c_i))) but each chosen
        peer has a `diversity_p` probability of being swapped for a random
        non-self peer. Keeps self-organization as the default and injects
        selforg_random_graph's diversity benefit on the minority of edges.

        Ablation hook: when `enable_contribution_routing` is False, build a
        fully-connected graph (vanilla MAD topology) instead, bypassing both
        contribution scoring and diversity injection. DAG enforcement still
        applies so the topological ordering remains well-defined.
        """
        if not self.enable_contribution_routing:
            edges = {(j, i) for i in range(n) for j in range(n) if j != i}
            edge_w = {e: 1.0 for e in edges}
            if self.enforce_dag:
                edges, edge_w = self._dagify(edges, edge_w)
            return edges, edge_w

        helpful: List[List[int]] = []
        for i in range(n):
            scored = []
            for j in range(n):
                if j == i:
                    continue
                base = sims[i][j]
                adj = base * (1.0 + n * (contributions[j] - contributions[i]))
                scored.append((j, adj, base))

            scored = [p for p in scored if p[2] >= self.sim_threshold]
            scored.sort(key=lambda x: (x[1], contributions[x[0]]), reverse=True)
            keep = [j for (j, _, _) in scored[: self.top_k]]

            if self.diversity_p > 0.0:
                all_others = [j for j in range(n) if j != i]
                swapped: List[int] = []
                for j in keep:
                    if self._diversity_rng.random() < self.diversity_p:
                        pool = [k for k in all_others if k != j and k not in swapped]
                        if pool:
                            swapped.append(self._diversity_rng.choice(pool))
                            continue
                    swapped.append(j)
                keep = swapped

            helpful.append(sorted(set(keep)))

        edges = set()
        edge_w = {}
        for i in range(n):
            for j in helpful[i]:
                edges.add((j, i))
                adj_val = sims[i][j] * (1.0 + n * (contributions[j] - contributions[i]))
                edge_w[(j, i)] = max(0.0, adj_val)

        if self.enforce_dag:
            edges, edge_w = self._dagify(edges, edge_w)

        return edges, edge_w

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def inference(self, sample):
        if self.emit_diagnostic:
            return self._inference_with_diagnostic(sample)

        query = sample["query"]
        reference = sample.get("reference", None)
        task_type = self._detect_task_type(sample)

        n = self.num_agents

        # --- Round 0: independent answers with task-typed prompt ---
        init_answers: List[str] = []
        prompt0 = self._init_prompt(query, task_type)
        for i in range(n):
            system_prompt = self.role_map.get(self.roles[i], self.role_map["Assistant"])
            ans = self._call_llm(
                prompt=prompt0,
                system_prompt=system_prompt,
                temperature=self.temperature,
            )
            init_answers.append(ans)

        # --- PC1 contributions (and populate self._last_spectral) ---
        contributions = self._approx_shapley(init_answers, None)

        # --- Early answer-level consensus (skip debate if strong majority) ---
        canonical, size = self._plurality(init_answers, contributions, task_type)
        if (
            self.enable_answer_consensus
            and canonical
            and size >= self.answer_consensus_min_initial
        ):
            return {"response": self._format_final(canonical, task_type)}

        # --- Spectral consensus (v2 signal): answers are semantically close
        #     even when no discrete extracted cluster reached threshold ---
        if (
            self.enable_spectral_consensus
            and self._last_spectral is not None
            and self._last_spectral["trace"] < self.variance_consensus_thr
        ):
            return {
                "response": self._aggregate_from(
                    init_answers, contributions, task_type
                )
            }

        # --- Pairwise similarity and diversified DAG ---
        sims = self._pairwise_sims(init_answers)
        edges, _edge_w = self._build_diverse_graph(sims, contributions, n)

        # --- Debate rounds with per-round consensus check ---
        final_answers = self._propagate_with_typed_prompts(
            query,
            init_answers,
            edges,
            rounds=self.max_rounds,
            contributions=contributions,
            task_type=task_type,
        )

        # Final contribution estimate
        contributions = self._approx_shapley(final_answers, reference)

        # --- Final aggregation ---
        canonical, _ = self._plurality(final_answers, contributions, task_type)
        if canonical:
            return {"response": self._format_final(canonical, task_type)}

        # Open-ended or extraction failed on every agent: centroid fallback.
        # Ablation: when conservative aggregation is disabled, skip the
        # contribution-weighted centroid and just return the first agent's
        # initial answer (deterministic, contribution-free fallback).
        if not self.enable_contribution_aggregation:
            return {"response": init_answers[0]}
        return {"response": self._aggregate_from(final_answers, contributions, task_type)}

    # ------------------------------------------------------------------
    # Consensus check override — honour the spectral toggle
    # ------------------------------------------------------------------

    def _check_for_consensus(self, sims):
        # Only the spectral tr(S_c) signal is used. SelfOrg's raw pairwise-
        # similarity consensus is disabled in v3 because the 2^3 early-stop
        # ablation showed it interacts destructively with the other two
        # checks (AAS=111 underperformed AAS=110 by ~2pp on mean accuracy).
        if (
            self.enable_spectral_consensus
            and self._last_spectral is not None
            and self._last_spectral["trace"] < self.variance_consensus_thr
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers: aggregation / propagation
    # ------------------------------------------------------------------

    def _aggregate_from(
        self,
        answers: List[str],
        contributions: List[float],
        task_type: str,
    ) -> str:
        """Pick a single answer by the inherited centroid/argmax strategy."""
        n = len(answers)
        if n == 0:
            return ""

        if self.aggregate_mode == "single":
            best_idx = max(range(n), key=lambda i: contributions[i])
            return answers[best_idx]

        embs = self._embed_many(answers)
        weights = list(contributions)
        if sum(weights) <= 1e-9:
            weights = [1.0 / n] * n
        agg = self._weighted_centroid(embs, weights)
        nearest = max(range(n), key=lambda i: self._cosine(agg, embs[i]))
        return answers[nearest]

    def _propagate_with_typed_prompts(
        self,
        query,
        init_answers,
        edges,
        rounds: int,
        contributions: List[float],
        task_type: str,
    ) -> List[str]:
        """Replica of SelfOrg_Main._propagate_on_dag with two changes:
          * the update prompt is task-typed,
          * after each debate round we check answer-level plurality and
            short-circuit when the extracted-answer cluster is large enough.
        """
        current = list(init_answers)
        adj_in = defaultdict(list)
        for u, v in edges:
            adj_in[v].append(u)

        order = self._topo_order_by_contributions(edges, contributions) or list(
            range(self.num_agents)
        )

        for _r in range(max(1, rounds)):
            for i in order:
                preds = adj_in.get(i, [])
                is_leader = i == order[0]
                if not preds and not is_leader:
                    continue

                incoming = [(pid, current[pid]) for pid in preds]
                own_response = current[i]
                system_prompt = self.role_map.get(
                    self.roles[i], self.role_map["Assistant"]
                )

                if is_leader and not preds:
                    prompt = self._update_prompt_leader_agent(
                        query, own_response, task_type
                    )
                else:
                    prompt = self._update_prompt(
                        query, own_response, incoming, task_type
                    )

                current[i] = self._call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                )

            sims = self._pairwise_sims(current)
            contributions = self._approx_shapley(current, None)

            # Answer-level early stop mid-debate.
            canonical, size = self._plurality(current, contributions, task_type)
            if (
                self.enable_answer_consensus
                and canonical
                and size >= self.answer_consensus_min_round
            ):
                return current

            if self._check_for_consensus(sims):
                return current

            if self.reform and (_r + 1) < rounds:
                # Rebuild the diverse graph with updated contributions.
                edges, _ew = self._build_diverse_graph(
                    sims, contributions, self.num_agents
                )
                adj_in = defaultdict(list)
                for u, v in edges:
                    adj_in[v].append(u)
                order = self._topo_order_by_contributions(
                    edges, contributions
                ) or list(range(self.num_agents))

        return current

    # ==================================================================
    # Diagnostic emission (only invoked when emit_diagnostic=True).
    # Schema mirrors methods/mad_vote/mad_vote_main.py:218-388 so the same
    # analyzer (scripts/diagnostic/analyze_diagnostic.py) reads both.
    # ==================================================================

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def _call_llm_with_usage(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, int, int]:
        """Token-tracking LLM call. Returns (response, prompt_tokens,
        completion_tokens). Updates self.token_stats[model_name]."""
        model_name = self.model_name
        model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        backend_name, model_url, api_key = (
            model_dict["model_name"],
            model_dict["model_url"],
            model_dict["api_key"],
        )

        messages: List[Dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_dict: Dict[str, Any] = {
            "model": backend_name,
            "messages": messages,
            "max_tokens": self.model_max_tokens,
            "timeout": self.model_timeout,
        }
        if "o1" not in backend_name:
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

        if backend_name not in self.token_stats:
            self.token_stats[backend_name] = {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        self.token_stats[backend_name]["num_llm_calls"] += 1
        self.token_stats[backend_name]["prompt_tokens"] += n_prompt
        self.token_stats[backend_name]["completion_tokens"] += n_completion

        return response, n_prompt, n_completion

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def _call_xverify(
        self, query: str, response_text: Optional[str], gt
    ) -> Tuple[bool, str]:
        if response_text is None or str(response_text).strip() == "":
            return False, "empty-response"
        if self.xverify_model_name not in self.model_api_config:
            raise RuntimeError(
                f"xverify model '{self.xverify_model_name}' not found in "
                "model_api_config; configure it in "
                "model_api_configs/model_api_config.json"
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
        llm = openai.OpenAI(
            base_url=model_dict["model_url"], api_key=model_dict["api_key"]
        )
        try:
            completion = llm.chat.completions.create(**request_dict)
            out = completion.choices[0].message.content
            n_p = completion.usage.prompt_tokens
            n_c = completion.usage.completion_tokens
        finally:
            llm.close()

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

    def _judge_or_default(
        self, query: str, response_text: Optional[str], gt
    ) -> Tuple[bool, str]:
        try:
            result = self._call_xverify(query, response_text, gt)
        except Exception as e:  # noqa: BLE001
            return False, f"judge-error:{type(e).__name__}"
        if result is None:
            return False, "judge-error:retry-exhausted"
        return result

    def _canonical_response_text(
        self, canonical: Optional[str], task_type: str
    ) -> Optional[str]:
        """Synthesise an 'answer-only' sentence so xverify sees a consistent
        input form. Mirrors mad_vote but parametrised on task_type.

        Code is special-cased: the canonical is already an executable function
        (signature preserved), so we return it verbatim. Diagnostic-mode
        consumers must NOT route this through xverify — they need an
        execution-based grader (see evaluations/evaluate_code.py). The
        recommended setup is `emit_diagnostic=false` for code configs.
        """
        if canonical is None:
            return None
        if task_type == "mcq":
            return f"The answer is ({canonical})."
        if task_type == "math":
            return f"The answer is \\boxed{{{canonical}}}."
        if task_type == "code":
            return str(canonical)
        return str(canonical)

    @staticmethod
    def _normalize_extracted(ext) -> Optional[str]:
        """SCC's `_extract_answer` returns "" on failure; the diagnostic
        schema (and judge_cache) treats None as the failed-extraction key."""
        if ext is None or (isinstance(ext, str) and ext.strip() == ""):
            return None
        return ext

    def _pad_rounds(
        self,
        *,
        start_round: int,
        end_round: int,
        canonical_extracted: List[Optional[str]],
        canonical_correct_flags: List[bool],
        vote: Optional[str],
        vote_correct: bool,
        vote_judge_status: str,
        round_responses: Dict[str, List[Dict[str, Any]]],
        round_vote_history: List[Optional[str]],
        round_vote_correct_history: List[bool],
        round_vote_judge_status: List[str],
        round_extracted_history: List[List[Optional[str]]],
        round_correct_history: List[List[bool]],
    ) -> None:
        """Pad rounds [start_round, end_round] with the converged state.

        Rationale: SCC may early-stop after round k < max_rounds. To keep
        round_vote_correct_history at a consistent length across samples
        (so build_table5 in analyze_diagnostic.py reads a flat curve after
        convergence), we synthesise no-LLM-call records that carry the
        converged answer forward."""
        for r in range(start_round, end_round + 1):
            synthetic_records = []
            for i in range(self.num_agents):
                synthetic_records.append({
                    "agent_id": i,
                    "role": self.roles[i],
                    "raw_response": "",
                    "extracted_answer": canonical_extracted[i],
                    "is_correct": canonical_correct_flags[i],
                    "judge_status": "early-stop-padding",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens": 0,
                })
            round_responses[str(r)] = synthetic_records
            round_vote_history.append(vote)
            round_vote_correct_history.append(vote_correct)
            round_vote_judge_status.append(vote_judge_status)
            round_extracted_history.append(list(canonical_extracted))
            round_correct_history.append(list(canonical_correct_flags))

    def _build_diag_return(
        self,
        *,
        gold,
        task_type: str,
        initial_records: List[Dict[str, Any]],
        round_responses: Dict[str, List[Dict[str, Any]]],
        initial_vote: Optional[str],
        initial_vote_correct: bool,
        initial_vote_judge_status: str,
        initial_vote_counts: Dict[Any, int],
        final_vote: Optional[str],
        final_vote_correct: bool,
        initial_oracle_coverage: bool,
        final_oracle_coverage: bool,
        round_vote_history: List[Optional[str]],
        round_vote_correct_history: List[bool],
        round_vote_judge_status: List[str],
        round_extracted_history: List[List[Optional[str]]],
        round_correct_history: List[List[bool]],
        judge_cache: Dict[Optional[str], Tuple[bool, str]],
        final_response: str,
    ) -> Dict[str, Any]:
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

        diagnostic: Dict[str, Any] = {
            "dataset": self.dataset_name,
            "task_type": task_type,
            "agents_num": self.num_agents,
            "rounds_num": self.max_rounds,
            "judge_protocol": self.JUDGE_PROTOCOL,
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
            "judge_cache_keys": [
                ("__none__" if k is None else str(k)) for k in judge_cache.keys()
            ],
        }

        response_str = final_response if final_response else (
            final_vote if final_vote is not None else ""
        )
        return {"response": response_str, "diagnostic": diagnostic}

    def _inference_with_diagnostic(self, sample: Dict) -> Dict:
        """SCC inference with mad_vote-compatible diagnostic emission."""
        # Local import: keeps the fast path free of mad_vote module load cost.
        from methods.mad_vote.extractor import extract_gold

        query = sample["query"]
        gt = sample.get("gt")
        reference = sample.get("reference", None)
        task_type = self._detect_task_type(sample)
        if task_type == "code":
            # Same placeholder convention as mad_vote.extractor.extract_gold:
            # store str(gt) (canonical solution) as gold_answer so the
            # downstream analyze_diagnostic filter doesn't drop code
            # samples. Never used for grading — grading goes through
            # evaluations.evaluate_code.grade_code_sample.
            gold = str(gt) if gt and str(gt).strip() else "code-no-string-gold"
        elif task_type in ("math", "mcq"):
            gold = extract_gold(gt, task_type)
        else:
            gold = gt

        n = self.num_agents

        # Code-grading context (used only when task_type == "code").
        code_entry_point = sample.get("entry_point", "")
        code_test = sample.get("test")
        code_test_list = sample.get("test_list", [])
        code_test_setup = sample.get("test_setup_code", "")

        judge_cache: Dict[Optional[str], Tuple[bool, str]] = {
            None: (False, "no-extraction")
        }

        def grade_canonical(canonical: Optional[str]) -> Tuple[bool, str]:
            if canonical in judge_cache:
                return judge_cache[canonical]
            if task_type == "code":
                from evaluations.evaluate_code import grade_code_sample
                verdict = grade_code_sample(
                    canonical,
                    entry_point=code_entry_point,
                    test=code_test,
                    test_list=code_test_list,
                    test_setup_code=code_test_setup,
                )
            else:
                response_text = self._canonical_response_text(canonical, task_type)
                verdict = self._judge_or_default(query, response_text, gt)
            judge_cache[canonical] = verdict
            return verdict

        # ---------------- Round 0 ----------------
        init_answers: List[str] = []
        initial_records: List[Dict[str, Any]] = []
        prompt0 = self._init_prompt(query, task_type)
        for i in range(n):
            system_prompt = self.role_map.get(self.roles[i], self.role_map["Assistant"])
            try:
                resp, n_p, n_c = self._call_llm_with_usage(
                    prompt=prompt0,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                )
            except Exception as e:  # noqa: BLE001
                resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
            init_answers.append(resp)
            ext = self._normalize_extracted(self._extract_answer(resp, task_type))
            is_correct, judge_status = grade_canonical(ext)
            initial_records.append({
                "agent_id": i,
                "role": self.roles[i],
                "raw_response": resp,
                "extracted_answer": ext,
                "is_correct": is_correct,
                "judge_status": judge_status,
                "prompt_tokens": n_p,
                "completion_tokens": n_c,
                "tokens": n_p + n_c,
            })

        initial_extracted: List[Optional[str]] = [r["extracted_answer"] for r in initial_records]
        initial_correct_flags: List[bool] = [r["is_correct"] for r in initial_records]
        initial_oracle_coverage = any(initial_correct_flags)

        # Round-0 plurality (uniform-weighted; matches mad_vote.plurality_vote).
        counts_dict: Dict[Any, int] = {}
        first_pos: Dict[Any, int] = {}
        for idx, ext in enumerate(initial_extracted):
            key = ext if ext is not None else "__none__"
            counts_dict[key] = counts_dict.get(key, 0) + 1
            first_pos.setdefault(key, idx)
        real_keys = [k for k in counts_dict if k != "__none__"]
        if real_keys:
            initial_vote = max(
                real_keys,
                key=lambda k: (counts_dict[k], -first_pos[k]),
            )
        else:
            initial_vote = None
        initial_vote_correct, initial_vote_judge_status = grade_canonical(initial_vote)
        initial_vote_counts = counts_dict

        # Round trackers (round 0 == initial).
        round_responses: Dict[str, List[Dict[str, Any]]] = {}
        round_vote_history: List[Optional[str]] = [initial_vote]
        round_vote_correct_history: List[bool] = [initial_vote_correct]
        round_vote_judge_status: List[str] = [initial_vote_judge_status]
        round_extracted_history: List[List[Optional[str]]] = [list(initial_extracted)]
        round_correct_history: List[List[bool]] = [list(initial_correct_flags)]

        # Contributions (PC1 of S_c) and self._last_spectral side-effect.
        contributions = self._approx_shapley(init_answers, None)

        # ---- Branch A: answer-consensus early exit ----
        canonical_a, size_a = self._plurality(init_answers, contributions, task_type)
        if (
            self.enable_answer_consensus
            and canonical_a
            and size_a >= self.answer_consensus_min_initial
        ):
            final_vote = canonical_a
            final_vote_correct, final_vote_judge_status = grade_canonical(final_vote)
            self._pad_rounds(
                start_round=1, end_round=self.max_rounds,
                canonical_extracted=initial_extracted,
                canonical_correct_flags=initial_correct_flags,
                vote=final_vote, vote_correct=final_vote_correct,
                vote_judge_status=final_vote_judge_status,
                round_responses=round_responses,
                round_vote_history=round_vote_history,
                round_vote_correct_history=round_vote_correct_history,
                round_vote_judge_status=round_vote_judge_status,
                round_extracted_history=round_extracted_history,
                round_correct_history=round_correct_history,
            )
            final_oracle_coverage = initial_oracle_coverage
            return self._build_diag_return(
                gold=gold, task_type=task_type,
                initial_records=initial_records, round_responses=round_responses,
                initial_vote=initial_vote, initial_vote_correct=initial_vote_correct,
                initial_vote_judge_status=initial_vote_judge_status,
                initial_vote_counts=initial_vote_counts,
                final_vote=final_vote, final_vote_correct=final_vote_correct,
                initial_oracle_coverage=initial_oracle_coverage,
                final_oracle_coverage=final_oracle_coverage,
                round_vote_history=round_vote_history,
                round_vote_correct_history=round_vote_correct_history,
                round_vote_judge_status=round_vote_judge_status,
                round_extracted_history=round_extracted_history,
                round_correct_history=round_correct_history,
                judge_cache=judge_cache,
                final_response=self._format_final(final_vote, task_type),
            )

        # ---- Branch B: spectral early exit ----
        if (
            self.enable_spectral_consensus
            and self._last_spectral is not None
            and self._last_spectral["trace"] < self.variance_consensus_thr
        ):
            agg_response = self._aggregate_from(init_answers, contributions, task_type)
            final_vote = self._normalize_extracted(
                self._extract_answer(agg_response, task_type)
            )
            final_vote_correct, final_vote_judge_status = grade_canonical(final_vote)
            self._pad_rounds(
                start_round=1, end_round=self.max_rounds,
                canonical_extracted=initial_extracted,
                canonical_correct_flags=initial_correct_flags,
                vote=final_vote, vote_correct=final_vote_correct,
                vote_judge_status=final_vote_judge_status,
                round_responses=round_responses,
                round_vote_history=round_vote_history,
                round_vote_correct_history=round_vote_correct_history,
                round_vote_judge_status=round_vote_judge_status,
                round_extracted_history=round_extracted_history,
                round_correct_history=round_correct_history,
            )
            final_oracle_coverage = initial_oracle_coverage
            return self._build_diag_return(
                gold=gold, task_type=task_type,
                initial_records=initial_records, round_responses=round_responses,
                initial_vote=initial_vote, initial_vote_correct=initial_vote_correct,
                initial_vote_judge_status=initial_vote_judge_status,
                initial_vote_counts=initial_vote_counts,
                final_vote=final_vote, final_vote_correct=final_vote_correct,
                initial_oracle_coverage=initial_oracle_coverage,
                final_oracle_coverage=final_oracle_coverage,
                round_vote_history=round_vote_history,
                round_vote_correct_history=round_vote_correct_history,
                round_vote_judge_status=round_vote_judge_status,
                round_extracted_history=round_extracted_history,
                round_correct_history=round_correct_history,
                judge_cache=judge_cache,
                final_response=agg_response,
            )

        # ---- Branch C: instrumented debate ----
        sims = self._pairwise_sims(init_answers)
        edges, _edge_w = self._build_diverse_graph(sims, contributions, n)

        final_answers, final_correct_flags = self._propagate_with_typed_prompts_with_diag(
            query=query,
            init_answers=init_answers,
            initial_extracted=initial_extracted,
            initial_correct_flags=initial_correct_flags,
            edges=edges,
            rounds=self.max_rounds,
            contributions=list(contributions),
            task_type=task_type,
            grade_canonical=grade_canonical,
            round_responses=round_responses,
            round_vote_history=round_vote_history,
            round_vote_correct_history=round_vote_correct_history,
            round_vote_judge_status=round_vote_judge_status,
            round_extracted_history=round_extracted_history,
            round_correct_history=round_correct_history,
        )

        # ---- Final aggregation ----
        final_contributions = self._approx_shapley(final_answers, reference)
        canonical_f, _ = self._plurality(final_answers, final_contributions, task_type)
        if canonical_f:
            final_vote = canonical_f
            final_response = self._format_final(canonical_f, task_type)
        elif self.enable_contribution_aggregation:
            final_response = self._aggregate_from(
                final_answers, final_contributions, task_type
            )
            final_vote = self._normalize_extracted(
                self._extract_answer(final_response, task_type)
            )
        else:
            final_response = init_answers[0]
            final_vote = self._normalize_extracted(
                self._extract_answer(final_response, task_type)
            )

        final_vote_correct, _ = grade_canonical(final_vote)
        final_oracle_coverage = any(final_correct_flags)

        # Replace the LAST round's vote with the post-aggregation final_vote
        # so round_vote_history[-1] reflects the actually-returned answer.
        if round_vote_history:
            round_vote_history[-1] = final_vote
            round_vote_correct_history[-1] = final_vote_correct

        return self._build_diag_return(
            gold=gold, task_type=task_type,
            initial_records=initial_records, round_responses=round_responses,
            initial_vote=initial_vote, initial_vote_correct=initial_vote_correct,
            initial_vote_judge_status=initial_vote_judge_status,
            initial_vote_counts=initial_vote_counts,
            final_vote=final_vote, final_vote_correct=final_vote_correct,
            initial_oracle_coverage=initial_oracle_coverage,
            final_oracle_coverage=final_oracle_coverage,
            round_vote_history=round_vote_history,
            round_vote_correct_history=round_vote_correct_history,
            round_vote_judge_status=round_vote_judge_status,
            round_extracted_history=round_extracted_history,
            round_correct_history=round_correct_history,
            judge_cache=judge_cache,
            final_response=final_response,
        )

    def _propagate_with_typed_prompts_with_diag(
        self,
        *,
        query,
        init_answers: List[str],
        initial_extracted: List[Optional[str]],
        initial_correct_flags: List[bool],
        edges,
        rounds: int,
        contributions: List[float],
        task_type: str,
        grade_canonical: Callable[[Optional[str]], Tuple[bool, str]],
        round_responses: Dict[str, List[Dict[str, Any]]],
        round_vote_history: List[Optional[str]],
        round_vote_correct_history: List[bool],
        round_vote_judge_status: List[str],
        round_extracted_history: List[List[Optional[str]]],
        round_correct_history: List[List[bool]],
    ) -> Tuple[List[str], List[bool]]:
        """Instrumented copy of `_propagate_with_typed_prompts`. Mutates the
        round_* lists in-place; returns (final_answers, final_correct_flags)."""
        current = list(init_answers)
        current_extracted: List[Optional[str]] = list(initial_extracted)
        current_correct_flags: List[bool] = list(initial_correct_flags)

        adj_in: Dict[int, List[int]] = defaultdict(list)
        for u, v in edges:
            adj_in[v].append(u)

        order = self._topo_order_by_contributions(edges, contributions) or list(
            range(self.num_agents)
        )

        last_completed_round = 0

        for _r in range(max(1, rounds)):
            round_idx = _r + 1
            per_agent_records: List[Dict[str, Any]] = [None] * self.num_agents  # type: ignore

            # Carry-over records for non-leader agents with no predecessors.
            for i in range(self.num_agents):
                preds = adj_in.get(i, [])
                is_leader = i == order[0]
                if not preds and not is_leader:
                    per_agent_records[i] = {
                        "agent_id": i,
                        "role": self.roles[i],
                        "raw_response": current[i],
                        "extracted_answer": current_extracted[i],
                        "is_correct": current_correct_flags[i],
                        "judge_status": "carry-over",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "tokens": 0,
                    }

            # LLM calls in topological order.
            for i in order:
                preds = adj_in.get(i, [])
                is_leader = i == order[0]
                if not preds and not is_leader:
                    continue

                incoming = [(pid, current[pid]) for pid in preds]
                own_response = current[i]
                system_prompt = self.role_map.get(
                    self.roles[i], self.role_map["Assistant"]
                )

                if is_leader and not preds:
                    prompt = self._update_prompt_leader_agent(
                        query, own_response, task_type
                    )
                else:
                    prompt = self._update_prompt(
                        query, own_response, incoming, task_type
                    )

                try:
                    resp, n_p, n_c = self._call_llm_with_usage(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=self.temperature,
                    )
                except Exception as e:  # noqa: BLE001
                    resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
                current[i] = resp
                ext = self._normalize_extracted(self._extract_answer(resp, task_type))
                is_correct, judge_status = grade_canonical(ext)
                current_extracted[i] = ext
                current_correct_flags[i] = is_correct

                per_agent_records[i] = {
                    "agent_id": i,
                    "role": self.roles[i],
                    "raw_response": resp,
                    "extracted_answer": ext,
                    "is_correct": is_correct,
                    "judge_status": judge_status,
                    "prompt_tokens": n_p,
                    "completion_tokens": n_c,
                    "tokens": n_p + n_c,
                }

            # Any leftover None slots indicate a topo-ordering miss; should not
            # happen with enforce_dag, but guard against it for robustness.
            for i in range(self.num_agents):
                if per_agent_records[i] is None:
                    per_agent_records[i] = {
                        "agent_id": i,
                        "role": self.roles[i],
                        "raw_response": current[i],
                        "extracted_answer": current_extracted[i],
                        "is_correct": current_correct_flags[i],
                        "judge_status": "skipped",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "tokens": 0,
                    }

            round_responses[str(round_idx)] = per_agent_records

            # Per-round contribution-weighted plurality vote.
            sims = self._pairwise_sims(current)
            contributions = self._approx_shapley(current, None)
            canonical, size = self._plurality(current, contributions, task_type)

            vote_r = canonical if canonical else None
            vote_r_correct, vote_r_status = grade_canonical(vote_r)

            round_vote_history.append(vote_r)
            round_vote_correct_history.append(vote_r_correct)
            round_vote_judge_status.append(vote_r_status)
            round_extracted_history.append(list(current_extracted))
            round_correct_history.append(list(current_correct_flags))

            last_completed_round = round_idx

            # Mid-debate early stop (answer consensus).
            if (
                self.enable_answer_consensus
                and canonical
                and size >= self.answer_consensus_min_round
            ):
                self._pad_rounds(
                    start_round=last_completed_round + 1,
                    end_round=self.max_rounds,
                    canonical_extracted=current_extracted,
                    canonical_correct_flags=current_correct_flags,
                    vote=vote_r, vote_correct=vote_r_correct,
                    vote_judge_status=vote_r_status,
                    round_responses=round_responses,
                    round_vote_history=round_vote_history,
                    round_vote_correct_history=round_vote_correct_history,
                    round_vote_judge_status=round_vote_judge_status,
                    round_extracted_history=round_extracted_history,
                    round_correct_history=round_correct_history,
                )
                return current, current_correct_flags

            # Mid-debate early stop (spectral).
            if self._check_for_consensus(sims):
                self._pad_rounds(
                    start_round=last_completed_round + 1,
                    end_round=self.max_rounds,
                    canonical_extracted=current_extracted,
                    canonical_correct_flags=current_correct_flags,
                    vote=vote_r, vote_correct=vote_r_correct,
                    vote_judge_status=vote_r_status,
                    round_responses=round_responses,
                    round_vote_history=round_vote_history,
                    round_vote_correct_history=round_vote_correct_history,
                    round_vote_judge_status=round_vote_judge_status,
                    round_extracted_history=round_extracted_history,
                    round_correct_history=round_correct_history,
                )
                return current, current_correct_flags

            if self.reform and (_r + 1) < rounds:
                edges, _ew = self._build_diverse_graph(
                    sims, contributions, self.num_agents
                )
                adj_in = defaultdict(list)
                for u, v in edges:
                    adj_in[v].append(u)
                order = self._topo_order_by_contributions(
                    edges, contributions
                ) or list(range(self.num_agents))

        return current, current_correct_flags
