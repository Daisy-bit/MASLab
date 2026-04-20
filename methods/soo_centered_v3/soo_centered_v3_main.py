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
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from methods.dylan.utils_math import get_examples
from methods.soo_centered_v2 import SOO_Centered_v2_Main
from methods.soo_math.math_answer_utils import (
    extract_math_answer,
    is_equiv as math_is_equiv,
    plurality_answer_by_contribution as math_plurality,
    strip_string,
)


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
        plurality (returns ("", 0)).
        """
        n = len(answers)
        if n == 0 or task_type == "open":
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
        """
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
