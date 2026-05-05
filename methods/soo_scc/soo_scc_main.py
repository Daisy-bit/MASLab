"""SOO-SCC: SCC reference baseline rebuilt on top of `methods.scc_components`.

Behavioural superset / drop-in replacement for `methods.soo_centered_v3`.
The orchestration code (`inference()` and `_propagate_with_typed_prompts`)
is rewritten to delegate every SCC component call to the shared module
`methods.scc_components`. All other behaviour — task-typed prompts,
centroid fallback, judge-emitting diagnostic path, format_final
post-processing — is inherited from `SOO_Centered_v3_Main`.

This means `soo_scc` is byte-equivalent to `soo_centered_v3` on the
default `inference()` path *by construction* (the parity tests in
`tests/test_scc_components_parity.py` verify each component returns the
same value for the same inputs). The diagnostic-emit path is inherited
verbatim from v3, so any consumer of v3's diagnostic JSONL works unchanged
on `soo_scc` outputs.

Why a separate class? `mad_scc` will reuse the same `methods.scc_components`
functions on top of a different baseline (mad_vote concurrent broadcast).
By extracting components first and rebuilding both methods on top of them,
we eliminate the per-baseline reimplementation drift documented in
`docs/scc_vs_v3_pseudocode.md`.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from methods.scc_components import (
    build_diverse_graph,
    count_first_plurality,
    detect_task_type,
    format_final,
    is_spectral_consensus,
    pairwise_cosine,
    pc1_contributions,
    topo_order_by_contributions,
)
from methods.soo_centered_v3 import SOO_Centered_v3_Main


# ---------------------------------------------------------------------------
# Shared initial-response pool — same protocol as mad_scc / mad_vote_scc.
# Diagnostic JSONLs from results_archive/results_diagnostic/... are valid
# pool sources: each record's `diagnostic.initial_responses[i].raw_response`
# is replayed verbatim as agent i's round-0 answer.
# ---------------------------------------------------------------------------

_INITIAL_POOL_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_INITIAL_POOL_LOCK = threading.Lock()


def _load_initial_pool(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with _INITIAL_POOL_LOCK:
        hit = _INITIAL_POOL_CACHE.get(path)
        if hit is not None:
            return hit
        out: Dict[str, List[Dict[str, Any]]] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("error"):
                        continue
                    diag = rec.get("diagnostic") or {}
                    init = diag.get("initial_responses")
                    q = rec.get("query")
                    if init and q is not None:
                        out[q] = init
        _INITIAL_POOL_CACHE[path] = out
        return out


class SOO_SCC_Main(SOO_Centered_v3_Main):
    """v3-faithful baseline whose SCC component calls go through
    `methods.scc_components`. Diagnostic-emit path is inherited from v3."""

    def __init__(self, general_config, method_config_name=None):
        method_config_name = (
            "config_main" if method_config_name is None else method_config_name
        )
        super().__init__(general_config, method_config_name=method_config_name)

        # Optional shared initial-response pool. When --initial_pool_dir is
        # supplied to inference.py, round-0 LLM calls are skipped for any
        # query that has a cached entry (matched by exact string).
        init_dir = general_config.get("initial_pool_dir") or ""
        init_pat = general_config.get(
            "initial_pool_filename_pattern",
            "mad_vote_{dataset}_infer.jsonl",
        )
        self.initial_pool: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if init_dir:
            init_path = os.path.join(
                init_dir, init_pat.format(dataset=self.dataset_name)
            )
            self.initial_pool = _load_initial_pool(init_path)
            if not self.initial_pool:
                print(
                    f"[WARN] initial_pool_dir set but no usable records found at "
                    f"{init_path}; falling back to fresh round-0 sampling for "
                    f"{self.dataset_name}."
                )

        # State used by `_call_llm_with_usage` to replay the first n round-0
        # responses from the cached pool when running the diagnostic-emit path.
        self._round_0_replay: Optional[List[Dict[str, Any]]] = None
        self._pool_replay_remaining: int = 0

    # ------------------------------------------------------------------
    # Override `_call_llm_with_usage` so the diagnostic-emit path inherited
    # from v3 can be transparently replayed from the shared initial pool.
    # The first `num_agents` calls of an inference (round 0) consume cached
    # responses; subsequent calls (debate rounds) fall through to the real
    # LLM. Default path uses self._call_llm and is handled separately in
    # `inference()`.
    # ------------------------------------------------------------------
    def _call_llm_with_usage(self, prompt, system_prompt=None, temperature=None):
        if self._round_0_replay is not None and self._pool_replay_remaining > 0:
            idx = self.num_agents - self._pool_replay_remaining
            rec = self._round_0_replay[idx]
            self._pool_replay_remaining -= 1
            if self._pool_replay_remaining <= 0:
                self._round_0_replay = None
                self._pool_replay_remaining = 0
            return (
                rec.get("raw_response", "") or "",
                int(rec.get("prompt_tokens", 0) or 0),
                int(rec.get("completion_tokens", 0) or 0),
            )
        return super()._call_llm_with_usage(
            prompt=prompt, system_prompt=system_prompt, temperature=temperature
        )

    # ------------------------------------------------------------------
    # Default inference path: rebuild on scc_components
    # ------------------------------------------------------------------
    def inference(self, sample):
        # Prime round-0 replay state for both the default path (consumed
        # below directly) and the diagnostic-emit path (consumed via the
        # overridden `_call_llm_with_usage` above).
        query0 = sample.get("query", "")
        if self.initial_pool is not None:
            entry0 = self.initial_pool.get(query0)
            if entry0 is not None and len(entry0) >= self.num_agents:
                self._round_0_replay = entry0
                self._pool_replay_remaining = self.num_agents
            else:
                self._round_0_replay = None
                self._pool_replay_remaining = 0
        else:
            self._round_0_replay = None
            self._pool_replay_remaining = 0

        if self.emit_diagnostic:
            return self._inference_with_diagnostic(sample)

        query = sample["query"]
        reference = sample.get("reference", None)
        task_type = detect_task_type(
            sample, force_task_type=self.force_task_type
        )

        n = self.num_agents

        # ---- Round 0: replay shared pool if configured, else live LLM ----
        cached_init: Optional[List[Dict[str, Any]]] = None
        if self.initial_pool is not None:
            entry = self.initial_pool.get(query)
            if entry is not None and len(entry) >= n:
                cached_init = entry

        init_answers: List[str] = []
        if cached_init is not None:
            for i in range(n):
                rec = cached_init[i]
                init_answers.append(rec.get("raw_response", "") or "")
        else:
            prompt0 = self._init_prompt(query, task_type)
            for i in range(n):
                sysp = self.role_map.get(self.roles[i], self.role_map["Assistant"])
                ans = self._call_llm(
                    prompt=prompt0,
                    system_prompt=sysp,
                    temperature=self.temperature,
                )
                init_answers.append(ans)

        # ---- Spectral analysis via shared component ----
        embs = self._embed_many(init_answers)
        contributions, spec = pc1_contributions(embs)
        self._last_spectral = spec  # so v3-inherited _check_for_consensus works

        # ---- Early-exit gate A: answer-level plurality ----
        canonical, size = count_first_plurality(
            init_answers, contributions, task_type,
            enable_contribution_aggregation=self.enable_contribution_aggregation,
        )
        if (
            self.enable_answer_consensus
            and canonical
            and size >= self.answer_consensus_min_initial
        ):
            return {"response": format_final(canonical, task_type)}

        # ---- Early-exit gate B: spectral trace ----
        if self.enable_spectral_consensus and is_spectral_consensus(
            spec, self.variance_consensus_thr
        ):
            return {
                "response": self._aggregate_from(
                    init_answers, contributions, task_type
                )
            }

        # ---- Build initial DAG ----
        sims = pairwise_cosine(embs)
        edges, _edge_w = build_diverse_graph(
            sims,
            contributions,
            n,
            top_k=self.top_k,
            sim_threshold=self.sim_threshold,
            diversity_p=self.diversity_p,
            enforce_dag=self.enforce_dag,
            enable_routing=self.enable_contribution_routing,
            rng=self._diversity_rng,
        )

        # ---- Cascade debate (still local: needs LLM + role state) ----
        final_answers = self._propagate_with_typed_prompts(
            query,
            init_answers,
            edges,
            rounds=self.max_rounds,
            contributions=contributions,
            task_type=task_type,
        )

        # ---- Final aggregation ----
        final_embs = self._embed_many(final_answers)
        contributions, _ = pc1_contributions(final_embs)
        canonical, _ = count_first_plurality(
            final_answers, contributions, task_type,
            enable_contribution_aggregation=self.enable_contribution_aggregation,
        )
        if canonical:
            return {"response": format_final(canonical, task_type)}

        # ---- Centroid / first-agent fallback when extraction fails ----
        if not self.enable_contribution_aggregation:
            return {"response": init_answers[0]}
        return {
            "response": self._aggregate_from(
                final_answers, contributions, task_type
            )
        }

    # ------------------------------------------------------------------
    # Cascade propagation (LLM-local; SCC component calls inside it
    # go through scc_components for parity with v3).
    # ------------------------------------------------------------------
    def _propagate_with_typed_prompts(
        self,
        query,
        init_answers,
        edges,
        rounds: int,
        contributions: List[float],
        task_type: str,
    ) -> List[str]:
        current = list(init_answers)
        adj_in = defaultdict(list)
        for u, v in edges:
            adj_in[v].append(u)

        order = topo_order_by_contributions(
            edges, contributions, self.num_agents
        ) or list(range(self.num_agents))

        for _r in range(max(1, rounds)):
            for i in order:
                preds = adj_in.get(i, [])
                is_leader = i == order[0]
                if not preds and not is_leader:
                    continue

                incoming = [(pid, current[pid]) for pid in preds]
                own_response = current[i]
                sysp = self.role_map.get(
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
                    system_prompt=sysp,
                    temperature=self.temperature,
                )

            # Recompute via shared spectral component.
            embs = self._embed_many(current)
            sims = pairwise_cosine(embs)
            contributions, spec = pc1_contributions(embs)
            self._last_spectral = spec

            # Mid-debate gate A: answer-level plurality.
            canonical, size = count_first_plurality(
                current, contributions, task_type,
                enable_contribution_aggregation=self.enable_contribution_aggregation,
            )
            if (
                self.enable_answer_consensus
                and canonical
                and size >= self.answer_consensus_min_round
            ):
                return current

            # Mid-debate gate B: spectral consensus (delegates to v3-inherited
            # `_check_for_consensus`, which already reads `self._last_spectral`).
            if self._check_for_consensus(sims):
                return current

            # Optional graph reform mid-debate.
            if self.reform and (_r + 1) < rounds:
                edges, _ew = build_diverse_graph(
                    sims, contributions, self.num_agents,
                    top_k=self.top_k,
                    sim_threshold=self.sim_threshold,
                    diversity_p=self.diversity_p,
                    enforce_dag=self.enforce_dag,
                    enable_routing=self.enable_contribution_routing,
                    rng=self._diversity_rng,
                )
                adj_in = defaultdict(list)
                for u, v in edges:
                    adj_in[v].append(u)
                order = topo_order_by_contributions(
                    edges, contributions, self.num_agents
                ) or list(range(self.num_agents))

        return current
