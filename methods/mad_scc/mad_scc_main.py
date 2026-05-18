"""mad_scc: vanilla mad_vote baseline + SCC modules from `methods.scc_components`.

Refactored replacement for `methods.mad_vote_scc`. Identical CLI and config
surface (same six A0-A5 ablation flags + the cached initial-pool replay
hook + the diagnostic JSONL schema), but every SCC component call is
delegated to `methods.scc_components` so behaviour is bit-for-bit aligned
with `soo_scc` / `soo_centered_v3`. This fixes the three failures of the
old `mad_vote_scc` (string-equality grouping, weight-first lexicographic
ordering, prompt-only "DAG" routing without dagify/topo/diversity/
sim_threshold).

The mad_vote BASELINE — fully-connected concurrent broadcast (all agents
speak every round, see each peer's *previous-round* response) — is
preserved unchanged; only the SCC layer is swapped.

Three small additions vs. mad_vote_scc:
  - `equiv_protocol`: "task_aware" (default; v3-faithful via math_is_equiv /
    mcq_is_equiv) or "string" (legacy escape hatch reproducing
    mad_vote_scc's `_canonicalize` behaviour for a controlled comparison).
  - `sim_threshold`, `diversity_p`, `enforce_dag`: routing knobs adopted
    from v3 reference; only consulted when `enable_routing` is True.
  - Routing edges built once per round via `build_diverse_graph`; each
    agent's peers come from `[u for (u,v) in edges if v == i]` ordered by
    the contribution-priority topological position. No mid-loop context
    mutation — peers always see the previous-round snapshot, preserving
    concurrent broadcast semantics.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from methods.mad_vote.mad_vote_main import MAD_Vote_Main
from methods.mad_vote.extractor import (
    _normalize_number_str,
    extract_answer,
    extract_gold,
)
from methods.mad_vote.prompts import get_debate_user_prompt, get_initial_user_prompt
from methods.scc_components import (
    build_diverse_graph,
    count_first_plurality,
    is_spectral_consensus,
    pairwise_cosine,
    pc1_contributions,
    topo_order_by_contributions,
)
from methods.soo_math.math_answer_utils import (
    is_equiv as math_is_equiv,
    strip_string,
)


# ---------------------------------------------------------------------------
# Shared initial-response pool (cached round-0 replay).
# Same protocol as mad_vote_scc so existing diagnostic JSONLs in
# results_archive/ are drop-in pool sources.
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


# ---------------------------------------------------------------------------
# Process-wide singleton sentence-transformer model.
# ---------------------------------------------------------------------------

_GLOBAL_EMB_LOCK = threading.Lock()
_GLOBAL_EMB_MODEL = None


def _get_emb_model(name: str):
    global _GLOBAL_EMB_MODEL
    if _GLOBAL_EMB_MODEL is not None:
        return _GLOBAL_EMB_MODEL
    with _GLOBAL_EMB_LOCK:
        if _GLOBAL_EMB_MODEL is None:
            from sentence_transformers import SentenceTransformer
            try:
                _GLOBAL_EMB_MODEL = SentenceTransformer(name, trust_remote_code=True)
            except Exception:
                _GLOBAL_EMB_MODEL = SentenceTransformer(
                    name, device="cpu", trust_remote_code=True
                )
    return _GLOBAL_EMB_MODEL


class MAD_SCC_Main(MAD_Vote_Main):
    """5-agent mad_vote + SCC modules (Triggering / Routing / Aggregation),
    all delegating to methods.scc_components for parity with soo_scc."""

    JUDGE_PROTOCOL = "xverify-cached-on-canonical"

    def __init__(self, general_config, method_config_name=None):
        super().__init__(general_config, method_config_name=method_config_name)

        mc = self.method_config
        self.enable_triggering = bool(mc.get("enable_triggering", False))
        self.enable_routing = bool(mc.get("enable_routing", False))
        self.enable_weighted_aggregation = bool(
            mc.get("enable_weighted_aggregation", False)
        )
        # Master switch for plurality on code task_type:
        #   True  → A0 baseline path: BLEU plurality used as the vote (matches
        #           dylan-style aggregation, no early stop because triggering
        #           is off).
        #   False → SCC variant path: plurality is fully skipped; `_vote`
        #           returns the argmax-contribution agent's code, and
        #           `_trigger_hit` ignores the size-based plurality check.
        # Default True preserves math / mcq behaviour (where plurality is
        # always the aggregation primitive).
        self.enable_answer_consensus = bool(
            mc.get("enable_answer_consensus", True)
        )
        self.top_k = int(mc.get("top_k", 2))
        self.sim_threshold = float(mc.get("sim_threshold", 0.0))
        self.diversity_p = float(mc.get("diversity_p", 0.0))
        self.enforce_dag = bool(mc.get("enforce_dag", True))
        self.variance_consensus_thr = float(mc.get("variance_consensus_thr", 0.05))
        self.answer_consensus_min = int(mc.get("answer_consensus_min", 3))
        self.equiv_protocol = str(mc.get("equiv_protocol", "task_aware")).lower()
        if self.equiv_protocol not in ("task_aware", "string"):
            self.equiv_protocol = "task_aware"
        self.emb_model_name = mc.get("emb_model", "model/all-MiniLM-L6-v2")

        import random as _random
        seed_base = int(mc.get("random_seed", 101))
        self._diversity_rng = _random.Random(seed_base + 9157)
        self._emb_call_lock = threading.Lock()

        # Optional shared initial-response pool.
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

    # ------------------------------------------------------------------
    # Embedding access (shared singleton; same pattern as mad_vote_scc).
    # ------------------------------------------------------------------
    def _embed_many(self, texts: List[str]) -> List[np.ndarray]:
        with self._emb_call_lock:
            model = _get_emb_model(self.emb_model_name)
            embs = model.encode(texts, batch_size=8, normalize_embeddings=True)
        return [np.asarray(e, dtype=float) for e in embs]

    # ------------------------------------------------------------------
    # Voting helpers — preserve mad_vote_scc's diagnostic schema while
    # delegating the lex-score logic to scc_components.count_first_plurality.
    # ------------------------------------------------------------------
    def _legacy_canonical(self, ans: Optional[str]) -> Optional[str]:
        """Reproduces mad_vote_scc._canonicalize for `equiv_protocol=string`."""
        if ans is None:
            return None
        if self.task_type == "mcq":
            return ans.strip().upper()
        n = _normalize_number_str(ans)
        return n if n is not None else ans.strip()

    def _group_for_diag(
        self, extracted: List[Optional[str]], contributions: List[float]
    ) -> Tuple[List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
        """Group by equivalence relation and produce per-group records.

        Returns (groups, agent_to_group) where:
          groups[k] = {"canonical": str, "size": int, "weight": float,
                       "members": [(agent_idx, ext)], "first_orig": str}
          agent_to_group[i] = the group dict that contains agent i
                              (or None if the agent's ext was empty).

        Uses task_aware equivalence by default; falls back to string equality
        on `_legacy_canonical` keys when `equiv_protocol=string`.
        """
        groups: List[Dict[str, Any]] = []
        agent_to_group: List[Optional[Dict[str, Any]]] = [None] * len(extracted)

        for i, ext in enumerate(extracted):
            if not ext:
                continue
            placed = False
            if self.equiv_protocol == "task_aware":
                if self.task_type == "mcq":
                    eq = lambda a, b: bool(a) and bool(b) and a.strip().upper() == b.strip().upper()
                elif self.task_type == "math":
                    eq = math_is_equiv
                elif self.task_type == "code":
                    # BLEU-based equivalence on the full extracted code.
                    # Syntactically invalid candidates are filtered upstream
                    # (in extract_answer + this `if not ext` check); BLEU
                    # threshold is 0.9 (dylan's CODE_THRESHOLD).
                    from methods.scc_components.voting import code_is_equiv
                    eq = code_is_equiv
                else:
                    # open: each ext is its own group
                    eq = lambda a, b: a == b
                for g in groups:
                    if eq(ext, g["members"][0][1]):
                        g["members"].append((i, ext))
                        g["size"] += 1
                        g["weight"] += float(contributions[i])
                        agent_to_group[i] = g
                        placed = True
                        break
                if not placed:
                    if self.task_type == "math":
                        canon = strip_string(ext)
                    elif self.task_type == "mcq":
                        canon = ext.strip().upper()
                    elif self.task_type == "code":
                        # Code canonicals are returned verbatim — they are
                        # the executable response surface.
                        canon = ext
                    else:
                        canon = ext
                    g = {
                        "canonical": canon,
                        "size": 1,
                        "weight": float(contributions[i]),
                        "members": [(i, ext)],
                        "first_orig": ext,
                    }
                    groups.append(g)
                    agent_to_group[i] = g
            else:  # equiv_protocol == "string"
                key = self._legacy_canonical(ext)
                for g in groups:
                    if g["canonical"] == key:
                        g["members"].append((i, ext))
                        g["size"] += 1
                        g["weight"] += float(contributions[i])
                        agent_to_group[i] = g
                        placed = True
                        break
                if not placed:
                    g = {
                        "canonical": key,
                        "size": 1,
                        "weight": float(contributions[i]),
                        "members": [(i, ext)],
                        "first_orig": ext,
                    }
                    groups.append(g)
                    agent_to_group[i] = g
        return groups, agent_to_group

    def _vote(
        self, extracted: List[Optional[str]], contributions: List[float]
    ) -> Tuple[Optional[str], Dict[Any, int], Dict[Any, float]]:
        """Tally one round's extracted answers.

        Score per group (count-first lex):
          enable_weighted_aggregation=True  → (size, weight)
          enable_weighted_aggregation=False → (size, -group_index)

        For code task_type with `enable_answer_consensus=False` (SCC variant
        path), plurality is skipped entirely: the winner is the
        argmax-contribution agent's full code. counts/weights are emitted
        with a single entry (the winner) so the diagnostic schema stays
        well-formed but downstream `vote_counts` analysis is meaningless
        for code SCC variants (documented behaviour).

        Returns (winner_orig, counts_dict, weights_dict). When no agent
        produced a non-empty extraction, returns (None, {}, {}).
        """
        if self.task_type == "code" and not self.enable_answer_consensus:
            # SCC variant on code: pick argmax(contributions) among agents
            # whose extracted code is non-empty.
            real = [
                (i, ext) for i, ext in enumerate(extracted)
                if ext is not None and ext != ""
            ]
            if not real:
                return None, {}, {}
            i_best, ext_best = max(
                real, key=lambda p: contributions[p[0]]
            )
            return (
                ext_best,
                {ext_best: 1},
                {ext_best: float(contributions[i_best])},
            )

        groups, _ = self._group_for_diag(extracted, contributions)
        if not groups:
            return None, {}, {}

        if self.enable_weighted_aggregation:
            best = max(groups, key=lambda g: (g["size"], g["weight"]))
        else:
            best = max(
                groups, key=lambda g: (g["size"], -groups.index(g))
            )

        counts = {g["canonical"]: g["size"] for g in groups}
        weights = {g["canonical"]: g["weight"] for g in groups}
        return best["first_orig"], counts, weights

    # ------------------------------------------------------------------
    # Triggering — uses task_aware cluster size (count_first_plurality)
    # plus the spectral trace gate.
    # ------------------------------------------------------------------
    def _trigger_hit(
        self, extracted: List[Optional[str]], spec_diag: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        if not self.enable_triggering:
            return False, None
        # Code task: SCC triggering uses spectral signal only — answer
        # plurality on code is too noisy (BLEU clusters fluctuate) to be a
        # reliable early-stop. This matches the user-requested design:
        # "对 coding task，scc 组件并不需要多数投票...只采用谱共识早停条件".
        if self.task_type == "code":
            if is_spectral_consensus(spec_diag, self.variance_consensus_thr):
                return True, "spectral_trace"
            return False, None
        # Plurality cluster size under task-aware equivalence.
        # `enable_contribution_aggregation=False` makes the size the only
        # signal — contribution sums never affect triggering.
        if self.equiv_protocol == "task_aware" and self.task_type in ("math", "mcq"):
            answers = [ext or "" for ext in extracted]
            # count_first_plurality re-extracts; pass already-extracted strings
            # by faking task_type-specific "raw" replies. For mcq, the bare
            # letter is recognised by extract_answer's regex chain. For math,
            # the bare number wraps into "\\boxed{X}" so the math extractor
            # finds it.
            if self.task_type == "math":
                wrapped = [
                    f"The answer is {a}. \\boxed{{{a}}}" if a else ""
                    for a in answers
                ]
            else:
                wrapped = [f"The answer is ({a})" if a else "" for a in answers]
            _, size = count_first_plurality(
                wrapped, [0.0] * len(wrapped), self.task_type,
                enable_contribution_aggregation=False,
            )
        else:
            # Legacy / string-equiv counting.
            counts: Dict[str, int] = {}
            for ext in extracted:
                if ext is None:
                    continue
                key = (
                    self._legacy_canonical(ext)
                    if self.equiv_protocol == "string"
                    else ext
                )
                counts[key] = counts.get(key, 0) + 1
            size = max(counts.values()) if counts else 0
        if size >= self.answer_consensus_min:
            return True, "answer_plurality"
        if is_spectral_consensus(spec_diag, self.variance_consensus_thr):
            return True, "spectral_trace"
        return False, None

    # ------------------------------------------------------------------
    # Routing — full mesh when off, contribution-DAG when on.
    # ------------------------------------------------------------------
    def _peers_per_agent(
        self,
        embs: List[np.ndarray],
        contributions: List[float],
        n_agents: int,
    ) -> List[List[int]]:
        """Return [peers_for_agent_i for i in range(n)].

        enable_routing=False → full mesh (every agent sees every other).
        enable_routing=True  → build_diverse_graph + topo_order; each i
        receives from {j : (j,i) in edges}, ordered by topo position.
        """
        if not self.enable_routing:
            return [[j for j in range(n_agents) if j != i] for i in range(n_agents)]

        sims = pairwise_cosine(embs)
        edges, _edge_w = build_diverse_graph(
            sims,
            contributions,
            n_agents,
            top_k=self.top_k,
            sim_threshold=self.sim_threshold,
            diversity_p=self.diversity_p,
            enforce_dag=self.enforce_dag,
            enable_routing=True,
            rng=self._diversity_rng,
        )
        order = topo_order_by_contributions(edges, contributions, n_agents) or list(
            range(n_agents)
        )
        order_pos = {a: idx for idx, a in enumerate(order)}

        per_i: List[List[int]] = []
        for i in range(n_agents):
            peers = [u for (u, v) in edges if v == i]
            peers.sort(key=lambda j: order_pos.get(j, j))
            per_i.append(peers)
        return per_i

    # ------------------------------------------------------------------
    # Inference (overrides MAD_Vote_Main.inference).
    # ------------------------------------------------------------------
    def inference(self, sample: Dict) -> Dict:
        query = sample["query"]
        gt = sample.get("gt")
        gold = extract_gold(gt, self.task_type)

        judge_cache: Dict[Optional[str], Tuple[bool, str]] = {
            None: (False, "no-extraction")
        }

        # Code-grading context (used only when task_type == "code").
        code_entry_point = sample.get("entry_point", "")
        code_test = sample.get("test")
        code_test_list = sample.get("test_list", [])
        code_test_setup = sample.get("test_setup_code", "")

        def grade_canonical(canonical: Optional[str]) -> Tuple[bool, str]:
            if not self.emit_diagnostic:
                return False, "emit-disabled"
            if canonical in judge_cache:
                return judge_cache[canonical]
            if self.task_type == "code":
                from evaluations.evaluate_code import grade_code_sample
                verdict = grade_code_sample(
                    canonical,
                    entry_point=code_entry_point,
                    test=code_test,
                    test_list=code_test_list,
                    test_setup_code=code_test_setup,
                )
            else:
                response_text = self._canonical_response_text(canonical)
                verdict = self._judge_or_default(query, response_text, gt)
            judge_cache[canonical] = verdict
            return verdict

        # Per-agent contexts.
        agent_contexts: List[List[Dict[str, str]]] = []
        for role_def in self.roles:
            ctx = [
                {"role": "system", "content": role_def["system_prompt"]},
                {"role": "user", "content": get_initial_user_prompt(query, self.task_type)},
            ]
            agent_contexts.append(ctx)

        # ---------- Round 0 ----------
        cached_init: Optional[List[Dict[str, Any]]] = None
        if self.initial_pool is not None:
            entry = self.initial_pool.get(query)
            if entry is not None and len(entry) == len(agent_contexts):
                cached_init = entry

        initial_records = []
        if cached_init is not None:
            for i, ctx in enumerate(agent_contexts):
                rec = cached_init[i]
                resp = rec.get("raw_response", "") or ""
                ctx.append({"role": "assistant", "content": resp})
                ext = rec.get("extracted_answer")
                is_correct = bool(rec.get("is_correct", False))
                judge_status = rec.get("judge_status", "ok-cached") or "ok-cached"
                judge_cache[ext] = (is_correct, judge_status)
                initial_records.append({
                    "agent_id": i,
                    "role": rec.get("role", self.roles[i]["name"]),
                    "raw_response": resp,
                    "extracted_answer": ext,
                    "is_correct": is_correct,
                    "judge_status": judge_status,
                    "prompt_tokens": int(rec.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(rec.get("completion_tokens", 0) or 0),
                    "tokens": int(rec.get("tokens", 0) or 0),
                    "cached_initial": True,
                })
        else:
            for i, ctx in enumerate(agent_contexts):
                try:
                    resp, n_p, n_c = self._call_llm_with_usage(ctx)
                except Exception as e:
                    resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
                ctx.append({"role": "assistant", "content": resp})
                ext = extract_answer(resp, self.task_type)
                is_correct, judge_status = grade_canonical(ext)
                initial_records.append({
                    "agent_id": i,
                    "role": self.roles[i]["name"],
                    "raw_response": resp,
                    "extracted_answer": ext,
                    "is_correct": is_correct,
                    "judge_status": judge_status,
                    "prompt_tokens": n_p,
                    "completion_tokens": n_c,
                    "tokens": n_p + n_c,
                })

        initial_extracted = [r["extracted_answer"] for r in initial_records]
        initial_correct_flags = [r["is_correct"] for r in initial_records]
        initial_oracle_coverage = any(initial_correct_flags)

        # Spectral analysis (always — the trace + contributions feed
        # downstream checks regardless of which flags are on).
        n_agents = len(initial_records)
        need_spectral = (
            self.enable_triggering
            or self.enable_routing
            or self.enable_weighted_aggregation
        )
        if need_spectral:
            init_responses = [r["raw_response"] for r in initial_records]
            embs = self._embed_many(init_responses)
            contributions, spec_diag = pc1_contributions(embs)
        else:
            embs = []
            contributions = [1.0 / max(1, n_agents)] * n_agents
            spec_diag = {"trace": 0.0, "gap_ratio": 0.0, "lam1": 1e-12, "lam2": 0.0}

        trace_Sc_init = float(spec_diag.get("trace", 0.0))

        # Round-0 vote.
        initial_vote, initial_vote_counts, initial_vote_weights = self._vote(
            initial_extracted, contributions
        )
        initial_vote_correct, initial_vote_judge_status = grade_canonical(initial_vote)

        # Trigger after round 0?
        triggered, trigger_reason = self._trigger_hit(initial_extracted, spec_diag)
        early_exit_round: Optional[int] = 0 if triggered else None

        # Histories (always padded to length rounds_num + 1).
        round_responses: Dict[str, List[Dict]] = {}
        round_vote_history: List[Optional[str]] = [initial_vote]
        round_vote_correct_history: List[bool] = [initial_vote_correct]
        round_vote_judge_status: List[str] = [initial_vote_judge_status]
        round_extracted_history: List[List[Optional[str]]] = [list(initial_extracted)]
        round_correct_history: List[List[bool]] = [list(initial_correct_flags)]
        trace_history: List[float] = [trace_Sc_init]
        contributions_history: List[List[float]] = [list(contributions)]
        trigger_history: List[Optional[str]] = [trigger_reason]
        peers_used_history: List[List[List[int]]] = [[]]
        vote_weights_history: List[Dict[Any, float]] = [dict(initial_vote_weights)]

        # ---------- Rounds 1..rounds_num ----------
        for r in range(1, self.rounds_num + 1):
            if early_exit_round is not None:
                round_responses[str(r)] = []
                round_vote_history.append(round_vote_history[-1])
                round_vote_correct_history.append(round_vote_correct_history[-1])
                round_vote_judge_status.append(round_vote_judge_status[-1])
                round_extracted_history.append(list(round_extracted_history[-1]))
                round_correct_history.append(list(round_correct_history[-1]))
                trace_history.append(trace_history[-1])
                contributions_history.append(list(contributions_history[-1]))
                trigger_history.append(None)
                peers_used_history.append([])
                vote_weights_history.append(dict(vote_weights_history[-1]))
                continue

            # Snapshot peers' last-round responses BEFORE updating any context
            # (concurrent broadcast; peers see the previous round only).
            last_round_responses = [ctx[-1]["content"] for ctx in agent_contexts]

            # Build peer adjacency once per round using the previous round's
            # spectral contributions / embeddings.
            if self.enable_routing and need_spectral:
                # Recompute embeddings from the last-round responses so the
                # routing graph reflects the answers peers will actually see.
                routing_embs = self._embed_many(last_round_responses)
            else:
                routing_embs = embs  # only used if enable_routing
            peers_per_agent = self._peers_per_agent(
                routing_embs, contributions, len(agent_contexts)
            )

            this_round_records = []
            this_round_peers: List[List[int]] = []
            for i, ctx in enumerate(agent_contexts):
                peer_ids = peers_per_agent[i]
                this_round_peers.append(list(peer_ids))
                peer_responses = [last_round_responses[j] for j in peer_ids]
                user_prompt = get_debate_user_prompt(
                    query, peer_responses, self.task_type
                )
                ctx.append({"role": "user", "content": user_prompt})
                try:
                    resp, n_p, n_c = self._call_llm_with_usage(ctx)
                except Exception as e:
                    resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
                ctx.append({"role": "assistant", "content": resp})
                ext = extract_answer(resp, self.task_type)
                is_correct, judge_status = grade_canonical(ext)
                this_round_records.append({
                    "agent_id": i,
                    "role": self.roles[i]["name"],
                    "raw_response": resp,
                    "extracted_answer": ext,
                    "is_correct": is_correct,
                    "judge_status": judge_status,
                    "prompt_tokens": n_p,
                    "completion_tokens": n_c,
                    "tokens": n_p + n_c,
                    "peers_used": peer_ids,
                })

            round_responses[str(r)] = this_round_records
            extracted_r = [rec["extracted_answer"] for rec in this_round_records]
            correct_r = [rec["is_correct"] for rec in this_round_records]

            # Recompute spectral on this round's outputs FIRST so the
            # weighted vote uses fresh PC1 contributions.
            if need_spectral:
                cur_responses = [rec["raw_response"] for rec in this_round_records]
                embs = self._embed_many(cur_responses)
                contributions_r, spec_diag_r = pc1_contributions(embs)
                trace_r = float(spec_diag_r.get("trace", 0.0))
            else:
                contributions_r = [1.0 / len(agent_contexts)] * len(agent_contexts)
                spec_diag_r = {"trace": 0.0, "gap_ratio": 0.0, "lam1": 1e-12, "lam2": 0.0}
                trace_r = 0.0

            vote_r, _vote_r_counts, vote_r_weights = self._vote(
                extracted_r, contributions_r
            )
            vote_r_correct, vote_r_status = grade_canonical(vote_r)

            round_vote_history.append(vote_r)
            round_vote_correct_history.append(vote_r_correct)
            round_vote_judge_status.append(vote_r_status)
            round_extracted_history.append(extracted_r)
            round_correct_history.append(correct_r)
            peers_used_history.append(this_round_peers)
            trace_history.append(trace_r)
            contributions_history.append(contributions_r)
            vote_weights_history.append(dict(vote_r_weights))
            contributions = contributions_r
            spec_diag = spec_diag_r

            triggered, trigger_reason = self._trigger_hit(extracted_r, spec_diag)
            trigger_history.append(trigger_reason)
            if triggered:
                early_exit_round = r

        # ---------- Final ----------
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
            "scc_modules": {
                "enable_triggering": self.enable_triggering,
                "enable_routing": self.enable_routing,
                "enable_weighted_aggregation": self.enable_weighted_aggregation,
                "top_k": self.top_k,
                "sim_threshold": self.sim_threshold,
                "diversity_p": self.diversity_p,
                "enforce_dag": self.enforce_dag,
                "variance_consensus_thr": self.variance_consensus_thr,
                "answer_consensus_min": self.answer_consensus_min,
                "equiv_protocol": self.equiv_protocol,
            },
            "early_exit_round": early_exit_round,
            "trace_history": trace_history,
            "contributions_history": contributions_history,
            "trigger_history": trigger_history,
            "peers_used_history": peers_used_history,
            "vote_weights_history": [
                {
                    (k if k is not None else "__none__"): v
                    for k, v in d.items()
                }
                for d in vote_weights_history
            ],
        }

        response_str = final_vote if final_vote is not None else ""
        if not self.emit_diagnostic:
            return {"response": response_str}
        return {"response": response_str, "diagnostic": diagnostic}
