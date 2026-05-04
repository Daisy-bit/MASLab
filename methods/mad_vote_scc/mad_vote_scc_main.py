"""
mad_vote_scc: Vanilla MAD (mad_vote) base + three optional SCC modules.

Three flags, each toggleable independently:
  * enable_triggering          : spectral-trace + answer-plurality early stop.
        After round 0 (and after every subsequent round), check if the 5
        agents have reached genuine consensus. If yes, freeze the current
        round's plurality vote as the final answer and skip all remaining
        rounds.
        Trigger reasons:
          - "answer_plurality" : >= answer_consensus_min agents share the same
            extracted canonical answer.
          - "spectral_trace"   : tr(S_c) < variance_consensus_thr where
            S_c = H S H is the double-centered cosine-similarity Gram matrix.
  * enable_routing             : contribution-guided top-k DAG. Instead of the
        round t -> t+1 fully-connected peer broadcast, each agent only sees
        the top-k peers (excluding itself) ranked by PC1 contribution score.
        diversity_p is hard-coded to 0 (no random swap).
  * enable_weighted_aggregation: contribution-weighted plurality. Round-0 and
        every per-round vote tally each agent's canonical answer with weight
        = its PC1 softmax contribution score (instead of uniform 1/N). When
        all flags off, behaviour is identical to methods/mad_vote.

Diagnostic output schema is a strict superset of mad_vote's, so existing
scripts/diagnostic/analyze_diagnostic.py works without modification. The
`scc_modules` field exposes which flags were active for this run, and
`vote_weights_history` records the per-round contribution weights actually
used for tallying (uniform 1/N when weighted aggregation is off).
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from methods.mad_vote.mad_vote_main import MAD_Vote_Main
from methods.mad_vote.extractor import (
    _normalize_number_str,
    extract_answer,
    extract_gold,
    plurality_vote,
)
from methods.mad_vote.prompts import get_debate_user_prompt, get_initial_user_prompt


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


class MAD_Vote_SCC_Main(MAD_Vote_Main):
    """5-agent MAD with optional spectral-trigger and contribution-guided routing."""

    def __init__(self, general_config, method_config_name=None):
        super().__init__(general_config, method_config_name=method_config_name)

        mc = self.method_config
        self.enable_triggering = bool(mc.get("enable_triggering", False))
        self.enable_routing = bool(mc.get("enable_routing", False))
        self.enable_weighted_aggregation = bool(
            mc.get("enable_weighted_aggregation", False)
        )
        self.top_k = int(mc.get("top_k", 2))
        self.variance_consensus_thr = float(mc.get("variance_consensus_thr", 0.05))
        self.answer_consensus_min = int(mc.get("answer_consensus_min", 3))
        self.emb_model_name = mc.get("emb_model", "model/all-MiniLM-L6-v2")
        self._emb_call_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Spectral helpers (self-contained: no SelfOrg subclassing).
    # ------------------------------------------------------------------
    def _embed_many(self, texts: List[str]) -> List[np.ndarray]:
        with self._emb_call_lock:
            model = _get_emb_model(self.emb_model_name)
            embs = model.encode(texts, batch_size=8, normalize_embeddings=True)
        return [np.asarray(e, dtype=float) for e in embs]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)) or 1.0
        nb = float(np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _double_center(S: np.ndarray) -> np.ndarray:
        row_mean = S.mean(axis=1, keepdims=True)
        col_mean = S.mean(axis=0, keepdims=True)
        total_mean = S.mean()
        return S - row_mean - col_mean + total_mean

    def _spectral_analysis(
        self, answers: List[str]
    ) -> Tuple[float, List[float], List[float]]:
        """Return (trace_Sc, contribution_softmax[N], pc1_abs[N])."""
        n = len(answers)
        if n == 0:
            return 0.0, [], []
        embs = self._embed_many(answers)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = self._cosine(embs[i], embs[j])
        S_c = self._double_center(S)
        trace_Sc = float(np.trace(S_c))
        eig_vals, eig_vecs = np.linalg.eigh(S_c)
        pc1 = np.abs(eig_vecs[:, -1]).tolist()  # last column = largest eigenvalue
        s = sum(pc1)
        if s <= 1e-12:
            contributions = [1.0 / n] * n
        else:
            t = max(1e-6, 1.0 / 10.0)
            mx = max(pc1)
            ex = [math.exp((r - mx) / t) for r in pc1]
            ts = sum(ex) or 1.0
            contributions = [e / ts for e in ex]
        return trace_Sc, contributions, pc1

    @staticmethod
    def _answer_plurality_count(extracted: List[Optional[str]]) -> int:
        counts: Dict[str, int] = {}
        for ext in extracted:
            if ext is None:
                continue
            counts[ext] = counts.get(ext, 0) + 1
        return max(counts.values()) if counts else 0

    def _trigger_hit(
        self, extracted: List[Optional[str]], trace_Sc: float
    ) -> Tuple[bool, Optional[str]]:
        if not self.enable_triggering:
            return False, None
        if self._answer_plurality_count(extracted) >= self.answer_consensus_min:
            return True, "answer_plurality"
        if trace_Sc < self.variance_consensus_thr:
            return True, "spectral_trace"
        return False, None

    def _select_peers(
        self, i: int, n: int, contributions: List[float]
    ) -> List[int]:
        """Return ordered list of peer indices that agent i sees this round.

        - enable_routing OFF : full mesh ([j for j in range(n) if j != i]).
        - enable_routing ON  : top-k peers by contribution descending; ties
          broken by ascending index. Deterministic; diversity_p hard-coded 0.
        """
        if not self.enable_routing:
            return [j for j in range(n) if j != i]
        others = [j for j in range(n) if j != i]
        others.sort(key=lambda j: (-contributions[j], j))
        k = min(self.top_k, len(others))
        return others[:k]

    # ------------------------------------------------------------------
    # Aggregation: equal-weight vs contribution-weighted plurality.
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalize(answer: Optional[str], task_type: str) -> Optional[str]:
        """Same key derivation as mad_vote.extractor.plurality_vote so that
        weighted and equal-weight tallies agree on what counts as the same
        canonical answer."""
        if answer is None:
            return None
        if task_type == "mcq":
            return answer.strip().upper()
        n = _normalize_number_str(answer)
        return n if n is not None else answer.strip()

    def _weighted_plurality(
        self,
        extracted: List[Optional[str]],
        contributions: List[float],
        task_type: str,
    ) -> Tuple[Optional[str], Dict[Any, float], Dict[Any, int]]:
        """Contribution-weighted plurality.

        Each agent contributes its PC1 softmax score to the canonical it
        produced; winner = argmax over canonicals of total weight. Ties
        broken first by agent count (so a 3-agent canonical beats a 1-agent
        canonical when their weights tie), then by first-seen position.
        Records both `weights` and integer `counts` for diagnostic emit.
        Returns the original (non-canonical) extracted string of the
        first agent who voted for the winning canonical, so that downstream
        canonical-keyed judge_cache lookups see the same canonical that any
        agent already produced.
        """
        weights: Dict[Any, float] = {}
        counts: Dict[Any, int] = {}
        first_pos: Dict[Any, int] = {}
        first_orig: Dict[Any, Optional[str]] = {}
        order: List[Any] = []
        n_agents = len(extracted)
        for i, ext in enumerate(extracted):
            key = self._canonicalize(ext, task_type)
            if key not in weights:
                weights[key] = 0.0
                counts[key] = 0
                first_pos[key] = i
                first_orig[key] = ext
                order.append(key)
            w = (
                float(contributions[i])
                if i < len(contributions)
                else (1.0 / n_agents if n_agents else 0.0)
            )
            weights[key] += w
            counts[key] += 1
        real = [k for k in order if k is not None]
        if not real:
            return None, weights, counts
        winner_key = max(
            real, key=lambda k: (weights[k], counts[k], -first_pos[k])
        )
        return first_orig[winner_key], weights, counts

    def _vote(
        self,
        extracted: List[Optional[str]],
        contributions: List[float],
    ) -> Tuple[Optional[str], Dict[Any, int], Dict[Any, float]]:
        """Dispatch round-r tallying.

        Returns (winner_extracted, counts_dict, weights_dict) where:
          - `counts_dict` always reports integer agent counts per canonical
            (matches mad_vote.plurality_vote's second return), so existing
            sanity checks keep working.
          - `weights_dict` reports the contribution sums per canonical (or
            uniform 1/N when aggregation is unweighted).
        Winner: weighted plurality if `enable_weighted_aggregation`, else
        the equal-weight plurality_vote() chosen by mad_vote.
        """
        n = len(extracted)
        if self.enable_weighted_aggregation:
            winner, weights, counts = self._weighted_plurality(
                extracted, contributions, self.task_type
            )
            return winner, counts, weights
        winner, counts = plurality_vote(extracted, self.task_type)
        # Synthesise uniform 1/N weights so the diagnostic schema is symmetric.
        uniform = (1.0 / n) if n else 0.0
        weights = {k: float(v) * uniform for k, v in counts.items()}
        return winner, counts, weights

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

        def grade_canonical(canonical: Optional[str]) -> Tuple[bool, str]:
            if canonical in judge_cache:
                return judge_cache[canonical]
            response_text = self._canonical_response_text(canonical)
            verdict = self._judge_or_default(query, response_text, gt)
            judge_cache[canonical] = verdict
            return verdict

        # Per-agent contexts (identical to mad_vote vanilla).
        agent_contexts: List[List[Dict[str, str]]] = []
        for role_def in self.roles:
            ctx = [
                {"role": "system", "content": role_def["system_prompt"]},
                {"role": "user", "content": get_initial_user_prompt(query, self.task_type)},
            ]
            agent_contexts.append(ctx)

        # ---------- Round 0 ----------
        initial_records = []
        for i, ctx in enumerate(agent_contexts):
            try:
                resp, n_p, n_c = self._call_llm_with_usage(ctx)
            except Exception as e:
                resp, n_p, n_c = f"[CALL_LLM_ERROR] {e}", 0, 0
            ctx.append({"role": "assistant", "content": resp})
            ext = extract_answer(resp, self.task_type)
            is_correct, judge_status = grade_canonical(ext)
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
        initial_oracle_coverage = any(initial_correct_flags)

        # Need spectral metrics if any of the three modules is on (weighted
        # aggregation also depends on PC1 contributions).
        need_spectral = (
            self.enable_triggering
            or self.enable_routing
            or self.enable_weighted_aggregation
        )
        if need_spectral:
            init_responses_for_emb = [r["raw_response"] for r in initial_records]
            trace_Sc_init, contributions, _ = self._spectral_analysis(
                init_responses_for_emb
            )
        else:
            trace_Sc_init = 0.0
            contributions = [1.0 / max(1, len(initial_records))] * len(initial_records)

        # Round-0 vote (uses fresh contributions when weighted aggregation
        # is on, else equal-weight plurality identical to mad_vote).
        initial_vote, initial_vote_counts, initial_vote_weights = self._vote(
            initial_extracted, contributions
        )
        initial_vote_correct, initial_vote_judge_status = grade_canonical(initial_vote)

        # Trigger after round 0?
        triggered, trigger_reason = self._trigger_hit(initial_extracted, trace_Sc_init)
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
        peers_used_history: List[List[List[int]]] = [[]]  # round 0 has no peers
        vote_weights_history: List[Dict[Any, float]] = [dict(initial_vote_weights)]

        # ---------- Rounds 1..rounds_num ----------
        for r in range(1, self.rounds_num + 1):
            if early_exit_round is not None:
                # Pad histories with last values; do NOT issue any LLM call.
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

            # Snapshot peers' last-round responses BEFORE updating any context.
            last_round_responses = [ctx[-1]["content"] for ctx in agent_contexts]

            this_round_records = []
            this_round_peers: List[List[int]] = []
            n_agents = len(agent_contexts)
            for i, ctx in enumerate(agent_contexts):
                # Routing uses the PREVIOUS round's contributions (`contributions`
                # is updated only at the end of this loop body).
                peer_ids = self._select_peers(i, n_agents, contributions)
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
                        "peers_used": peer_ids,
                    }
                )

            round_responses[str(r)] = this_round_records
            extracted_r = [rec["extracted_answer"] for rec in this_round_records]
            correct_r = [rec["is_correct"] for rec in this_round_records]

            # Recompute spectral metrics on this round's outputs FIRST so the
            # weighted vote can use fresh PC1 contributions.
            if need_spectral:
                cur_responses = [rec["raw_response"] for rec in this_round_records]
                trace_r, contributions_r, _ = self._spectral_analysis(cur_responses)
            else:
                trace_r = 0.0
                contributions_r = [1.0 / n_agents] * n_agents

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

            triggered, trigger_reason = self._trigger_hit(extracted_r, trace_r)
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

        # Token total: initial responses + actually-issued debate responses.
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
            # SCC-specific instrumentation
            "scc_modules": {
                "enable_triggering": self.enable_triggering,
                "enable_routing": self.enable_routing,
                "enable_weighted_aggregation": self.enable_weighted_aggregation,
                "top_k": self.top_k,
                "variance_consensus_thr": self.variance_consensus_thr,
                "answer_consensus_min": self.answer_consensus_min,
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
        return {"response": response_str, "diagnostic": diagnostic}
