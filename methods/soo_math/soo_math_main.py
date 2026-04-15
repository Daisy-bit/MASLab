from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from methods.dylan.utils_math import get_examples
from methods.utils import load_config
from methods.soo import SOO_Main

from .math_answer_utils import (
    extract_math_answer,
    min_offdiag_similarity,
    plurality_answer_by_contribution,
)


class SOO_Math_Main(SOO_Main):
    """
    SOO + DyLAN-Math few-shot prompts + contribution-weighted answer plurality.

    Inherits SOO's Perron-Frobenius contribution estimation and the official
    SelfOrg DAG pipeline.  Overrides inference() with math-specific logic:
    debate-style multi-turn context, extracted-answer plurality voting,
    anchor-based fallback, and optional inference tracing.
    """

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_soo_math" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        mc = self.method_config
        self.early_math_consensus_min_agents = int(
            mc.get("early_math_consensus_min_agents", 3)
        )
        self._math_mode = str(mc.get("mode", "complex")).lower()
        self._few_shot_block = get_examples(self._math_mode)
        self._system_prompt = "It's a debate. Explain your reasons at each round thoroughly."
        if self._math_mode == "complex":
            self._system_prompt += "\nFollow the given examples and answer the mathematics problem."

        self.final_anchor_mode = str(mc.get("final_anchor_mode", "single")).lower()
        self.final_anchor_top_k = max(1, int(mc.get("final_anchor_top_k", 2)))
        self.final_anchor_mass = float(mc.get("final_anchor_mass", 0.85))
        self.final_anchor_rel_floor = float(mc.get("final_anchor_rel_floor", 0.92))
        self.trace_inference = bool(mc.get("trace_inference", False))

    # ------------------------------------------------------------------
    # Helpers: embedding / contributions / graph (using official SelfOrg internals)
    # ------------------------------------------------------------------

    def _perron_contributions(self, embeddings_np: np.ndarray) -> List[float]:
        """Perron-Frobenius eigenvector contributions from pre-computed numpy embeddings."""
        S = embeddings_np @ embeddings_np.T
        A = np.exp(S / self.consensus_tau)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return np.abs(eigenvectors[:, -1]).tolist()

    def _select_final_idx(self, emb_lists: List[list], contributions: List[float]) -> int:
        """Index of response nearest to contribution-weighted centroid."""
        centroid = self._weighted_centroid(emb_lists, contributions)
        sims = [self._cosine(centroid, e) for e in emb_lists]
        return int(np.argmax(sims))

    def _build_graph(self, emb_lists: List[list], contributions: List[float], n: int):
        """
        Build DAG using the official SelfOrg pattern.
        Returns (topo_order, adj_in, sim_matrix_np).
        """
        sims = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    sims[i][j] = self._cosine(emb_lists[i], emb_lists[j])

        helpful = []
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
            keep = {j for (j, _, _) in scored[:self.top_k]}
            helpful.append(sorted(list(keep)))

        edges = set()
        edge_w = {}
        for i in range(n):
            for j in helpful[i]:
                edges.add((j, i))
                adj_val = sims[i][j] * (1.0 + n * (contributions[j] - contributions[i]))
                edge_w[(j, i)] = max(0.0, adj_val)

        if self.enforce_dag:
            edges, edge_w = self._dagify(edges, edge_w)

        topo = self._topo_order_by_contributions(edges, contributions) or list(range(n))

        adj_in = defaultdict(list)
        for u, v in edges:
            adj_in[v].append(u)

        sim_matrix = np.array(sims, dtype=np.float64)
        return topo, adj_in, sim_matrix

    # ------------------------------------------------------------------
    # Trace / anchor / prompt helpers (preserved from original)
    # ------------------------------------------------------------------

    def _trace_snapshot(
        self,
        *,
        label: str,
        completed_debate_rounds: Optional[int],
        responses: List[str],
        contributions: Optional[List[float]],
        emb_lists: Optional[List[list]],
        similarity_matrix: Optional[np.ndarray],
        anchor_indices: Optional[Sequence[int]],
    ) -> Dict:
        n = len(responses)
        extracted = [extract_math_answer(r) for r in responses]
        voted_u, sz_u = plurality_answer_by_contribution(
            responses, np.ones(n, dtype=np.float64)
        )
        out: Dict = {
            "label": label,
            "completed_debate_rounds": completed_debate_rounds,
            "agent_responses": [r if r is not None else "" for r in responses],
            "extracted_answers": extracted,
            "plurality_uniform_answer": voted_u,
            "plurality_uniform_size": int(sz_u),
            "final_idx": None,
            "final_idx_extracted": None,
            "plurality_weighted_answer": None,
            "plurality_weighted_size": None,
            "pairwise_sim_min": None,
            "contributions": None,
            "anchor_indices": [int(x) for x in anchor_indices] if anchor_indices is not None else None,
        }

        if contributions is not None and emb_lists is not None:
            c = contributions
            voted_w, sz_w = plurality_answer_by_contribution(responses, c)
            fi = self._select_final_idx(emb_lists, c)
            fe = extracted[fi] if 0 <= fi < len(extracted) else ""
            out["plurality_weighted_answer"] = voted_w
            out["plurality_weighted_size"] = int(sz_w)
            out["final_idx"] = fi
            out["final_idx_extracted"] = fe
            out["contributions"] = [float(x) for x in c]
        elif contributions is not None:
            out["contributions"] = [float(x) for x in contributions]

        if similarity_matrix is not None:
            out["pairwise_sim_min"] = min_offdiag_similarity(similarity_matrix)

        return out

    def _anchor_indices(
        self,
        contributions: List[float],
        emb_lists: Optional[List[list]] = None,
    ) -> List[int]:
        c = np.asarray(contributions, dtype=np.float64).reshape(-1)
        n = len(c)
        if n == 0:
            return []

        mode = self.final_anchor_mode
        if mode in ("weighted_centroid", "centroid", "select_final_response"):
            if emb_lists is None:
                return [int(np.argmax(c))]
            return [self._select_final_idx(emb_lists, contributions)]

        order = [int(i) for i in np.argsort(-c)]

        if mode == "single":
            return [order[0]]

        if mode == "top_k":
            k = min(self.final_anchor_top_k, n)
            return order[:k]

        if mode == "adaptive_mass":
            total = float(np.sum(np.maximum(c, 0.0)))
            if total <= 0:
                return [order[0]]
            target = self.final_anchor_mass * total
            cum = 0.0
            out: List[int] = []
            for idx in order:
                cum += float(max(c[idx], 0.0))
                out.append(idx)
                if cum >= target:
                    break
            return out if out else [order[0]]

        if mode in ("adaptive_rel", "relative", "adaptive_relative"):
            mx = float(np.max(c))
            if mx <= 0:
                return [order[0]]
            thr = self.final_anchor_rel_floor * mx
            out = [idx for idx in order if c[idx] >= thr]
            return out if out else [order[0]]

        return [order[0]]

    @staticmethod
    def _peer_texts_from_anchors(
        anchor_indices: Sequence[int],
        node: int,
        prev_responses: List[str],
    ) -> List[str]:
        own = prev_responses[node] if 0 <= node < len(prev_responses) else ""
        seen: set[str] = set()
        peer_texts: List[str] = []
        for aid in anchor_indices:
            if aid == node or not (0 <= aid < len(prev_responses)):
                continue
            txt = prev_responses[aid]
            if not txt or txt == own or txt in seen:
                continue
            seen.add(txt)
            peer_texts.append(txt)
        return peer_texts

    def _fallback_response_from_anchors(
        self,
        anchor_indices: Sequence[int],
        responses: List[str],
        n: int,
    ) -> str:
        for idx in anchor_indices:
            if not (0 <= idx < n):
                continue
            r = responses[idx]
            ext = extract_math_answer(r)
            if ext:
                return ext
        for idx in anchor_indices:
            if 0 <= idx < n and responses[idx]:
                return responses[idx].strip()
        return ""

    def _prepare_question(self, query: str) -> str:
        return (
            self._few_shot_block
            + f"\n\nPlease solve the problem below.\nProblem: {query}\nAnswer:"
        )

    def _user_debate(self, prepared_question: str, peer_texts: List[str]) -> str:
        prefix = (
            "Follow the given examples and answer the mathematics problem.\n\n"
            + prepared_question
            + "\n\nThese are the solutions to the problem from other agents: "
        )
        for resp in peer_texts:
            text = resp if resp is not None else ""
            prefix += f"\n\nOne agent solution: ```{text}```"
        prefix += (
            "\n\nUsing the reasoning from other agents as additional advice with critical thinking, "
            "can you give an updated answer? Examine your solution and that other agents step by step. "
            "Notice that the former answers might be all wrong."
        )
        return prefix

    def _agent_completion(self, context: List[Dict[str, str]]) -> str:
        completion = self.call_llm(messages=context)
        context.append({"role": "assistant", "content": completion})
        return completion

    def _replace_last_turn_with_user(self, context: List[Dict[str, str]], user_text: str) -> None:
        if (
            len(context) >= 3
            and context[-1].get("role") == "assistant"
            and context[-2].get("role") == "user"
        ):
            context.pop()
            context.pop()
        else:
            context[:] = [{"role": "system", "content": self._system_prompt}]
        context.append({"role": "user", "content": user_text})

    # ------------------------------------------------------------------
    # Main inference (math-specific pipeline)
    # ------------------------------------------------------------------

    def inference(self, sample):
        query = sample["query"]
        prepared = self._prepare_question(query)
        n = self.num_agents
        t = max(1, self.max_rounds)

        trace: Optional[Dict] = None
        if self.trace_inference:
            trace = {
                "trace_schema": "soo_math_v2",
                "stages": [],
                "events": [],
            }

        def pack_out(response: str, exit_reason: str, **exit_extra) -> Dict:
            if not trace:
                return {"response": response}
            trace["exit"] = {"reason": exit_reason, "response": response, **exit_extra}
            return {"response": response, "inference_trace": trace}

        # Round 0: independent answers
        agent_contexts: List[List[Dict[str, str]]] = []
        for _ in range(n):
            agent_contexts.append(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prepared},
                ]
            )

        responses: List[str] = [""] * n
        for i in range(n):
            responses[i] = self._agent_completion(agent_contexts[i])

        if trace:
            trace["stages"].append(
                self._trace_snapshot(
                    label="initial_parallel",
                    completed_debate_rounds=None,
                    responses=responses,
                    contributions=None,
                    emb_lists=None,
                    similarity_matrix=None,
                    anchor_indices=None,
                )
            )

        # Early math consensus on initial answers
        if self.early_math_consensus_min_agents > 0:
            voted0, sz0 = plurality_answer_by_contribution(
                responses, np.ones(n, dtype=np.float64)
            )
            if voted0 and sz0 >= self.early_math_consensus_min_agents:
                return pack_out(
                    voted0,
                    "early_math_consensus_initial",
                    plurality_size=int(sz0),
                    threshold=self.early_math_consensus_min_agents,
                )

        # Embed, compute contributions, build graph
        emb_lists = self._embed_many(responses)
        embeddings_np = np.array(emb_lists, dtype=np.float32)
        contributions = self._perron_contributions(embeddings_np)
        topo, adj_in, sim_matrix = self._build_graph(emb_lists, contributions, n)
        anchor_indices = self._anchor_indices(contributions, emb_lists)

        if trace:
            trace["stages"].append(
                self._trace_snapshot(
                    label="after_embed",
                    completed_debate_rounds=0,
                    responses=responses,
                    contributions=contributions,
                    emb_lists=emb_lists,
                    similarity_matrix=sim_matrix,
                    anchor_indices=anchor_indices,
                )
            )

        # Debate rounds
        completed_debate = 0
        for _round in range(1, t):
            # Early stop by similarity gamma (if configured)
            if hasattr(self, 'early_stop_gamma') and self.early_stop_gamma is not None:
                try:
                    gamma = float(self.early_stop_gamma)
                    sim_min = min_offdiag_similarity(sim_matrix)
                    if sim_min >= gamma:
                        if trace:
                            trace["events"].append(
                                {
                                    "kind": "early_stop_similarity_gamma",
                                    "before_debate_iteration": _round,
                                    "gamma": gamma,
                                    "pairwise_sim_min": sim_min,
                                }
                            )
                        break
                except Exception:
                    pass

            prev_responses = responses
            responses = [""] * n

            for node in topo:
                predecessors = adj_in.get(node, [])
                peer_texts: List[str] = []
                for pred in predecessors:
                    peer_texts.append(responses[pred] if responses[pred] else prev_responses[pred])
                if not predecessors:
                    peer_texts = self._peer_texts_from_anchors(
                        anchor_indices, node, prev_responses
                    )

                user_prompt = (
                    self._user_debate(prepared, peer_texts) if peer_texts else prepared
                )
                self._replace_last_turn_with_user(agent_contexts[node], user_prompt)
                responses[node] = self._agent_completion(agent_contexts[node])

            # Re-embed and rebuild
            emb_lists = self._embed_many(responses)
            embeddings_np = np.array(emb_lists, dtype=np.float32)
            contributions = self._perron_contributions(embeddings_np)
            topo, adj_in, sim_matrix = self._build_graph(emb_lists, contributions, n)
            anchor_indices = self._anchor_indices(contributions, emb_lists)
            completed_debate += 1

            if trace:
                trace["stages"].append(
                    self._trace_snapshot(
                        label=f"after_embed_debate_{completed_debate}",
                        completed_debate_rounds=completed_debate,
                        responses=responses,
                        contributions=contributions,
                        emb_lists=emb_lists,
                        similarity_matrix=sim_matrix,
                        anchor_indices=anchor_indices,
                    )
                )

            # Mid-round math consensus
            if self.early_math_consensus_min_agents > 0:
                voted_m, sz_m = plurality_answer_by_contribution(responses, contributions)
                if voted_m and sz_m >= self.early_math_consensus_min_agents:
                    return pack_out(
                        voted_m,
                        "early_math_consensus_round",
                        plurality_size=int(sz_m),
                        threshold=self.early_math_consensus_min_agents,
                        completed_debate_rounds=completed_debate,
                    )

        # Final output
        voted, _sz = plurality_answer_by_contribution(responses, contributions)
        if voted:
            return pack_out(voted, "final_plurality_weighted", completed_debate_rounds=completed_debate)

        anchor_indices = self._anchor_indices(contributions, emb_lists)
        fb = self._fallback_response_from_anchors(anchor_indices, responses, n)
        return pack_out(fb, "anchor_fallback", completed_debate_rounds=completed_debate)
