"""
Ablation: Random communication graph with debate.
Replaces SelfOrg's contribution-based DAG with random peer selection and
random edge direction. Tests whether structured orchestration matters.
"""

import random as _rng
from collections import defaultdict

from .selforg_main import SelfOrg_Main


class SelfOrg_RandomGraph(SelfOrg_Main):

    def inference(self, sample):
        query = sample["query"]
        reference = sample.get("reference", None)

        # 1) Round 0: independent answers (same as SelfOrg)
        init_answers = []
        for i in range(self.num_agents):
            system_prompt = self.role_map.get(self.roles[i], self.role_map["Assistant"])
            prompt = self._init_prompt(query)
            ans = self._call_llm(prompt=prompt, system_prompt=system_prompt, temperature=self.temperature)
            init_answers.append(ans)

        # 2) Random peer selection (ignoring similarity / contribution)
        helpful = []
        for i in range(self.num_agents):
            others = [j for j in range(self.num_agents) if j != i]
            k = min(self.top_k, len(others))
            keep = set(_rng.sample(others, k))
            helpful.append(sorted(keep))

        # 3) Random edge direction (not contribution-based)
        edges = set()
        edge_w = {}
        for i in range(self.num_agents):
            for j in helpful[i]:
                if _rng.random() > 0.5:
                    edges.add((j, i))
                    edge_w[(j, i)] = 1.0
                else:
                    edges.add((i, j))
                    edge_w[(i, j)] = 1.0

        # 4) DAG enforcement
        if self.enforce_dag:
            edges, edge_w = self._dagify(edges, edge_w)

        # 5) Propagation with debate (same mechanism, random graph)
        contributions = self._approx_shapley(init_answers, None)
        final_answers = self._propagate_on_dag(
            query, init_answers, edges,
            rounds=self.max_rounds, contributions=contributions,
        )
        contributions = self._approx_shapley(final_answers, reference)

        # 6) Aggregation (same as SelfOrg)
        if self.aggregate_mode == "single":
            best_idx = max(range(self.num_agents), key=lambda i: contributions[i])
            response = final_answers[best_idx]
        else:
            final_embs = self._embed_many(final_answers)
            weights = contributions
            if sum(weights) <= 1e-9:
                weights = [1.0 / self.num_agents] * self.num_agents
            agg = self._weighted_centroid(final_embs, weights)
            nearest = max(range(self.num_agents), key=lambda i: self._cosine(agg, final_embs[i]))
            response = final_answers[nearest]

        return {"response": response}
