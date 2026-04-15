"""
Ablation: No debate rounds. Only round-0 independent sampling + centroid selection.
Tests whether multi-agent debate adds value over single-round parallel sampling.
"""

from .selforg_main import SelfOrg_Main


class SelfOrg_NoDebate(SelfOrg_Main):

    def inference(self, sample):
        query = sample["query"]
        reference = sample.get("reference", None)

        # Round 0 only: independent answers
        init_answers = []
        for i in range(self.num_agents):
            system_prompt = self.role_map.get(self.roles[i], self.role_map["Assistant"])
            prompt = self._init_prompt(query)
            ans = self._call_llm(prompt=prompt, system_prompt=system_prompt, temperature=self.temperature)
            init_answers.append(ans)

        # Contribution estimation + aggregation (no debate)
        contributions = self._approx_shapley(init_answers, reference)

        if self.aggregate_mode == "single":
            best_idx = max(range(self.num_agents), key=lambda i: contributions[i])
            response = init_answers[best_idx]
        else:
            final_embs = self._embed_many(init_answers)
            weights = contributions
            if sum(weights) <= 1e-9:
                weights = [1.0 / self.num_agents] * self.num_agents
            agg = self._weighted_centroid(final_embs, weights)
            nearest = max(range(self.num_agents), key=lambda i: self._cosine(agg, final_embs[i]))
            response = init_answers[nearest]

        return {"response": response}
