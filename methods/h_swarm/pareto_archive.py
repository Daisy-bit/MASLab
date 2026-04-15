"""
Pareto archive management for multi-objective optimization.

Ported from h_swarm/multi_objective_topology/pareto_archive.py.
"""

import numpy as np


def dominates(obj1, obj2):
    """Check if obj1 dominates obj2 (minimization)."""
    return bool(np.all(obj1 <= obj2) and np.any(obj1 < obj2))


class ParetoArchive:
    """External archive maintaining the Pareto front."""

    def __init__(self):
        self.solutions = []        # [(d, r, objectives), ...]
        self.crowding_distances = []

    def add_solution(self, d, r, objectives):
        """Add a solution; returns True if accepted."""
        for sol in self.solutions:
            if (np.array_equal(sol[0], d) and np.array_equal(sol[1], r) and
                    np.allclose(sol[2], objectives, rtol=1e-5, atol=1e-8)):
                return False
        if any(dominates(sol[2], objectives) for sol in self.solutions):
            return False
        self.solutions = [sol for sol in self.solutions if not dominates(objectives, sol[2])]
        self.solutions.append((d.copy(), r.copy(), objectives.copy()))
        return True

    def update_archive(self, new_solutions):
        """Batch update. Returns count of added solutions."""
        return sum(1 for d, r, obj in new_solutions if self.add_solution(d, r, obj))

    def calculate_crowding_distance(self):
        """Compute crowding distances for all archived solutions."""
        n = len(self.solutions)
        if n <= 2:
            self.crowding_distances = [float('inf')] * n
            return
        num_obj = len(self.solutions[0][2])
        self.crowding_distances = [0.0] * n
        for m in range(num_obj):
            vals = [sol[2][m] for sol in self.solutions]
            idx = np.argsort(vals)
            self.crowding_distances[idx[0]] = float('inf')
            self.crowding_distances[idx[-1]] = float('inf')
            rng = vals[idx[-1]] - vals[idx[0]]
            if rng > 0:
                for i in range(1, n - 1):
                    self.crowding_distances[idx[i]] += (vals[idx[i + 1]] - vals[idx[i - 1]]) / rng

    def roulette_select(self):
        """Select a guide based on crowding distance (prefer sparse regions)."""
        if not self.solutions:
            raise ValueError("Archive is empty")
        if len(self.solutions) == 1:
            return self.solutions[0][0].copy(), self.solutions[0][1].copy()
        if len(self.crowding_distances) != len(self.solutions):
            self.calculate_crowding_distance()
        dists = np.array(self.crowding_distances)
        finite = np.isfinite(dists)
        if np.any(finite) and np.max(dists[finite]) > 0:
            probs = np.zeros(len(dists))
            probs[finite] = dists[finite] / np.sum(dists[finite])
            inf_count = np.sum(~finite)
            if inf_count > 0:
                probs[~finite] = 1.0 / (len(dists) * inf_count)
        else:
            probs = np.ones(len(dists)) / len(dists)
        probs /= probs.sum()
        idx = np.random.choice(len(self.solutions), p=probs)
        return self.solutions[idx][0].copy(), self.solutions[idx][1].copy()

    def get_pareto_front(self):
        return self.solutions.copy()

    def get_pareto_front_objectives(self):
        return [sol[2] for sol in self.solutions]

    def size(self):
        return len(self.solutions)
