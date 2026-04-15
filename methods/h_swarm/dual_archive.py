"""
Dual archive management: Convergence Archive (CA) + Diversity Archive (DA).

Ported from h_swarm/multi_objective_topology/dual_archive.py.
Simplified for MASLab integration (persistent homology is optional).
"""

import logging
import random
import numpy as np

from .pareto_archive import dominates


class ConvergenceArchive:
    """
    Convergence Archive (CA): maintains Pareto-optimal solutions
    to approximate the true Pareto front.
    """

    def __init__(self, max_size=20):
        self.max_size = max_size
        self.solutions = []     # [(d, r, objectives), ...]

    def update(self, new_solutions):
        """Update archive with new solutions. Returns count of added."""
        unique_new = []
        for new_sol in new_solutions:
            dup = False
            for existing in self.solutions + unique_new:
                if (np.array_equal(existing[0], new_sol[0]) and
                        np.array_equal(existing[1], new_sol[1])):
                    dup = True
                    break
            if not dup:
                unique_new.append(new_sol)

        combined = self.solutions + unique_new
        fronts = self._fast_nondominated_sort(combined)

        self.solutions = []
        fi = 0
        while fi < len(fronts) and len(self.solutions) + len(fronts[fi]) <= self.max_size:
            if fronts[fi]:
                for idx in fronts[fi]:
                    self.solutions.append(combined[idx])
            fi += 1
        if len(self.solutions) < self.max_size and fi < len(fronts) and fronts[fi]:
            # Sort by crowding distance and fill
            cd = self._crowding_distance(fronts[fi], combined)
            ranked = sorted(zip(fronts[fi], cd), key=lambda x: x[1], reverse=True)
            for idx, _ in ranked:
                if len(self.solutions) >= self.max_size:
                    break
                self.solutions.append(combined[idx])
        return len(unique_new)

    def select(self):
        """Select a solution from the archive."""
        if not self.solutions:
            return None
        return random.choice(self.solutions)

    def _fast_nondominated_sort(self, solutions):
        n = len(solutions)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dominates(solutions[i][2], solutions[j][2]):
                    dom_set[i].append(j)
                elif dominates(solutions[j][2], solutions[i][2]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                fronts[0].append(i)
        k = 0
        while fronts[k]:
            temp = []
            for i in fronts[k]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        temp.append(j)
            k += 1
            fronts.append(temp)
        return fronts

    def _crowding_distance(self, front_indices, solutions):
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        cd = [0.0] * n
        num_obj = len(solutions[front_indices[0]][2])
        for m in range(num_obj):
            vals = [solutions[idx][2][m] for idx in front_indices]
            order = np.argsort(vals)
            cd[order[0]] = float('inf')
            cd[order[-1]] = float('inf')
            rng = vals[order[-1]] - vals[order[0]]
            if rng > 0:
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng
        return cd

    def size(self):
        return len(self.solutions)


class DiversityArchive:
    """
    Diversity Archive (DA): maintains structurally diverse solutions.
    Uses simple objective-space diversity when persistent homology is unavailable.
    """

    def __init__(self, max_size=20):
        self.max_size = max_size
        self.solutions = []

    def update(self, new_solutions):
        """Update with new solutions, maintaining diversity."""
        for sol in new_solutions:
            dup = any(
                np.array_equal(s[0], sol[0]) and np.array_equal(s[1], sol[1])
                for s in self.solutions
            )
            if not dup:
                self.solutions.append(sol)
        self._prune()

    def select(self):
        if not self.solutions:
            return None
        return random.choice(self.solutions)

    def _prune(self):
        """Prune to max_size keeping diverse solutions."""
        if len(self.solutions) <= self.max_size:
            return
        # Use greedy furthest-point sampling in objective space
        objs = np.array([s[2] for s in self.solutions])
        # Normalize
        mins, maxs = objs.min(axis=0), objs.max(axis=0)
        rng = maxs - mins
        rng[rng == 0] = 1
        norm_objs = (objs - mins) / rng

        selected = [0]
        for _ in range(1, self.max_size):
            dists = np.full(len(norm_objs), np.inf)
            for s in selected:
                d = np.linalg.norm(norm_objs - norm_objs[s], axis=1)
                dists = np.minimum(dists, d)
            for s in selected:
                dists[s] = -1
            selected.append(int(np.argmax(dists)))

        self.solutions = [self.solutions[i] for i in selected]

    def size(self):
        return len(self.solutions)
