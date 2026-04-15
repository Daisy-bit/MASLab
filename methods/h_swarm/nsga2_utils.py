"""
NSGA-II utility class for multi-objective optimization.
Implements non-dominated sorting, crowding distance, tournament selection,
and genetic operators for hybrid (d, pi) encoding.

Ported from h_swarm/multi_objective_topology/nsga2_hybrid.py.
"""

import random
import numpy as np

from .individual import HybridIndividual


class NSGA2HybridUtils:
    """NSGA-II utilities for optimizing hybrid individuals with (d, pi) encoding."""

    def __init__(self, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, crossover_prob=0.9, mutation_prob=None):
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def fast_nondominated_sort(self, population):
        """Fast non-dominated sorting (NSGA-II)."""
        fronts = [[]]
        for ind in population:
            ind.domination_count = 0
            ind.dominated_solutions = []
            for other in population:
                if ind.dominates(other):
                    ind.dominated_solutions.append(other)
                elif other.dominates(ind):
                    ind.domination_count += 1
            if ind.domination_count == 0:
                ind.rank = 0
                fronts[0].append(ind)
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for ind in fronts[i]:
                for other in ind.dominated_solutions:
                    other.domination_count -= 1
                    if other.domination_count == 0:
                        other.rank = i + 1
                        temp.append(other)
            i += 1
            fronts.append(temp)
        return fronts

    def calculate_crowding_distance(self, front):
        """Calculate crowding distance for a front."""
        if not front:
            return
        n = len(front)
        for ind in front:
            ind.crowding_distance = 0
        for m in range(len(front[0].objectives)):
            front.sort(key=lambda ind: ind.objectives[m])
            front[0].crowding_distance = 1e9
            front[n - 1].crowding_distance = 1e9
            m_values = [ind.objectives[m] for ind in front]
            scale = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / scale

    def crowding_operator(self, ind, other):
        """Compare two individuals: 1 if ind is better, -1 otherwise."""
        if (ind.rank < other.rank) or \
                (ind.rank == other.rank and ind.crowding_distance > other.crowding_distance):
            return 1
        return -1

    def tournament_selection(self, population):
        """Tournament selection with preference for rank-0 individuals."""
        best_front = [ind for ind in population if ind.rank is not None and ind.rank == 0]
        if len(best_front) >= self.num_of_tour_particips:
            participants = random.sample(best_front, self.num_of_tour_particips)
        elif best_front:
            participants = best_front.copy()
            others = [ind for ind in population if ind not in best_front]
            remaining = self.num_of_tour_particips - len(participants)
            if others:
                participants.extend(random.sample(others, min(remaining, len(others))))
        else:
            participants = random.sample(population, min(self.num_of_tour_particips, len(population)))

        best = None
        for p in participants:
            if best is None or (self.crowding_operator(p, best) == 1 and random.random() <= self.tournament_prob):
                best = p
        return best

    def create_children_d(self, population):
        """Create offspring d-vectors via binary crossover and mutation."""
        children = []
        while len(children) < len(population):
            p1 = self.tournament_selection(population)
            p2 = p1
            while p1 == p2:
                p2 = self.tournament_selection(population)
            c1, c2 = self._binary_crossover(p1.d, p2.d)
            c1 = self._binary_mutation(c1)
            c2 = self._binary_mutation(c2)
            children.append(c1)
            if len(children) < len(population):
                children.append(c2)
        return children[:len(population)]

    def create_children_pi(self, population):
        """Create offspring permutations via PMX crossover and swap mutation."""
        children = []
        while len(children) < len(population):
            p1 = self.tournament_selection(population)
            p2 = p1
            while p1 == p2:
                p2 = self.tournament_selection(population)
            c1, c2 = self._pmx_crossover(p1.r, p2.r)
            c1 = self._swap_mutation(c1)
            c2 = self._swap_mutation(c2)
            children.append(c1)
            if len(children) < len(population):
                children.append(c2)
        return children[:len(population)]

    def _binary_crossover(self, d1, d2):
        """Uniform crossover for binary vectors."""
        if random.random() > self.crossover_prob:
            return d1.copy(), d2.copy()
        mask = np.random.random(len(d1)) < 0.5
        return np.where(mask, d1, d2).astype(int), np.where(mask, d2, d1).astype(int)

    def _binary_mutation(self, d):
        """Bit-flip mutation."""
        prob = self.mutation_prob if self.mutation_prob is not None else 1.0 / len(d)
        d_new = d.copy()
        for i in range(len(d)):
            if random.random() < prob:
                d_new[i] = 1 - d_new[i]
        return d_new

    def _pmx_crossover(self, pi1, pi2):
        """Partially-mapped crossover (PMX) for permutations."""
        if random.random() > self.crossover_prob:
            return pi1.copy().astype(int), pi2.copy().astype(int)
        p1, p2 = np.asarray(pi1, dtype=int), np.asarray(pi2, dtype=int)
        n = len(p1)
        if n < 2:
            return p1.copy(), p2.copy()
        a, b = sorted(random.sample(range(n), 2))

        def _pmx(pa, pb):
            child = np.full(n, -1, dtype=int)
            child[a:b + 1] = pa[a:b + 1]
            mapping = {int(pa[i]): int(pb[i]) for i in range(a, b + 1)}
            for i in range(n):
                if a <= i <= b:
                    continue
                gene = int(pb[i])
                while gene in child:
                    gene = mapping.get(gene, gene)
                    if gene == int(pb[i]) and gene in child:
                        break
                if gene in child:
                    remaining = [g for g in range(n) if g not in child]
                    gene = remaining[0]
                child[i] = gene
            return child

        return _pmx(p1, p2), _pmx(p2, p1)

    def _swap_mutation(self, pi):
        """Swap mutation for permutations."""
        pi_new = np.asarray(pi, dtype=int).copy()
        n = len(pi_new)
        if n < 2:
            return pi_new
        prob = self.mutation_prob if self.mutation_prob is not None else 1.0 / n
        if random.random() < prob:
            i, j = random.sample(range(n), 2)
            pi_new[i], pi_new[j] = pi_new[j], pi_new[i]
        return pi_new

    def environmental_selection(self, parent_pop, child_pop, num_individuals):
        """NSGA-II environmental selection with elitism and deduplication."""
        combined = parent_pop + child_pop
        # Deduplicate
        unique = []
        for ind in combined:
            if not any(np.array_equal(ind.d, u.d) and np.array_equal(ind.r, u.r) for u in unique):
                unique.append(ind)

        fronts = self.fast_nondominated_sort(unique)
        new_pop = []
        fi = 0
        while fi < len(fronts) and len(new_pop) + len(fronts[fi]) <= num_individuals:
            self.calculate_crowding_distance(fronts[fi])
            new_pop.extend(fronts[fi])
            fi += 1
        if len(new_pop) < num_individuals and fi < len(fronts):
            self.calculate_crowding_distance(fronts[fi])
            fronts[fi].sort(key=lambda ind: ind.crowding_distance, reverse=True)
            new_pop.extend(fronts[fi][:num_individuals - len(new_pop)])
        return new_pop
