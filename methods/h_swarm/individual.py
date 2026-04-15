"""
Hybrid individual class for NSGA-II multi-objective optimization.
Contains binary connectivity vector d and permutation priority gene r (pi).

Ported from h_swarm/multi_objective_topology/individual_hybrid.py.
"""

import numpy as np


class HybridIndividual:
    """
    Hybrid individual with dual-chromosome encoding:
    - d: binary connectivity vector, shape (D,)
    - r: permutation vector pi, shape (N,), dtype=int, values are [0..N-1]
    """

    def __init__(self):
        self.d = None
        self.r = None
        self.objectives = None       # shape (2,): [-f_utility, f_sparsity]
        self.rank = None             # NSGA-II front rank
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (np.array_equal(self.d, other.d) and
                    np.array_equal(self.r, other.r))
        return False

    def dominates(self, other):
        """Check if self dominates other (minimization)."""
        and_cond = True
        or_cond = False
        for a, b in zip(self.objectives, other.objectives):
            and_cond = and_cond and a <= b
            or_cond = or_cond or a < b
        return and_cond and or_cond
