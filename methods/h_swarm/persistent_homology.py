"""
Persistent homology computation for topology analysis.
Used for measuring structural diversity between DAG topologies.

Ported from h_swarm/multi_objective_topology/persistent_homology.py.
Optional dependency on ripser library.
"""

import logging
import numpy as np

try:
    from ripser import ripser
    try:
        from ripser.distance import wasserstein
    except ImportError:
        try:
            from ripser import wasserstein
        except ImportError:
            from scipy.stats import wasserstein_distance as _wasserstein_scipy

            def wasserstein(pd1, pd2, matching=False, order=2):
                if len(pd1) == 0 and len(pd2) == 0:
                    return 0.0
                if len(pd1) == 0:
                    pd1 = np.array([[0, 0]])
                if len(pd2) == 0:
                    pd2 = np.array([[0, 0]])
                return _wasserstein_scipy(pd1.flatten(), pd2.flatten())
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


def compute_filtration_function(A_dag, pi):
    """
    Compute filtration values based on topological rank: f(v) = Rank(v)/N.

    Returns:
        node_filtration: array of node filtration values
        edge_filtration: dict {(u,v): f_value}
    """
    N = A_dag.shape[0]
    physical_to_rank = np.zeros(N, dtype=int)
    for rank in range(N):
        if pi[rank] < N:
            physical_to_rank[pi[rank]] = rank
    node_filtration = physical_to_rank.astype(float) / N
    edge_filtration = {}
    for u in range(N):
        for v in range(N):
            if A_dag[u, v] > 0:
                edge_filtration[(u, v)] = max(node_filtration[u], node_filtration[v])
    return node_filtration, edge_filtration


def build_distance_matrix_from_dag(A_dag, node_filtration, edge_filtration):
    """Build distance matrix for ripser from DAG and filtration."""
    N = A_dag.shape[0]
    dist = np.full((N, N), np.inf)
    np.fill_diagonal(dist, 0.0)
    for (u, v), f_val in edge_filtration.items():
        if u < N and v < N:
            dist[u, v] = f_val
    return dist


def get_persistence_diagram(A_dag, pi, max_dim=1):
    """Compute persistence diagram for a DAG topology."""
    if not RIPSER_AVAILABLE:
        logging.warning("ripser not available, returning empty diagram")
        return [np.array([]).reshape(0, 2)]
    N = A_dag.shape[0]
    if N == 0:
        return [np.array([]).reshape(0, 2)]
    node_filt, edge_filt = compute_filtration_function(A_dag, pi)
    dist = build_distance_matrix_from_dag(A_dag, node_filt, edge_filt)
    result = ripser(dist, maxdim=max_dim, distance_matrix=True)
    return result['dgms']


def get_persistence_diagram_for_individual(ind, decode_fn):
    """Compute persistence diagram for a HybridIndividual."""
    A_dag, _ = decode_fn(ind.d, ind.r)
    r_arr = np.asarray(ind.r, dtype=int)
    return get_persistence_diagram(A_dag, r_arr)


def compute_topology_distance(dgm1, dgm2, dim=0):
    """Compute Wasserstein distance between two persistence diagrams."""
    if not RIPSER_AVAILABLE:
        return 0.0
    d1 = dgm1[dim] if dim < len(dgm1) else np.array([]).reshape(0, 2)
    d2 = dgm2[dim] if dim < len(dgm2) else np.array([]).reshape(0, 2)
    # Filter out infinite persistence
    d1 = d1[np.isfinite(d1[:, 1])] if len(d1) > 0 else d1
    d2 = d2[np.isfinite(d2[:, 1])] if len(d2) > 0 else d2
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        d1 = np.array([[0, 0]])
    if len(d2) == 0:
        d2 = np.array([[0, 0]])
    return float(wasserstein(d1, d2))
