"""
Multi-objective graph decoding utilities for h_swarm.
Handles (d, r) encoding: binary connectivity vector + permutation priority gene.

Ported from h_swarm/multi_objective_topology/graph_decode_multiobj.py.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def vector_to_upper_triangular(d, N):
    """Convert 1D binary vector d to NxN strict upper triangular matrix."""
    D = N * (N - 1) // 2
    assert len(d) == D, f"d length should be {D}, got {len(d)}"
    U = np.zeros((N, N), dtype=int)
    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            U[i, j] = d[k]
            k += 1
    return U


def upper_triangular_to_vector(U_or_M, N):
    """Extract upper triangular elements from matrix into 1D vector."""
    assert U_or_M.shape == (N, N)
    D = N * (N - 1) // 2
    d = np.zeros(D, dtype=int)
    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            d[k] = U_or_M[i, j]
            k += 1
    return d


def decode_and_repair(d, r):
    """
    Decode particle (d, r) and apply repair strategy (keep largest connected component).

    Args:
        d: binary connectivity vector, length D = N*(N-1)/2
        r: permutation vector pi (int) or continuous ranking vector (float)

    Returns:
        (A_dag, A_dag_original, sink_node, is_valid)
    """
    N = len(r)
    D = N * (N - 1) // 2
    assert len(d) == D

    # Determine topological order from r
    r_arr = np.asarray(r)
    is_perm = False
    if r_arr.shape == (N,) and np.issubdtype(r_arr.dtype, np.integer):
        r_int = r_arr.astype(int, copy=False)
        is_perm = (len(np.unique(r_int)) == N and r_int.min() == 0 and r_int.max() == N - 1)
        pi = r_int if is_perm else np.argsort(r_arr)
    elif r_arr.shape == (N,):
        r_round = np.rint(r_arr).astype(int)
        if (np.allclose(r_arr, r_round, rtol=0, atol=0) and
                len(np.unique(r_round)) == N and r_round.min() == 0 and r_round.max() == N - 1):
            pi = r_round
        else:
            pi = np.argsort(r_arr)
    else:
        pi = np.argsort(r_arr)

    # Permutation matrix P
    P = np.zeros((N, N), dtype=int)
    for i in range(N):
        P[i, pi[i]] = 1

    # Map d to upper triangular then to physical space
    U = vector_to_upper_triangular(d, N)
    A_dag_original = P.T @ U @ P

    # Repair: detect connected components, keep largest
    A_dag_original, sink_node, is_valid = _repair_graph(A_dag_original, pi)

    if is_valid:
        A_dag = A_dag_original.copy()
    else:
        A_dag = np.zeros((N, N), dtype=int)
        A_dag_original = np.zeros((N, N), dtype=int)
        sink_node = None

    return A_dag, A_dag_original, sink_node, is_valid


def _repair_graph(A_full, pi):
    """Keep the largest non-trivial connected component, disconnect the rest."""
    N = A_full.shape[0]
    A_symmetric = ((A_full + A_full.T) > 0).astype(int)
    n_components, labels = connected_components(
        csgraph=csr_matrix(A_symmetric), directed=False, return_labels=True
    )

    components = {}
    for node in range(N):
        comp_id = labels[node]
        components.setdefault(comp_id, []).append(node)

    non_trivial = {cid: nodes for cid, nodes in components.items() if len(nodes) > 1}

    if len(non_trivial) == 0:
        return np.zeros((N, N), dtype=int), None, False

    if len(non_trivial) == 1:
        max_nodes = list(non_trivial.values())[0]
    else:
        # Pick largest; tie-break by lowest rank
        max_size = max(len(nodes) for nodes in non_trivial.values())
        candidates = [(cid, nodes) for cid, nodes in non_trivial.items() if len(nodes) == max_size]
        if len(candidates) == 1:
            max_nodes = candidates[0][1]
        else:
            best_nodes = None
            best_rank = float('inf')
            for _, nodes in candidates:
                ranks = [np.where(pi == n)[0][0] if n in pi else N for n in nodes]
                mr = min(ranks)
                if mr < best_rank:
                    best_rank = mr
                    best_nodes = nodes
            max_nodes = best_nodes

    A_repaired = _disconnect_other_components(A_full, max_nodes, components)
    sink_node = _find_sink(A_repaired, max_nodes, pi)
    return A_repaired, sink_node, True


def _disconnect_other_components(A_full, max_component_nodes, all_components):
    """Zero out edges for nodes not in the max component."""
    N = A_full.shape[0]
    A_repaired = A_full.copy()
    max_set = set(max_component_nodes)
    for _, nodes in all_components.items():
        node_set = set(nodes)
        if not node_set.issubset(max_set):
            for u in nodes:
                for v in nodes:
                    if u != v:
                        A_repaired[u, v] = 0
                for other in range(N):
                    if other not in node_set:
                        A_repaired[u, other] = 0
                        A_repaired[other, u] = 0
    return A_repaired


def _find_sink(A_repaired, component_nodes, pi):
    """Find the sink node (out-degree 0) with the highest rank in pi."""
    if not component_nodes:
        return None
    sinks = []
    for node in component_nodes:
        out_deg = sum(1 for other in component_nodes if other != node and A_repaired[node, other] > 0)
        if out_deg == 0:
            sinks.append(node)
    if not sinks:
        return None
    if len(sinks) == 1:
        return sinks[0]
    # Pick the one with highest rank (latest in topological order)
    best = max(sinks, key=lambda n: (np.where(pi == n)[0][0] if n in pi else -1))
    return best


def decode_particle_to_dag(d, r):
    """Decode (d, r) particle to DAG. Returns (A_dag, A_dag_original)."""
    A_dag, A_dag_original, _, _ = decode_and_repair(d, r)
    return A_dag, A_dag_original


def get_edge_count(A_dag):
    """Count edges in the DAG."""
    return int(np.sum(A_dag))
