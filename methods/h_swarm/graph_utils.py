"""
Graph decoding utilities for h_swarm.
Converts continuous adjacency matrices to discrete directed acyclic graphs (DAGs).

Ported from h_swarm/graph_decode.py for MASLab integration.
"""

import numpy as np


def softmax(values):
    """Softmax over non-zero entries. Zero entries remain zero."""
    values = np.array(values, dtype=float)
    exp_values = np.exp(values)
    for i in range(len(exp_values)):
        if exp_values[i] == np.exp(0):  # original value was 0 => no edge
            exp_values[i] = 0
    total = np.sum(exp_values)
    if total == 0:
        return np.ones(len(values)) / len(values)
    probs = exp_values / total
    return probs


def top_p_sampling_selection(probs, top_p_threshold):
    """Top-p (nucleus) sampling selection from a probability distribution."""
    prob_index_pairs = sorted(
        enumerate(probs), key=lambda x: x[1], reverse=True
    )
    cum_prob = 0.0
    selected_indices = []
    for idx, prob in prob_index_pairs:
        cum_prob += prob
        selected_indices.append(idx)
        if cum_prob > top_p_threshold:
            break

    selected_probs = np.array([probs[i] for i in selected_indices], dtype=float)
    selected_probs = np.exp(selected_probs)
    for i in range(len(selected_probs)):
        if selected_probs[i] == np.exp(0):
            selected_probs[i] = 0
    total = np.sum(selected_probs)
    if total == 0:
        selected_probs = np.ones(len(selected_probs)) / len(selected_probs)
    else:
        selected_probs = selected_probs / total

    return np.random.choice(selected_indices, p=selected_probs)


def graph_decode(adjacency_matrix, top_p_threshold=0):
    """
    Decode a continuous adjacency matrix into a discrete DAG.

    Args:
        adjacency_matrix: np.ndarray (N, N), continuous values.
        top_p_threshold: float, 0 for deterministic decoding.

    Returns:
        discrete_adjacency_matrix: np.ndarray (N, N), binary DAG.
    """
    n = adjacency_matrix.shape[0]
    discrete = np.zeros((n, n))
    remaining = list(range(n))
    existing = []

    assert np.all(np.diag(adjacency_matrix) == 0), "Diagonal must be zero"

    # Select end point (node with lowest out-degree)
    out_degrees = np.sum(adjacency_matrix, axis=1)
    inv_out = np.array([1.0 / v if v != 0 else 0 for v in out_degrees])
    inv_out = softmax(inv_out)
    end_point = top_p_sampling_selection(inv_out, top_p_threshold)
    existing.append(end_point)
    remaining.remove(end_point)

    # Iteratively add nodes
    while remaining:
        out_degrees = np.sum(adjacency_matrix, axis=1).copy()
        for node in existing:
            out_degrees[node] = 0
        out_degrees = softmax(out_degrees)
        selected_node = top_p_sampling_selection(out_degrees, top_p_threshold)

        # Select an existing node to connect to
        edge_weights = adjacency_matrix[selected_node].copy()
        for node in remaining:
            edge_weights[node] = 0
        edge_weights = softmax(edge_weights)
        target_node = top_p_sampling_selection(edge_weights, top_p_threshold)

        discrete[selected_node, target_node] = 1
        existing.append(selected_node)
        remaining.remove(selected_node)

    return discrete


def topological_sort(discrete_adjacency_matrix):
    """Kahn's algorithm for topological ordering of a DAG."""
    n = discrete_adjacency_matrix.shape[0]
    in_degrees = np.sum(discrete_adjacency_matrix, axis=0).astype(float)
    order = []
    while len(order) < n:
        for node in range(n):
            if in_degrees[node] == 0:
                order.append(node)
                in_degrees[node] = -1
                for j in range(n):
                    if discrete_adjacency_matrix[node, j] == 1:
                        in_degrees[j] -= 1
    return order


def get_active_nodes(discrete_adjacency_matrix):
    """
    Find all nodes on valid paths from sources to sinks via reverse DFS.
    For sparse graphs, only returns nodes that actually participate.
    """
    n = discrete_adjacency_matrix.shape[0]
    out_degrees = np.sum(discrete_adjacency_matrix, axis=1)

    end_nodes = [i for i in range(n) if out_degrees[i] == 0]
    if not end_nodes:
        return set(range(n))

    active = set()

    def reverse_dfs(node):
        if node in active:
            return
        active.add(node)
        for i in range(n):
            if discrete_adjacency_matrix[i, node] == 1:
                reverse_dfs(i)

    for end_node in end_nodes:
        reverse_dfs(end_node)

    return active
