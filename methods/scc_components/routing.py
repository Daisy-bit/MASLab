"""Routing — diversity-augmented contribution graph + DAG enforcement + topo order.

Combines:
  * `_build_diverse_graph` from soo_centered_v3_main.py:374-438
  * `_dagify`               from selforg_main.py:246-304
  * `_topo_order_by_contributions` from selforg_main.py:306-336

Pure functions; caller provides sims / contributions / RNG.
"""

from __future__ import annotations

import heapq
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


Edge = Tuple[int, int]
EdgeSet = Set[Edge]
EdgeWeights = Dict[Edge, float]


# ---------------------------------------------------------------------------
# DAG enforcement
# ---------------------------------------------------------------------------

def dagify(edges: EdgeSet, edge_w: EdgeWeights, n: int) -> Tuple[EdgeSet, EdgeWeights]:
    """Repeatedly DFS for a cycle and remove its weakest edge until the graph
    is acyclic. Algorithm copied from `selforg_main.SelfOrg_Main._dagify`
    (lines 246-304); behaviour is identical.

    Returns NEW (edges, edge_w) sets — does not mutate inputs.
    """
    E: EdgeSet = set(edges)
    W: EdgeWeights = dict(edge_w)

    def build_adj(es: EdgeSet) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = defaultdict(list)
        for a, b in es:
            adj[a].append(b)
        return adj

    while True:
        adj = build_adj(E)
        color = {u: 0 for u in range(n)}
        parent: Dict[int, int] = {u: None for u in range(n)}
        removed_any = False

        def dfs(u: int) -> None:
            nonlocal removed_any
            color[u] = 1
            for v in adj.get(u, []):
                if removed_any:
                    return
                if color[v] == 0:
                    parent[v] = u
                    dfs(v)
                elif color[v] == 1:
                    # back edge → cycle: rebuild path v → ... → u → v
                    cycle_nodes = [u]
                    x = u
                    while x != v and parent.get(x) is not None:
                        x = parent[x]
                        cycle_nodes.append(x)
                    cycle_nodes.reverse()
                    cycle_edges: List[Edge] = []
                    for i in range(len(cycle_nodes) - 1):
                        cycle_edges.append((cycle_nodes[i], cycle_nodes[i + 1]))
                    cycle_edges.append((u, v))
                    weakest = min(
                        cycle_edges,
                        key=lambda e: (W.get(e, 0.0), e[0], e[1]),
                    )
                    E.discard(weakest)
                    W.pop(weakest, None)
                    removed_any = True
                    return
            color[u] = 2

        for u in range(n):
            if removed_any:
                break
            if color[u] == 0:
                dfs(u)

        if not removed_any:
            break

    return E, W


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------

def topo_order_by_contributions(
    edges: EdgeSet, contributions: Sequence[float], n: int
) -> List[int]:
    """Kahn's algorithm with priority: higher contribution first, then
    smaller index. Returns [] if the graph contains a cycle (caller should
    have run `dagify` first). Algorithm copied from
    `selforg_main.SelfOrg_Main._topo_order_by_contributions` (lines 306-336).
    """
    indeg = [0] * n
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    heap: List[Tuple[float, int]] = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(heap, (-contributions[i], i))

    order: List[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (-contributions[v], v))

    return order if len(order) == n else []


# ---------------------------------------------------------------------------
# Diversity-augmented contribution graph
# ---------------------------------------------------------------------------

def build_diverse_graph(
    sims: np.ndarray,
    contributions: Sequence[float],
    n: int,
    *,
    top_k: int,
    sim_threshold: float = 0.0,
    diversity_p: float = 0.0,
    enforce_dag: bool = True,
    enable_routing: bool = True,
    rng: random.Random = None,
) -> Tuple[EdgeSet, EdgeWeights]:
    """Build the directed peer graph used by SCC routing.

    Algorithm (copied from soo_centered_v3._build_diverse_graph):

      For each agent i:
        score(j → i) = sim[i][j] · (1 + n·(c_j − c_i))
        Filter peers by base sim ≥ sim_threshold
        Sort by (adj_score, c_j) descending; take top_k
        With probability diversity_p, swap each kept peer for a random
          non-self peer (without re-picking already-swapped indices)

      Build edges {(j, i)} for each i's kept peers, weight = max(0, score)
      Optionally `dagify` to enforce no cycles.

    Ablation hooks:
      enable_routing=False → ignore contributions/sims entirely; build a
        full mesh with weight 1.0 (still optionally dagified). This matches
        v3's behaviour when `enable_contribution_routing=False`.

    rng: caller-supplied random.Random instance (deterministic per seed). If
    None and diversity_p > 0, a fresh non-seeded Random() is used.
    """
    if rng is None:
        rng = random.Random()

    if not enable_routing:
        edges: EdgeSet = {(j, i) for i in range(n) for j in range(n) if j != i}
        edge_w: EdgeWeights = {e: 1.0 for e in edges}
        if enforce_dag:
            edges, edge_w = dagify(edges, edge_w, n)
        return edges, edge_w

    helpful: List[List[int]] = []
    for i in range(n):
        scored = []
        for j in range(n):
            if j == i:
                continue
            base = sims[i][j]
            adj = base * (1.0 + n * (contributions[j] - contributions[i]))
            scored.append((j, adj, base))

        scored = [p for p in scored if p[2] >= sim_threshold]
        scored.sort(
            key=lambda x: (x[1], contributions[x[0]]), reverse=True
        )
        keep = [j for (j, _, _) in scored[:top_k]]

        if diversity_p > 0.0:
            all_others = [j for j in range(n) if j != i]
            swapped: List[int] = []
            for j in keep:
                if rng.random() < diversity_p:
                    pool = [k for k in all_others if k != j and k not in swapped]
                    if pool:
                        swapped.append(rng.choice(pool))
                        continue
                swapped.append(j)
            keep = swapped

        helpful.append(sorted(set(keep)))

    edges: EdgeSet = set()
    edge_w: EdgeWeights = {}
    for i in range(n):
        for j in helpful[i]:
            edges.add((j, i))
            adj_val = sims[i][j] * (
                1.0 + n * (contributions[j] - contributions[i])
            )
            edge_w[(j, i)] = max(0.0, adj_val)

    if enforce_dag:
        edges, edge_w = dagify(edges, edge_w, n)

    return edges, edge_w
