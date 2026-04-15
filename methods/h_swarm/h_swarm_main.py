"""
HSwarm_Main: PSO-based topology optimization + graph-based multi-agent inference.

Integrates h_swarm's particle swarm optimization for topology search into MASLab's
unified MAS framework. Supports:
  1. Pre-optimized topology inference (load adjacency matrix from file)
  2. PSO topology optimization via optimizing(val_data)
  3. Graph-based multi-step inference through DAGs via inference(sample)
"""

import os
import random
import logging
from typing import Optional

import numpy as np

from methods.mas_base import MAS
from methods.utils import load_config
from .graph_utils import graph_decode, topological_sort, get_active_nodes

logger = logging.getLogger(__name__)


# Default prompt templates (same as original h_swarm)
DEFAULT_FIRST_INSTRUCTION = "Please answer the following question."
DEFAULT_NON_LAST_INSTRUCTION = (
    "Please answer the following question with the help of previous responses, "
    "feel free to ignore wrong or unhelpful responses."
)
DEFAULT_LAST_INSTRUCTION = (
    "Please answer the following question with the help of previous responses, "
    "feel free to ignore wrong or unhelpful responses."
)


class HSwarm_Main(MAS):
    """
    PSO-based topology optimization for multi-agent inference.

    Usage:
      - With pre-optimized topology: set adjacency_matrix_path in config
      - With PSO optimization: call optimizing(val_data) first, then inference()
      - Without optimization: uses random topology (num_agents nodes)
    """

    def __init__(self, general_config, method_config_name: Optional[str] = None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        self.method_config = load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml")
        )

        # Topology parameters
        self.num_agents = int(self.method_config.get("num_agents", 5))
        self.use_discrete_matrix = bool(self.method_config.get("use_discrete_matrix", False))

        # PSO parameters
        self.graph_num = int(self.method_config.get("graph_num", 10))
        self.max_iteration = int(self.method_config.get("max_iteration", 50))
        self.patience = int(self.method_config.get("patience", 10))
        self.inertia = float(self.method_config.get("inertia", 0.4))
        self.cognitive_coeff = float(self.method_config.get("cognitive_coeff", 0.3))
        self.social_coeff = float(self.method_config.get("social_coeff", 0.3))
        self.repel_coeff = float(self.method_config.get("repel_coeff", 0.0))
        self.step_length = float(self.method_config.get("step_length", 1.0))
        self.step_length_factor = float(self.method_config.get("step_length_factor", 1.0))
        self.minimum_step_length = float(self.method_config.get("minimum_step_length", 0.1))
        self.weight_randomness = bool(self.method_config.get("weight_randomness", True))

        # Prompt templates
        self.first_instruction = self.method_config.get("first_instruction", DEFAULT_FIRST_INSTRUCTION)
        self.non_last_instruction = self.method_config.get("non_last_instruction", DEFAULT_NON_LAST_INSTRUCTION)
        self.last_instruction = self.method_config.get("last_instruction", DEFAULT_LAST_INSTRUCTION)

        # Load pre-optimized adjacency matrix if provided
        self.adjacency_matrix = None
        self._cached_dag = None  # decoded DAG cache
        adj_path = self.method_config.get("adjacency_matrix_path", None)
        if adj_path and os.path.exists(adj_path):
            self.adjacency_matrix = np.load(adj_path)
            self.num_agents = self.adjacency_matrix.shape[0]
            # Pre-optimized matrices are typically already discrete
            self.use_discrete_matrix = True

    # ----------------------------------------------------------------
    # Inference: graph-based multi-step reasoning through a DAG
    # ----------------------------------------------------------------

    def inference(self, sample):
        """
        Run graph-based multi-agent inference on a single sample.

        Each node in the DAG represents an agent that calls the LLM.
        Nodes are processed in topological order:
          - Source nodes: get the question directly
          - Intermediate nodes: get the question + previous nodes' responses
          - Sink nodes: produce the final answer
        """
        query = sample["query"]
        dag, topo_order, active_nodes = self._get_decoded_dag()
        n = dag.shape[0]
        intermediate_outputs = [None] * n

        for node in topo_order:
            if node not in active_nodes:
                continue

            in_degree = int(np.sum(dag[:, node]))
            out_degree = int(np.sum(dag[node, :]))

            if in_degree == 0:
                # Source node
                prompt = f"{self.first_instruction}\nQuestion: {query}"
            else:
                # Collect predecessor responses
                predecessors = [j for j in range(n) if dag[j, node] == 1]
                prev_text = ""
                for k, pred in enumerate(predecessors):
                    if intermediate_outputs[pred] is not None:
                        prev_text += f"Previous response {k + 1}: {intermediate_outputs[pred]}\n"

                if out_degree == 0:
                    # Sink node
                    prompt = f"{self.last_instruction}\n{prev_text}Question: {query}"
                else:
                    # Intermediate node
                    prompt = f"{self.non_last_instruction}\n{prev_text}Question: {query}"

            response = self.call_llm(prompt=prompt)
            intermediate_outputs[node] = response.replace("\n", " ") if isinstance(response, str) else response

        # Return the output of the last node in topological order
        final_output = intermediate_outputs[topo_order[-1]]
        if final_output is None:
            # Fallback: find any node with output
            for node in reversed(topo_order):
                if intermediate_outputs[node] is not None:
                    final_output = intermediate_outputs[node]
                    break
        if final_output is None:
            final_output = ""

        return {"response": final_output}

    def _get_decoded_dag(self):
        """
        Get the decoded DAG (discrete adjacency + topo order + active nodes).
        Caches the result to avoid redundant decoding across samples.
        """
        if self._cached_dag is not None:
            return self._cached_dag

        adj = self._get_adjacency_matrix()

        if self.use_discrete_matrix:
            dag = adj.astype(int)
        else:
            dag = graph_decode(adj, top_p_threshold=0)

        topo_order = topological_sort(dag)
        active_nodes = get_active_nodes(dag)
        self._cached_dag = (dag, topo_order, active_nodes)
        return self._cached_dag

    def _get_adjacency_matrix(self):
        """Return the current adjacency matrix, generating one if needed."""
        if self.adjacency_matrix is not None:
            return self.adjacency_matrix
        # Generate a random adjacency matrix
        adj = np.random.rand(self.num_agents, self.num_agents)
        np.fill_diagonal(adj, 0)
        self._set_adjacency_matrix(adj)
        return adj

    def _set_adjacency_matrix(self, adj):
        """Set adjacency matrix and invalidate DAG cache."""
        self.adjacency_matrix = adj
        self._cached_dag = None

    # ----------------------------------------------------------------
    # Optimization: PSO search for optimal topology
    # ----------------------------------------------------------------

    def optimizing(self, val_data):
        """
        Run PSO topology optimization using validation data.

        Optimizes the adjacency matrix (graph topology) to maximize
        accuracy on the validation set.

        Args:
            val_data: list of dicts with 'query' and 'gt' keys
        """
        N = self.num_agents
        G = self.graph_num

        # Initialize graph particles
        particles = []
        for _ in range(G):
            adj = np.random.rand(N, N)
            np.fill_diagonal(adj, 0)
            particles.append({
                "now": adj.copy(),
                "velocity": np.random.randn(N, N) * 0.1,
                "personal_best": adj.copy(),
                "personal_best_score": -1.0,
            })

        global_best = None
        global_best_score = -1.0
        global_worst = None
        global_worst_score = float('inf')
        g_history = []

        logger.info(f"Starting PSO topology optimization: {G} particles, {N} agents, max {self.max_iteration} iters")

        # Evaluate initial particles
        for i, p in enumerate(particles):
            score = self._evaluate_topology(p["now"], val_data)
            p["personal_best_score"] = score
            if score > global_best_score:
                global_best_score = score
                global_best = p["now"].copy()
            if score < global_worst_score:
                global_worst_score = score
                global_worst = p["now"].copy()

        g_history.append(global_best_score)
        logger.info(f"Init: global best = {global_best_score:.4f}")

        step_length = self.step_length

        # Main PSO loop
        for iteration in range(self.max_iteration):
            # Check patience
            if len(g_history) > self.patience:
                recent = g_history[-self.patience:]
                if max(recent) == min(recent):
                    logger.info(f"Patience reached at iteration {iteration}")
                    break

            # Update each particle
            for p in particles:
                self._graph_pso_update(p, global_best, global_worst, step_length)

            # Evaluate
            iter_best_score = -1.0
            iter_worst_score = float('inf')
            iter_best_adj = None
            iter_worst_adj = None

            for p in particles:
                score = self._evaluate_topology(p["now"], val_data)
                if score > p["personal_best_score"]:
                    p["personal_best_score"] = score
                    p["personal_best"] = p["now"].copy()
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_adj = p["now"].copy()
                if score < iter_worst_score:
                    iter_worst_score = score
                    iter_worst_adj = p["now"].copy()

            if iter_best_score > global_best_score:
                global_best_score = iter_best_score
                global_best = iter_best_adj
            if iter_worst_score < global_worst_score:
                global_worst_score = iter_worst_score
                global_worst = iter_worst_adj

            g_history.append(global_best_score)

            # Step length decay
            step_length = max(step_length * self.step_length_factor, self.minimum_step_length)

            if (iteration + 1) % 5 == 0:
                logger.info(f"Iter {iteration + 1}: best = {global_best_score:.4f}, step = {step_length:.4f}")

        # Save the best topology
        self._set_adjacency_matrix(global_best)
        logger.info(f"PSO done: best score = {global_best_score:.4f}")

    def _graph_pso_update(self, particle, global_best, global_worst, step_length):
        """Update a single graph particle using PSO velocity equation."""
        if self.weight_randomness:
            r_w = random.uniform(0, 1)
            r_p = random.uniform(0, 1)
            r_s = random.uniform(0, 1)
            r_b = random.uniform(0, 1)
        else:
            r_w = r_p = r_s = r_b = 1.0

        w_self = r_w * self.inertia
        w_cog = r_p * self.cognitive_coeff
        w_soc = r_s * self.social_coeff
        w_rep = r_b * self.repel_coeff if global_worst is not None else 0.0
        w_sum = w_self + w_cog + w_soc + w_rep
        if w_sum == 0:
            w_sum = 1.0

        w_self /= w_sum
        w_cog /= w_sum
        w_soc /= w_sum
        w_rep /= w_sum

        now = particle["now"]
        vel = particle["velocity"]
        pbest = particle["personal_best"]

        p_x = pbest - now                    # cognitive
        g_x = global_best - now              # social
        x_w = now - global_worst if global_worst is not None else np.zeros_like(now)  # repulsion

        new_vel = w_self * vel + w_cog * p_x + w_soc * g_x + w_rep * x_w
        new_pos = now + step_length * new_vel

        particle["velocity"] = new_vel
        particle["now"] = new_pos

    def _evaluate_topology(self, adjacency_matrix, val_data, max_samples=None):
        """
        Evaluate a topology on validation data by running graph inference
        and computing accuracy (substring match with ground truth).

        Decodes the graph once and reuses for all samples.
        """
        if max_samples and len(val_data) > max_samples:
            eval_data = random.sample(val_data, max_samples)
        else:
            eval_data = val_data

        # Decode once for this topology
        dag = graph_decode(adjacency_matrix, top_p_threshold=0)
        topo_order = topological_sort(dag)
        active_nodes = get_active_nodes(dag)

        # Temporarily set the cached DAG
        old_cache = self._cached_dag
        old_adj = self.adjacency_matrix
        self.adjacency_matrix = adjacency_matrix
        self._cached_dag = (dag, topo_order, active_nodes)

        correct = 0
        total = 0

        for sample in eval_data:
            try:
                result = self.inference(sample)
                response = result.get("response", "")
                gt = sample.get("gt", "")
                if gt and gt in response:
                    correct += 1
                total += 1
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")
                total += 1

        # Restore
        self.adjacency_matrix = old_adj
        self._cached_dag = old_cache
        return correct / total if total > 0 else 0.0
