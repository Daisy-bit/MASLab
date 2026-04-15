"""
HSwarm_MultiObj_Main: NSGA-II multi-objective topology optimization + inference.

Integrates h_swarm's multi-objective topology search into MASLab.
Optimizes two objectives simultaneously:
  - f1: Maximize utility (accuracy) -> minimize -accuracy
  - f2: Minimize sparsity (edge count + connected nodes)

Uses dual-encoding (d, pi) with NSGA-II evolutionary operators.
"""

import os
import random
import logging
from typing import Optional

import numpy as np

from methods.mas_base import MAS
from methods.utils import load_config
from .graph_utils import topological_sort, get_active_nodes
from .graph_utils_multiobj import decode_particle_to_dag, get_edge_count
from .individual import HybridIndividual
from .nsga2_utils import NSGA2HybridUtils
from .pareto_archive import ParetoArchive
from .dual_archive import ConvergenceArchive, DiversityArchive

logger = logging.getLogger(__name__)


# Default prompt templates
DEFAULT_FIRST_INSTRUCTION = "Please answer the following question."
DEFAULT_NON_LAST_INSTRUCTION = (
    "Please answer the following question with the help of previous responses, "
    "feel free to ignore wrong or unhelpful responses."
)
DEFAULT_LAST_INSTRUCTION = (
    "Please answer the following question with the help of previous responses, "
    "feel free to ignore wrong or unhelpful responses."
)


class HSwarm_MultiObj_Main(MAS):
    """
    NSGA-II multi-objective topology optimization for multi-agent inference.

    Dual objectives:
      - f_utility: negative accuracy (minimize -> maximize accuracy)
      - f_sparsity: edge count + connected nodes (minimize -> simpler topology)

    Usage:
      - With pre-optimized topology: set adjacency_matrix_path in config
      - With NSGA-II optimization: call optimizing(val_data) first
      - Pareto selection strategy: "utility" picks best accuracy, "balanced" picks knee point
    """

    def __init__(self, general_config, method_config_name: Optional[str] = None):
        method_config_name = "config_multiobj" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        self.method_config = load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml")
        )

        # Topology parameters
        self.num_agents = int(self.method_config.get("num_agents", 5))
        N = self.num_agents
        self.D = N * (N - 1) // 2  # dimension of d-vector

        # NSGA-II parameters
        self.num_of_individuals = int(self.method_config.get("num_of_individuals", 50))
        self.num_of_generations = int(self.method_config.get("num_of_generations", 100))
        self.tournament_prob = float(self.method_config.get("tournament_prob", 0.9))
        self.crossover_prob = float(self.method_config.get("crossover_prob", 0.9))
        mutation_prob = self.method_config.get("mutation_prob", None)
        self.mutation_prob = float(mutation_prob) if mutation_prob is not None else None

        # Archive parameters
        self.ca_max_size = int(self.method_config.get("ca_max_size", 20))
        self.da_max_size = int(self.method_config.get("da_max_size", 20))

        # Selection strategy
        self.pareto_selection = self.method_config.get("pareto_selection", "utility")

        # Prompt templates
        self.first_instruction = self.method_config.get("first_instruction", DEFAULT_FIRST_INSTRUCTION)
        self.non_last_instruction = self.method_config.get("non_last_instruction", DEFAULT_NON_LAST_INSTRUCTION)
        self.last_instruction = self.method_config.get("last_instruction", DEFAULT_LAST_INSTRUCTION)

        # Load pre-optimized topology
        self.dag = None
        adj_path = self.method_config.get("adjacency_matrix_path", None)
        r_path = self.method_config.get("r_vector_path", None)
        if adj_path and os.path.exists(adj_path):
            data = np.load(adj_path)
            if data.ndim == 2:
                # Already a discrete DAG matrix
                self.dag = data.astype(int)
                self.num_agents = data.shape[0]
            else:
                # It's a d-vector; decode with permutation
                r = np.load(r_path) if r_path and os.path.exists(r_path) else np.arange(self.num_agents)
                self.dag, _ = decode_particle_to_dag(data, r)

        # Store Pareto front after optimization
        self.pareto_front = None

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    def inference(self, sample):
        """Graph-based multi-agent inference using the (possibly optimized) DAG."""
        query = sample["query"]
        dag = self._get_dag()
        topo_order = topological_sort(dag)
        active_nodes = get_active_nodes(dag)
        n = dag.shape[0]
        outputs = [None] * n

        for node in topo_order:
            if node not in active_nodes:
                continue
            in_deg = int(np.sum(dag[:, node]))
            out_deg = int(np.sum(dag[node, :]))

            if in_deg == 0:
                prompt = f"{self.first_instruction}\nQuestion: {query}"
            else:
                preds = [j for j in range(n) if dag[j, node] == 1]
                prev_text = ""
                for k, pred in enumerate(preds):
                    if outputs[pred] is not None:
                        prev_text += f"Previous response {k + 1}: {outputs[pred]}\n"
                instruction = self.last_instruction if out_deg == 0 else self.non_last_instruction
                prompt = f"{instruction}\n{prev_text}Question: {query}"

            response = self.call_llm(prompt=prompt)
            outputs[node] = response.replace("\n", " ") if isinstance(response, str) else response

        final = outputs[topo_order[-1]]
        if final is None:
            for node in reversed(topo_order):
                if outputs[node] is not None:
                    final = outputs[node]
                    break
        return {"response": final or ""}

    def _get_dag(self):
        """Get the current DAG, generating a random one if needed."""
        if self.dag is not None:
            return self.dag
        N = self.num_agents
        d = np.random.randint(0, 2, size=self.D)
        r = np.arange(N)
        np.random.shuffle(r)
        dag, _ = decode_particle_to_dag(d, r)
        self.dag = dag
        return dag

    # ----------------------------------------------------------------
    # Multi-Objective Optimization: NSGA-II
    # ----------------------------------------------------------------

    def optimizing(self, val_data):
        """
        Run NSGA-II multi-objective topology optimization.

        Optimizes:
          - f1: -accuracy (minimize -> maximize accuracy)
          - f2: edge_count + connected_nodes (minimize -> simpler topology)
        """
        N = self.num_agents
        D = self.D
        pop_size = self.num_of_individuals

        nsga2 = NSGA2HybridUtils(
            num_of_individuals=pop_size,
            tournament_prob=self.tournament_prob,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
        )
        ca = ConvergenceArchive(max_size=self.ca_max_size)
        da = DiversityArchive(max_size=self.da_max_size)
        pareto = ParetoArchive()

        # Initialize population
        population = []
        for _ in range(pop_size):
            ind = HybridIndividual()
            ind.d = np.random.randint(0, 2, size=D)
            ind.r = np.arange(N)
            np.random.shuffle(ind.r)
            population.append(ind)

        # Evaluate initial population
        logger.info(f"NSGA-II init: {pop_size} individuals, {N} agents, D={D}")
        self._evaluate_population(population, val_data)

        # Update archives
        solutions = [(ind.d.copy(), ind.r.copy(), ind.objectives.copy()) for ind in population]
        ca.update(solutions)
        da.update(solutions)
        pareto.update_archive(solutions)

        # Sort initial population
        nsga2.fast_nondominated_sort(population)
        for front_idx in range(len(population)):
            front = [ind for ind in population if ind.rank == front_idx]
            if front:
                nsga2.calculate_crowding_distance(front)

        logger.info(f"Init Pareto front size: {pareto.size()}")

        # Main NSGA-II loop
        for gen in range(self.num_of_generations):
            # Create offspring
            children_d = nsga2.create_children_d(population)
            children_pi = nsga2.create_children_pi(population)

            # Build child population
            children = []
            for cd, cr in zip(children_d, children_pi):
                child = HybridIndividual()
                child.d = cd
                child.r = cr
                children.append(child)

            # Evaluate children
            self._evaluate_population(children, val_data)

            # Environmental selection
            population = nsga2.environmental_selection(population, children, pop_size)

            # Update archives
            new_solutions = [(ind.d.copy(), ind.r.copy(), ind.objectives.copy()) for ind in children]
            ca.update(new_solutions)
            da.update(new_solutions)
            pareto.update_archive(new_solutions)

            if (gen + 1) % 10 == 0:
                front_objs = pareto.get_pareto_front_objectives()
                best_util = min(obj[0] for obj in front_objs) if front_objs else float('inf')
                logger.info(
                    f"Gen {gen + 1}: Pareto size={pareto.size()}, "
                    f"best utility={-best_util:.4f}"
                )

        # Store Pareto front
        self.pareto_front = pareto.get_pareto_front()

        # Select the topology for inference
        self._select_topology_from_pareto()

        logger.info(f"NSGA-II done: Pareto front size = {pareto.size()}")

    def _evaluate_population(self, population, val_data):
        """Evaluate all individuals in the population."""
        for ind in population:
            objectives = self._evaluate_individual(ind, val_data)
            ind.objectives = objectives

    def _evaluate_individual(self, ind, val_data):
        """
        Evaluate a single individual on validation data.

        Decodes the DAG once and reuses for all samples.

        Returns: np.array([-accuracy, sparsity])
        """
        A_dag, _ = decode_particle_to_dag(ind.d, ind.r)

        # Compute sparsity (edge count + connected nodes)
        N_total = A_dag.shape[0]
        out_deg = np.sum(A_dag, axis=1)
        in_deg = np.sum(A_dag, axis=0)
        disconnected = (out_deg == 0) & (in_deg == 0)
        n_connected = N_total - np.sum(disconnected)
        f_sparsity = get_edge_count(A_dag) + n_connected

        # Check validity: no edges means no valid inference path
        if np.sum(A_dag) == 0:
            return np.array([0.0, float(f_sparsity)])

        # Temporarily set the DAG for inference
        old_dag = self.dag
        self.dag = A_dag

        correct = 0
        total = 0
        for sample in val_data:
            try:
                result = self.inference(sample)
                response = result.get("response", "")
                gt = sample.get("gt", "")
                if gt and gt in response:
                    correct += 1
                total += 1
            except Exception:
                total += 1

        self.dag = old_dag
        accuracy = correct / total if total > 0 else 0.0
        return np.array([-accuracy, float(f_sparsity)])

    def _select_topology_from_pareto(self):
        """Select a topology from the Pareto front for inference."""
        if not self.pareto_front:
            return

        if self.pareto_selection == "utility":
            # Pick the solution with best utility (lowest -accuracy = highest accuracy)
            best = min(self.pareto_front, key=lambda sol: sol[2][0])
        elif self.pareto_selection == "balanced":
            # Pick knee point: closest to the utopia point in normalized space
            objs = np.array([sol[2] for sol in self.pareto_front])
            mins = objs.min(axis=0)
            maxs = objs.max(axis=0)
            rng = maxs - mins
            rng[rng == 0] = 1
            norm = (objs - mins) / rng
            dists = np.linalg.norm(norm, axis=1)
            best = self.pareto_front[int(np.argmin(dists))]
        else:
            best = self.pareto_front[0]

        d_best, r_best, obj_best = best
        self.dag, _ = decode_particle_to_dag(d_best, r_best)
        logger.info(f"Selected topology: accuracy={-obj_best[0]:.4f}, sparsity={obj_best[1]:.0f}")

    def get_pareto_front(self):
        """Return the Pareto front after optimization (for analysis)."""
        return self.pareto_front
