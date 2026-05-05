"""Parity tests: scc_components functions must match SOO_Centered_v3_Main
methods on the same inputs.

This file is a regression gate. If any scc_components function drifts away
from the v3 reference, these tests should catch it before downstream methods
(soo_scc, mad_scc) silently change behaviour.

Run:
  pytest tests/test_scc_components_parity.py -v

Notes:
- Tests that need a live SOO_Centered_v3_Main instance call __init__ with a
  minimal general_config (no LLM endpoint required for routing/voting/
  spectral parity — those code paths never call call_llm).
- The embedding model is a process-wide singleton (model/all-MiniLM-L6-v2);
  if it isn't downloaded, spectral/routing tests fall back to manual seed
  vectors so the parity check stays usable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.scc_components import voting as scc_voting
from methods.scc_components import task_typing as scc_task_typing
from methods.scc_components import spectral as scc_spectral
from methods.scc_components import routing as scc_routing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v3_instance():
    """Build a SOO_Centered_v3_Main instance suitable for static-method
    parity (no LLM endpoint needed)."""
    from methods.soo_centered_v3 import SOO_Centered_v3_Main
    general_config = {
        "model_name": "qwen25-1.5b-instruct",
        "model_temperature": 0.5,
        "model_max_tokens": 512,
        "model_timeout": 60,
        "test_dataset_name": "GSM8K",
        # No model_api_config — we never call _call_llm in parity tests.
        "model_api_config": {
            "qwen25-1.5b-instruct": {
                "model_list": [{"model_name": "x", "model_url": "x", "api_key": "x"}],
                "max_workers_per_model": 1,
            }
        },
    }
    return SOO_Centered_v3_Main(general_config)


# ---------------------------------------------------------------------------
# Voting / extraction parity
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    def test_math_boxed(self):
        v3 = _v3_instance()
        reply = "Working it out... \\boxed{32} feet."
        assert (
            scc_voting.extract_answer(reply, "math")
            == v3._extract_answer(reply, "math")
        )

    def test_math_the_answer_is(self):
        v3 = _v3_instance()
        reply = "We compute 24 * 3 = 72, then 72 * 2/3 = 48, then 48 * 2/3 = 32. The answer is 32."
        assert (
            scc_voting.extract_answer(reply, "math")
            == v3._extract_answer(reply, "math")
        )

    def test_mcq_anchored(self):
        v3 = _v3_instance()
        reply = "After analysis, the correct option is. The answer is (B)."
        assert (
            scc_voting.extract_answer(reply, "mcq")
            == v3._extract_answer(reply, "mcq")
            == "B"
        )

    def test_mcq_paren_only(self):
        v3 = _v3_instance()
        reply = "I considered (A), (B), and (C). The right one is (C)."
        assert (
            scc_voting.extract_answer(reply, "mcq")
            == v3._extract_answer(reply, "mcq")
            == "C"
        )

    def test_mcq_line_start_fallback(self):
        v3 = _v3_instance()
        reply = "Options:\nA) wrong\nB) wrong\nD) right"
        assert (
            scc_voting.extract_answer(reply, "mcq")
            == v3._extract_answer(reply, "mcq")
        )

    def test_open_returns_empty(self):
        v3 = _v3_instance()
        reply = "The capital of France is Paris."
        assert (
            scc_voting.extract_answer(reply, "open")
            == v3._extract_answer(reply, "open")
            == ""
        )


class TestMcqIsEquiv:
    def test_basic_match(self):
        v3 = _v3_instance()
        for a, b in [("A", "a"), (" B ", "b"), ("C", "C"), ("d", "D")]:
            assert (
                scc_voting.mcq_is_equiv(a, b)
                == v3._mcq_is_equiv(a, b)
            )

    def test_basic_mismatch(self):
        v3 = _v3_instance()
        for a, b in [("A", "B"), ("", "A"), ("A", ""), ("AB", "A")]:
            assert (
                scc_voting.mcq_is_equiv(a, b)
                == v3._mcq_is_equiv(a, b)
            )


class TestCountFirstPlurality:
    """Each test compares scc_voting.count_first_plurality against v3._plurality
    on the SAME (answers, contributions) input. Toggles the
    enable_contribution_aggregation flag on both."""

    def _check(
        self,
        answers,
        contributions,
        task_type,
        enable_contribution_aggregation,
    ):
        v3 = _v3_instance()
        v3.enable_contribution_aggregation = enable_contribution_aggregation
        ours = scc_voting.count_first_plurality(
            answers,
            contributions,
            task_type,
            enable_contribution_aggregation=enable_contribution_aggregation,
        )
        ref = v3._plurality(answers, contributions, task_type)
        assert ours == ref, f"mismatch: ours={ours} ref={ref}"

    def test_math_3_vs_2_majority(self):
        answers = [
            "Working... \\boxed{32}",
            "Working... \\boxed{32}",
            "Working... \\boxed{32}",
            "Hmm \\boxed{48}",
            "Different \\boxed{48}",
        ]
        contributions = [0.25, 0.25, 0.05, 0.25, 0.20]
        for flag in (True, False):
            self._check(answers, contributions, "math", flag)

    def test_math_count_overrides_contribution(self):
        # Group A (3 agents) has lower per-agent contribution but bigger
        # size; v3 contract is count-first → A wins regardless of
        # contribution sums.
        answers = [
            "\\boxed{1}",
            "\\boxed{1}",
            "\\boxed{1}",
            "\\boxed{2}",
            "\\boxed{2}",
        ]
        contributions = [0.05, 0.05, 0.05, 0.40, 0.40]
        for flag in (True, False):
            self._check(answers, contributions, "math", flag)

    def test_mcq_letter_grouping(self):
        answers = [
            "Reasoning... The answer is (A).",
            "Reasoning... (A) is correct",
            "I think (b)",
            "Final answer: B",
            "The answer is (C).",
        ]
        contributions = [0.20, 0.20, 0.20, 0.20, 0.20]
        for flag in (True, False):
            self._check(answers, contributions, "mcq", flag)

    def test_open_returns_empty(self):
        answers = ["foo", "bar", "baz"]
        for flag in (True, False):
            self._check(answers, [0.33, 0.33, 0.34], "open", flag)

    def test_no_extraction_returns_empty(self):
        # All math replies fail extraction → should return ("", 0).
        answers = ["No box here", "Just prose", "Nothing", "", ""]
        for flag in (True, False):
            self._check(answers, [0.2] * 5, "math", flag)


# ---------------------------------------------------------------------------
# format_final parity
# ---------------------------------------------------------------------------

class TestFormatFinal:
    def test_math(self):
        v3 = _v3_instance()
        assert (
            scc_voting.format_final("32", "math")
            == v3._format_final("32", "math")
        )

    def test_mcq(self):
        v3 = _v3_instance()
        assert (
            scc_voting.format_final("a", "mcq")
            == v3._format_final("a", "mcq")
            == "The answer is (A)"
        )

    def test_empty_canonical(self):
        v3 = _v3_instance()
        assert (
            scc_voting.format_final("", "math")
            == v3._format_final("", "math")
            == ""
        )

    def test_open_passthrough(self):
        v3 = _v3_instance()
        assert (
            scc_voting.format_final("Paris", "open")
            == v3._format_final("Paris", "open")
        )


# ---------------------------------------------------------------------------
# task_typing.detect_task_type parity
# ---------------------------------------------------------------------------

class TestDetectTaskType:
    @pytest.mark.parametrize(
        "sample",
        [
            {"source": "GSM8K", "query": "How many feet?"},
            {"source": "MMLU-Pro", "query": "Pick one."},
            {"source": "AIME-2024", "query": "Find n."},
            {"source": "AQUA-RAT", "query": "(A) 1 (B) 2 (C) 3 (D) 4 (E) 5"},
            {"query": "(A) yes (B) no (C) maybe (D) skip"},
            {"query": "Define gravity in 3 sentences."},
        ],
    )
    def test_matches_v3(self, sample):
        v3 = _v3_instance()
        assert (
            scc_task_typing.detect_task_type(sample, force_task_type=None)
            == v3._detect_task_type(sample)
        )

    def test_force_override(self):
        sample = {"source": "GSM8K", "query": "trivial"}
        for forced in ("math", "mcq", "open"):
            assert (
                scc_task_typing.detect_task_type(
                    sample, force_task_type=forced
                )
                == forced
            )


# ---------------------------------------------------------------------------
# Spectral parity
# ---------------------------------------------------------------------------

def _seeded_normalised_embs(n: int, d: int, seed: int) -> "list[np.ndarray]":
    """Generate a fixed set of L2-normalised vectors of dimension d."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(n, d)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return [row.tolist() for row in (raw / norms)]


class TestSpectral:
    """Compare scc_components.spectral against v2's _approx_shapley directly.

    v3 inherits _approx_shapley from v2 unchanged. We bypass the embed step
    by building L2-normalised vectors and feeding them to v2's internal
    helpers (S_c double-center + eigh + softmax)."""

    def _v3_pc1(self, embs):
        """Reproduce v2._approx_shapley's algorithm given pre-computed
        embeddings, mirroring soo_centered_v2_main.py:56-114."""
        v3 = _v3_instance()
        n = len(embs)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = v3._cosine(embs[i], embs[j])
        S_c = v3._double_center(S)
        eigenvalues, eigenvectors = np.linalg.eigh(S_c)
        lam1 = float(max(eigenvalues[-1], 1e-12))
        lam2 = float(eigenvalues[-2]) if n >= 2 else 0.0
        spec = {
            "trace": float(np.trace(S_c)),
            "gap_ratio": (lam1 - lam2) / lam1,
            "lam1": lam1,
            "lam2": lam2,
        }
        raw = np.abs(eigenvectors[:, -1]).tolist()
        s = sum(raw)
        if s <= 1e-12:
            return [1.0 / n] * n, spec
        import math as _math
        t = max(1e-6, 1.0 / 10.0)
        mx = max(raw)
        ex = [_math.exp((r - mx) / t) for r in raw]
        sm = sum(ex) or 1.0
        return [e / sm for e in ex], spec

    def test_random_5_agents(self):
        embs = _seeded_normalised_embs(5, 32, seed=17)
        ours, ours_spec = scc_spectral.pc1_contributions(embs)
        ref, ref_spec = self._v3_pc1(embs)
        np.testing.assert_allclose(ours, ref, atol=1e-9)
        for k in ("trace", "gap_ratio", "lam1", "lam2"):
            assert abs(ours_spec[k] - ref_spec[k]) < 1e-9, k

    def test_degenerate_all_same(self):
        embs = [np.array([1.0, 0.0, 0.0, 0.0])] * 5
        ours, ours_spec = scc_spectral.pc1_contributions(embs)
        ref, ref_spec = self._v3_pc1(embs)
        np.testing.assert_allclose(ours, ref, atol=1e-9)
        # trace of zero matrix → 0
        assert abs(ours_spec["trace"]) < 1e-12

    def test_is_spectral_consensus(self):
        # Construct a near-identical pool: trace ≈ 0 → consensus.
        embs = _seeded_normalised_embs(5, 32, seed=42)
        # tweak so they all collapse: take the centroid and use it.
        centroid = np.mean(embs, axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-9)
        embs = [centroid.tolist()] * 5
        _, spec = scc_spectral.pc1_contributions(embs)
        assert scc_spectral.is_spectral_consensus(spec, thr=0.05)

    def test_not_in_consensus(self):
        # Diverse pool: trace large.
        embs = _seeded_normalised_embs(5, 32, seed=99)
        _, spec = scc_spectral.pc1_contributions(embs)
        assert not scc_spectral.is_spectral_consensus(spec, thr=0.05)


# ---------------------------------------------------------------------------
# Routing parity
# ---------------------------------------------------------------------------

class TestRouting:
    def _v3_routing(self, sims, contributions, n, *, top_k, sim_threshold,
                    diversity_p, enforce_dag, enable_routing, seed):
        """Run v3's _build_diverse_graph by instantiating a v3 with matching
        config. Uses its own _diversity_rng so we have to control the seed
        via random_seed in mc."""
        v3 = _v3_instance()
        v3.top_k = top_k
        v3.sim_threshold = sim_threshold
        v3.diversity_p = diversity_p
        v3.enforce_dag = enforce_dag
        v3.enable_contribution_routing = enable_routing
        v3._diversity_rng = random.Random(seed)
        return v3._build_diverse_graph(sims, contributions, n)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_no_diversity_matches(self, seed):
        import random as _random
        n = 5
        rng = np.random.RandomState(seed)
        sims = np.eye(n) + 0.5 * rng.rand(n, n)
        sims = (sims + sims.T) / 2
        np.fill_diagonal(sims, 1.0)
        contributions = (rng.rand(n)).tolist()

        ours = scc_routing.build_diverse_graph(
            sims, contributions, n,
            top_k=2, sim_threshold=0.0, diversity_p=0.0,
            enforce_dag=True, enable_routing=True,
            rng=_random.Random(seed),
        )
        ref = self._v3_routing(
            sims, contributions, n,
            top_k=2, sim_threshold=0.0, diversity_p=0.0,
            enforce_dag=True, enable_routing=True, seed=seed,
        )
        # Edges: exact equality
        assert ours[0] == ref[0]
        # Edge weights: numeric equality within atol
        assert set(ours[1].keys()) == set(ref[1].keys())
        for e in ours[1]:
            assert abs(ours[1][e] - ref[1][e]) < 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_with_diversity_matches(self, seed):
        import random as _random
        n = 5
        rng = np.random.RandomState(seed)
        sims = 0.5 + 0.4 * rng.rand(n, n)
        sims = (sims + sims.T) / 2
        np.fill_diagonal(sims, 1.0)
        contributions = (rng.rand(n)).tolist()

        ours = scc_routing.build_diverse_graph(
            sims, contributions, n,
            top_k=2, sim_threshold=0.0, diversity_p=0.3,
            enforce_dag=True, enable_routing=True,
            rng=_random.Random(seed),
        )
        ref = self._v3_routing(
            sims, contributions, n,
            top_k=2, sim_threshold=0.0, diversity_p=0.3,
            enforce_dag=True, enable_routing=True, seed=seed,
        )
        assert ours[0] == ref[0]
        for e in ours[1]:
            assert abs(ours[1][e] - ref[1][e]) < 1e-9

    @pytest.mark.parametrize("seed", [0, 1])
    def test_full_mesh_when_routing_disabled(self, seed):
        import random as _random
        n = 5
        sims = np.eye(n)
        contributions = [0.2] * n

        ours = scc_routing.build_diverse_graph(
            sims, contributions, n,
            top_k=2, sim_threshold=0.0, diversity_p=0.0,
            enforce_dag=True, enable_routing=False,
            rng=_random.Random(seed),
        )
        ref = self._v3_routing(
            sims, contributions, n,
            top_k=2, sim_threshold=0.0, diversity_p=0.0,
            enforce_dag=True, enable_routing=False, seed=seed,
        )
        assert ours[0] == ref[0]
        # Full mesh weights should all be 1.0
        for w in ours[1].values():
            assert abs(w - 1.0) < 1e-12

    def test_topo_order_matches(self):
        # Linear DAG 0 → 1 → 2 → 3 → 4
        edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
        contributions = [0.3, 0.25, 0.2, 0.15, 0.1]
        ours = scc_routing.topo_order_by_contributions(edges, contributions, 5)
        v3 = _v3_instance()
        ref = v3._topo_order_by_contributions(edges, contributions)
        assert ours == ref


# ensure `random` is in scope for parametrize lambdas above
import random  # noqa: E402  pylint: disable=wrong-import-position
