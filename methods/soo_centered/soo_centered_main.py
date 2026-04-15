"""
SOO-Centered: Double-Centered Spectral Self-Organized Orchestration.

Replaces exp(S/tau) consensus matrix with double-centered cosine similarity
S_c = HSH (Kernel PCA), fixing the spectral gap collapse problem where all
LLM embeddings have high baseline cosine similarity (~0.85-0.95), causing
the original SOO's spectral gap ratio to saturate near 1.0 regardless of
actual agreement structure.

Difference from SOO:
  Consensus matrix:  S_c = HSH  instead of  A = exp(S / tau)
  Contribution:      PC1 eigenvector of S_c  (baseline-removed centrality)

Everything else (DAG construction, debate, aggregation, consensus check)
is inherited unchanged from SelfOrg_Main.
"""

from __future__ import annotations

import math

import numpy as np

from methods.selforg import SelfOrg_Main


class SOO_Centered_Main(SelfOrg_Main):
    """
    Self-Organized Orchestration with double-centered spectral analysis.

    Inherits the full SelfOrg pipeline, overriding only _approx_shapley:
    PC1 of S_c = HSH instead of Perron(exp(S/tau)).
    """

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

    # ── helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _double_center(S: np.ndarray) -> np.ndarray:
        """S_c = H S H  where  H = I - (1/N) 1 1^T."""
        row_mean = S.mean(axis=1, keepdims=True)
        col_mean = S.mean(axis=0, keepdims=True)
        total_mean = S.mean()
        return S - row_mean - col_mean + total_mean

    # ── core override: contribution estimation ──────────────────────────

    def _approx_shapley(self, answers, reference):
        """
        Contribution estimation via double-centered spectral analysis.

        Steps
        -----
        1. Embed all answers  ->  e_1 ... e_N  (L2-normalised)
        2. S_ij = cos(e_i, e_j)
        3. S_c = H S H             (double centering = Kernel PCA)
        4. Eigendecompose S_c       (ascending via eigh)
        5. PC1 eigenvector as raw contribution scores
           |v1[i]| captures how strongly agent i participates in the
           dominant consensus direction after the shared baseline is removed.
        6. Softmax(T=0.1) normalisation  (same convention as SOO / SelfOrg)
        """
        embs = self._embed_many(answers)
        n = len(embs)
        if n == 0:
            return []

        # Build cosine similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = self._cosine(embs[i], embs[j])

        # Double centering
        S_c = self._double_center(S)

        # Eigendecomposition (eigh -> ascending order)
        eigenvalues, eigenvectors = np.linalg.eigh(S_c)

        # Contribution: absolute PC1 component (largest eigenvalue = last)
        raw = np.abs(eigenvectors[:, -1]).tolist()

        # Softmax normalisation (T = 0.1, matching SelfOrg / SOO convention)
        s = sum(raw)
        if s <= 1e-12:
            return [1.0 / n] * n

        t = max(1e-6, 1.0 / 10.0)
        mx = max(raw)
        ex = [math.exp((r - mx) / t) for r in raw]
        s = sum(ex) or 1.0
        return [e / s for e in ex]
