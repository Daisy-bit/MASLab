"""
SOO-Centered-v2: Double-Centered Spectral SOO with Variance-Based Consensus.

Builds on SOO-Centered (method A) by adding a spectral consensus detector:
  tr(S_c) = total variance of mean-centered embeddings.
When tr(S_c) is small, agents genuinely agree and debate can be skipped.

Differences from SOO:
  1. Consensus matrix:  S_c = HSH  instead of  A = exp(S / tau)
  2. Contribution:      PC1 eigenvector of S_c  (baseline-removed centrality)
  3. Consensus detect:  tr(S_c) < threshold  as primary early-stop signal
"""

from __future__ import annotations

import math

import numpy as np

from methods.selforg import SelfOrg_Main


class SOO_Centered_v2_Main(SelfOrg_Main):
    """
    Self-Organized Orchestration with double-centered spectral analysis
    and variance-based consensus detection.

    Inherits the full SelfOrg pipeline, overriding:
      - _approx_shapley:       PC1 of S_c = HSH  (same as SOO-Centered)
      - _check_for_consensus:  tr(S_c) total-variance early-stop
    """

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        mc = self.method_config
        # Threshold on tr(S_c): below this the agents are in genuine consensus.
        self.variance_consensus_thr = float(mc.get("variance_consensus_thr", 0.05))

        # Spectral diagnostics produced by the last _approx_shapley call.
        self._last_spectral: dict | None = None

    # ── helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _double_center(S: np.ndarray) -> np.ndarray:
        """S_c = H S H  where  H = I - (1/N) 1 1^T."""
        row_mean = S.mean(axis=1, keepdims=True)
        col_mean = S.mean(axis=0, keepdims=True)
        total_mean = S.mean()
        return S - row_mean - col_mean + total_mean

    # ── override 1: contribution estimation ─────────────────────────────

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
        7. Store tr(S_c) for consensus detection.
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

        # Store spectral diagnostics for consensus check
        lam1 = float(max(eigenvalues[-1], 1e-12))
        lam2 = float(eigenvalues[-2]) if n >= 2 else 0.0
        trace_Sc = float(np.trace(S_c))
        gap_ratio = (lam1 - lam2) / lam1

        self._last_spectral = {
            "trace": trace_Sc,
            "gap_ratio": gap_ratio,
            "lam1": lam1,
            "lam2": lam2,
        }

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

    # ── override 2: consensus check ─────────────────────────────────────

    def _check_for_consensus(self, sims):
        """
        Primary signal: tr(S_c) < threshold  ->  genuine consensus.
        Fallback: parent's pairwise-similarity check.

        tr(S_c) = sum of eigenvalues of the centered Gram matrix
                = total variance of mean-centred embeddings.
        When this is small, all agents are close to their centroid,
        meaning genuine agreement (not just high baseline cosine similarity).
        """
        if self._last_spectral is not None:
            if self._last_spectral["trace"] < self.variance_consensus_thr:
                return True
        return super()._check_for_consensus(sims)
