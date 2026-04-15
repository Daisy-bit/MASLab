from __future__ import annotations

import math

import numpy as np

from methods.selforg import SelfOrg_Main


class SOO_Main(SelfOrg_Main):
    """
    Self-Organized Orchestration (SOO).

    Inherits the full SelfOrg pipeline from the official implementation,
    overriding only the contribution estimation: Perron-Frobenius eigenvector
    of the consensus matrix instead of Shapley-approximated cosine-to-centroid.
    """

    def __init__(self, general_config, method_config_name=None):
        super().__init__(general_config, method_config_name)
        self.consensus_tau = float(self.method_config.get("consensus_tau", 1.0))

    def _approx_shapley(self, answers, reference):
        """
        Override: use consensus matrix eigenvector for contribution estimation.

        Steps:
          1) Embed all answers
          2) S = cosine similarity matrix (embeddings are L2-normalised)
          3) A_ij = exp(S_ij / tau_consensus)
          4) Principal eigenvector of A as reliability scores
          5) Softmax with temperature T=0.1 to get normalised weights
        """
        embs = self._embed_many(answers)
        n = len(embs)
        if n == 0:
            return []

        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = self._cosine(embs[i], embs[j])

        # Consensus matrix
        A = np.exp(S / self.consensus_tau)

        # Perron-Frobenius: principal eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        raw = np.abs(eigenvectors[:, -1]).tolist()

        # Softmax normalisation (temperature T=0.1, matching SelfOrg convention)
        s = sum(raw)
        if s <= 1e-12:
            return [1.0 / n] * n

        t = max(1e-6, 1.0 / float(10.0))
        mx = max(raw)
        ex = [math.exp((r - mx) / t) for r in raw]
        s = sum(ex) or 1.0
        return [e / s for e in ex]
