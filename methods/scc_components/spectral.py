"""Spectral primitives — PC1 contributions + tr(S_c) consensus signal.

Lifted verbatim from soo_centered_v2_main.py:46-114 with the embedding step
factored out (caller passes already-embedded vectors). All operations are
pure functions over numpy arrays / float lists.

Core math:
  S_ij = cos(e_i, e_j)         pairwise cosine similarity
  H = I_N - (1/N) 1 1^T        centering matrix
  S_c = H S H                  double-centered Gram matrix (Kernel PCA centring)
  trace(S_c)                   = sum of eigenvalues = total variance
                                  → small ↔ genuine semantic consensus
  v_1 = eigvec(S_c, max λ)     leading principal component
  c_i = softmax_T=0.1(|v_1[i]|) per-agent contribution
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


VectorLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Cosine / centering
# ---------------------------------------------------------------------------

def _cosine(a: VectorLike, b: VectorLike) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b))  # caller guarantees L2-normalised


def double_center(S: np.ndarray) -> np.ndarray:
    """Return H S H with H = I - (1/N) 11^T.

    Equivalent to subtracting row-means and column-means then adding the grand
    mean — the standard Kernel PCA centring used by SOO-Centered v2.
    """
    row_mean = S.mean(axis=1, keepdims=True)
    col_mean = S.mean(axis=0, keepdims=True)
    total_mean = S.mean()
    return S - row_mean - col_mean + total_mean


def pairwise_cosine(embs: Sequence[VectorLike]) -> np.ndarray:
    """N×N cosine similarity matrix from L2-normalised embeddings."""
    n = len(embs)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = _cosine(embs[i], embs[j])
    return S


# ---------------------------------------------------------------------------
# PC1 contribution
# ---------------------------------------------------------------------------

def pc1_contributions(
    embs: Sequence[VectorLike],
    *,
    softmax_temperature: float = 0.1,
) -> Tuple[List[float], Dict[str, float]]:
    """Return (contributions, spectral_diag).

    contributions[i] = softmax_T(|v_1[i]|) where v_1 is the eigenvector of
    the largest eigenvalue of the double-centered cosine Gram matrix.

    spectral_diag = {"trace", "gap_ratio", "lam1", "lam2"}, exactly the
    schema written into v2's `self._last_spectral`.

    Empty input returns ([], {"trace":0, "gap_ratio":0, "lam1":1e-12,
    "lam2":0}) so callers don't have to special-case n=0.
    """
    n = len(embs)
    if n == 0:
        return [], {"trace": 0.0, "gap_ratio": 0.0, "lam1": 1e-12, "lam2": 0.0}

    S = pairwise_cosine(embs)
    S_c = double_center(S)

    eigenvalues, eigenvectors = np.linalg.eigh(S_c)  # ascending order

    lam1 = float(max(eigenvalues[-1], 1e-12))
    lam2 = float(eigenvalues[-2]) if n >= 2 else 0.0
    trace_Sc = float(np.trace(S_c))
    gap_ratio = (lam1 - lam2) / lam1

    spec_diag = {
        "trace": trace_Sc,
        "gap_ratio": gap_ratio,
        "lam1": lam1,
        "lam2": lam2,
    }

    raw = np.abs(eigenvectors[:, -1]).tolist()
    s = sum(raw)
    if s <= 1e-12:
        return [1.0 / n] * n, spec_diag

    t = max(1e-6, float(softmax_temperature))
    mx = max(raw)
    ex = [math.exp((r - mx) / t) for r in raw]
    sm = sum(ex) or 1.0
    contributions = [e / sm for e in ex]
    return contributions, spec_diag


# ---------------------------------------------------------------------------
# Consensus check
# ---------------------------------------------------------------------------

def is_spectral_consensus(
    spec_diag: Dict[str, float], thr: float
) -> bool:
    """True iff `trace(S_c) < thr`. Mirrors v3 `_check_for_consensus`'s
    primary signal (tr(S_c) below threshold = genuine consensus)."""
    if not spec_diag:
        return False
    return spec_diag.get("trace", float("inf")) < thr
