"""SCC components: baseline-agnostic shared functions for the three SCC modules.

The reference implementation lives in `methods/soo_centered_v3`; this package
extracts its component logic so any baseline (mad_vote, selforg, dylan, ...)
can mix in the same SCC behaviour without re-implementing it. Functions here
are pure (no class state, no LLM calls, no embedding-model load) — caller
holds the embedding model and RNG.

Public re-exports:
  task_typing.detect_task_type
  voting.extract_answer
  voting.mcq_is_equiv
  voting.count_first_plurality
  voting.format_final
  spectral.double_center
  spectral.pairwise_cosine
  spectral.pc1_contributions
  spectral.is_spectral_consensus
  routing.build_diverse_graph
  routing.dagify
  routing.topo_order_by_contributions
"""

from .task_typing import detect_task_type
from .voting import (
    bleu_cluster_groups,
    code_is_equiv,
    count_first_plurality,
    extract_answer,
    format_final,
    mcq_is_equiv,
)

# spectral / routing are added in a follow-up step; keep imports guarded so
# this package is usable as soon as voting + task_typing are written.
try:
    from .spectral import (  # noqa: F401
        double_center,
        is_spectral_consensus,
        pairwise_cosine,
        pc1_contributions,
    )
    from .routing import (  # noqa: F401
        build_diverse_graph,
        dagify,
        topo_order_by_contributions,
    )
except ImportError:
    pass

__all__ = [
    "detect_task_type",
    "bleu_cluster_groups",
    "code_is_equiv",
    "count_first_plurality",
    "extract_answer",
    "format_final",
    "mcq_is_equiv",
]
