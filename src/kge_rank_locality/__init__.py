"""Public API for :mod:`kge_rank_locality`."""

from kge_rank_locality.core import (
    DEFAULT_FRACTIONS,
    DEFAULT_LAMBDAS,
    KGEEmbeddings,
    LocalityResult,
    ProtectionSubspace,
    SweepResult,
    candidate_column_locality,
    direct_promotion_delta,
    promotion_score_gap,
    rank_truncated_delta,
    ridge_preservation_delta,
    run_frontier,
)
from kge_rank_locality.io import ExperimentBundle, load_bundle, save_bundle

__all__ = [
    "DEFAULT_FRACTIONS",
    "DEFAULT_LAMBDAS",
    "ExperimentBundle",
    "KGEEmbeddings",
    "LocalityResult",
    "ProtectionSubspace",
    "SweepResult",
    "candidate_column_locality",
    "direct_promotion_delta",
    "load_bundle",
    "promotion_score_gap",
    "rank_truncated_delta",
    "ridge_preservation_delta",
    "run_frontier",
    "save_bundle",
]

__version__ = "0.1.0"
