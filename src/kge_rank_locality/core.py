"""Numerical core for rank-aware KGE editing.

The implementation is deliberately framework-independent. It operates on NumPy
arrays and supports KGE models whose tail score is linear in the candidate tail
embedding: DistMult and ComplEx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ModelName = Literal["distmult", "complex"]

DEFAULT_FRACTIONS = (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 1.0)
DEFAULT_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_MARGIN = 1e-6
DEFAULT_SVD_TOLERANCE = 1e-6
DEFAULT_NOOP_TOLERANCE = 1e-12


def _as_float64(value: NDArray[np.floating]) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class KGEEmbeddings:
    """A validated, in-memory linear KGE model."""

    entity: FloatArray
    relation: FloatArray
    model: ModelName

    def __post_init__(self) -> None:
        entity = _as_float64(self.entity)
        relation = _as_float64(self.relation)
        if entity.ndim != 2 or relation.ndim != 2:
            raise ValueError("entity and relation embeddings must be rank-2 arrays")
        if not np.isfinite(entity).all() or not np.isfinite(relation).all():
            raise ValueError("embeddings must contain only finite values")
        if self.model not in {"distmult", "complex"}:
            raise ValueError(f"unsupported model: {self.model!r}")
        if entity.shape[1] != relation.shape[1]:
            raise ValueError("entity and relation embedding widths must match")
        if self.model == "complex" and entity.shape[1] % 2:
            raise ValueError("ComplEx embeddings must have an even width")
        object.__setattr__(self, "entity", entity)
        object.__setattr__(self, "relation", relation)

    @property
    def dimension(self) -> int:
        return int(self.entity.shape[1])

    def query(self, head_id: int, relation_id: int) -> FloatArray:
        """Return the vector ``q`` such that every tail score is ``q @ e_t``."""
        head = self.entity[head_id]
        relation = self.relation[relation_id]
        if self.model == "distmult":
            return (head * relation).astype(np.float64)
        h_re, h_im = np.split(head, 2)
        r_re, r_im = np.split(relation, 2)
        return np.concatenate((h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re))

    def queries(self, head_ids: IntArray, relation_ids: IntArray) -> FloatArray:
        head_ids = np.asarray(head_ids, dtype=np.int64)
        relation_ids = np.asarray(relation_ids, dtype=np.int64)
        if head_ids.shape != relation_ids.shape or head_ids.ndim != 1:
            raise ValueError("head_ids and relation_ids must be same-length vectors")
        return np.vstack(
            [self.query(int(h), int(r)) for h, r in zip(head_ids, relation_ids, strict=True)]
        )

    def scores(self, head_id: int, relation_id: int) -> FloatArray:
        return self.query(head_id, relation_id) @ self.entity.T

    def tail_rank(self, head_id: int, relation_id: int, tail_id: int) -> int:
        return rank_from_scores(self.scores(head_id, relation_id), tail_id)


@dataclass(frozen=True)
class ProtectionSubspace:
    """SVD and Gram representations of protected query directions."""

    directions: FloatArray
    singular_values: FloatArray
    row_basis: FloatArray
    rank: int
    dimension: int
    tolerance: float

    @classmethod
    def from_directions(
        cls,
        directions: NDArray[np.floating],
        *,
        dimension: int | None = None,
        tolerance: float = DEFAULT_SVD_TOLERANCE,
    ) -> ProtectionSubspace:
        matrix = _as_float64(directions)
        if matrix.ndim != 2:
            raise ValueError("protected directions must be a rank-2 array")
        dim = int(matrix.shape[1] if dimension is None else dimension)
        if matrix.shape[1] != dim:
            raise ValueError("protected direction width does not match dimension")
        if tolerance < 0:
            raise ValueError("SVD tolerance must be non-negative")
        if matrix.shape[0] == 0:
            return cls(matrix, np.empty(0), np.empty((0, dim)), 0, dim, tolerance)
        _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.sum(singular_values > tolerance))
        return cls(
            matrix,
            singular_values.astype(np.float64),
            vh[:rank].astype(np.float64),
            rank,
            dim,
            tolerance,
        )

    @property
    def gram(self) -> FloatArray:
        return self.directions.T @ self.directions

    @property
    def null_dimension(self) -> int:
        return self.dimension - self.rank

    def project(
        self,
        vector: NDArray[np.floating],
        constrained_rank: int | None = None,
    ) -> FloatArray:
        value = _as_float64(vector)
        used = self.rank if constrained_rank is None else min(max(constrained_rank, 0), self.rank)
        if used == 0:
            return value.copy()
        basis = self.row_basis[:used]
        return value - basis.T @ (basis @ value)


@dataclass(frozen=True)
class LocalityResult:
    protected_count: int
    damage_count: int
    ranks_before: IntArray
    ranks_after: IntArray

    @property
    def no_damage(self) -> bool:
        return self.damage_count == 0


@dataclass(frozen=True)
class SweepResult:
    method: str
    parameter: float
    rank_before: int
    rank_after: int
    success: bool
    locality: LocalityResult
    update_norm: float
    residual_max_abs: float
    requested_rank: int | None = None
    used_rank: int | None = None

    @property
    def safe_success(self) -> bool:
        return self.success and self.locality.no_damage


def rank_from_scores(scores: NDArray[np.floating], tail_id: int) -> int:
    values = _as_float64(scores)
    if values.ndim != 1:
        raise ValueError("scores must be a vector")
    return int(1 + np.sum(values > values[tail_id]))


def promotion_score_gap(
    scores: NDArray[np.floating],
    target_id: int,
    *,
    top_k: int = 10,
    margin: float = DEFAULT_MARGIN,
) -> float:
    values = _as_float64(scores)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    k = min(top_k, values.size)
    threshold = float(np.partition(values, values.size - k)[values.size - k])
    return max(0.0, threshold - float(values[target_id]) + margin)


def direct_promotion_delta(
    query: NDArray[np.floating],
    score_gap: float,
    *,
    noop_tolerance: float = DEFAULT_NOOP_TOLERANCE,
) -> FloatArray:
    direction = _as_float64(query)
    denominator = float(direction @ direction)
    if score_gap <= 0 or denominator < noop_tolerance:
        return np.zeros_like(direction)
    return (score_gap / denominator) * direction


def rank_truncated_delta(
    query: NDArray[np.floating],
    score_gap: float,
    protection: ProtectionSubspace,
    fraction: float,
    *,
    noop_tolerance: float = DEFAULT_NOOP_TOLERANCE,
) -> tuple[FloatArray, int, int]:
    """Solve the rank-truncated projected promotion used by the paper frontier."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    direction = _as_float64(query)
    if direction.shape != (protection.dimension,):
        raise ValueError("query width does not match protection subspace")
    requested = min(
        protection.dimension,
        max(0, math.floor(fraction * protection.dimension + 0.5)),
    )
    used = min(requested, protection.rank)
    projected = protection.project(direction, used)
    denominator = float(direction @ projected)
    if score_gap <= 0 or abs(denominator) < noop_tolerance:
        return np.zeros_like(direction), requested, used
    return (score_gap / denominator) * projected, requested, used


def ridge_preservation_delta(
    query: NDArray[np.floating],
    score_gap: float,
    protection: ProtectionSubspace,
    lambda_preserve: float,
    *,
    jitter: float = 1e-8,
    noop_tolerance: float = DEFAULT_NOOP_TOLERANCE,
) -> FloatArray:
    """Solve the frozen ridge-preservation update with Sherman-Morrison scaling."""
    if lambda_preserve <= 0:
        raise ValueError("lambda_preserve must be positive")
    if jitter <= 0:
        raise ValueError("jitter must be positive")
    direction = _as_float64(query)
    if score_gap <= 0:
        return np.zeros_like(direction)
    base = lambda_preserve * protection.gram + jitter * np.eye(protection.dimension)
    z = np.linalg.solve(base, direction)
    denominator = 1.0 + float(direction @ z)
    if abs(denominator) < noop_tolerance:
        return np.zeros_like(direction)
    return (score_gap / denominator) * z


def candidate_column_locality(
    model: KGEEmbeddings,
    protected_triples: IntArray,
    edited_tail_id: int,
    delta: NDArray[np.floating],
    *,
    top_k: int = 10,
) -> LocalityResult:
    """Count protected top-k facts lost under a candidate-column intervention.

    Only the edited entity's candidate-tail column changes. This is the locality
    universe used by the rank/locality frontier, not a full entity-row-and-column edit.
    """
    triples = np.asarray(protected_triples, dtype=np.int64)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("protected_triples must have shape (n, 3)")
    if triples.shape[0] == 0:
        empty = np.empty(0, dtype=np.int64)
        return LocalityResult(0, 0, empty, empty)
    edited_tail = model.entity[edited_tail_id] + _as_float64(delta)
    before_ranks: list[int] = []
    after_ranks: list[int] = []
    for head_id, relation_id, true_tail_id in triples:
        query = model.query(int(head_id), int(relation_id))
        before = query @ model.entity.T
        after = before.copy()
        after[edited_tail_id] = float(query @ edited_tail)
        before_ranks.append(rank_from_scores(before, int(true_tail_id)))
        after_ranks.append(rank_from_scores(after, int(true_tail_id)))
    before_array = np.asarray(before_ranks, dtype=np.int64)
    after_array = np.asarray(after_ranks, dtype=np.int64)
    eligible = before_array <= top_k
    damage = int(np.sum(eligible & (after_array > top_k)))
    return LocalityResult(int(np.sum(eligible)), damage, before_array, after_array)


def build_relation_protection(
    model: KGEEmbeddings,
    train_triples: IntArray,
    relation_id: int,
    *,
    top_k: int = 10,
    max_protected: int = 5000,
    svd_tolerance: float = DEFAULT_SVD_TOLERANCE,
) -> tuple[IntArray, ProtectionSubspace]:
    triples = np.asarray(train_triples, dtype=np.int64)
    relation_triples = triples[triples[:, 1] == relation_id]
    protected: list[NDArray[np.int64]] = []
    directions: list[FloatArray] = []
    for triple in relation_triples:
        head_id, rel_id, tail_id = map(int, triple)
        if model.tail_rank(head_id, rel_id, tail_id) <= top_k:
            protected.append(triple)
            directions.append(model.query(head_id, rel_id))
            if max_protected > 0 and len(protected) >= max_protected:
                break
    triple_array = np.asarray(protected, dtype=np.int64).reshape(-1, 3)
    direction_array = np.asarray(directions, dtype=np.float64).reshape(-1, model.dimension)
    return triple_array, ProtectionSubspace.from_directions(
        direction_array, dimension=model.dimension, tolerance=svd_tolerance
    )


def run_frontier(
    model: KGEEmbeddings,
    train_triples: IntArray,
    edit_triple: NDArray[np.integer],
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    top_k: int = 10,
    margin: float = DEFAULT_MARGIN,
    max_protected: int = 5000,
) -> list[SweepResult]:
    """Evaluate both preservation frontiers for one edit triple."""
    head_id, relation_id, target_id = map(int, edit_triple)
    scores = model.scores(head_id, relation_id)
    rank_before = rank_from_scores(scores, target_id)
    score_gap = promotion_score_gap(scores, target_id, top_k=top_k, margin=margin)
    protected, subspace = build_relation_protection(
        model, train_triples, relation_id, top_k=top_k, max_protected=max_protected
    )
    query = model.query(head_id, relation_id)
    results: list[SweepResult] = []
    for fraction in fractions:
        delta, requested, used = rank_truncated_delta(query, score_gap, subspace, fraction)
        after = scores.copy()
        after[target_id] = float(query @ (model.entity[target_id] + delta))
        locality = candidate_column_locality(model, protected, target_id, delta, top_k=top_k)
        residual = subspace.directions @ delta
        results.append(
            SweepResult(
                "rank_truncated",
                float(fraction),
                rank_before,
                rank_from_scores(after, target_id),
                rank_from_scores(after, target_id) <= top_k,
                locality,
                float(np.linalg.norm(delta)),
                float(np.max(np.abs(residual))) if residual.size else 0.0,
                requested,
                used,
            )
        )
    for lambda_preserve in lambdas:
        delta = ridge_preservation_delta(query, score_gap, subspace, lambda_preserve)
        after = scores.copy()
        after[target_id] = float(query @ (model.entity[target_id] + delta))
        locality = candidate_column_locality(model, protected, target_id, delta, top_k=top_k)
        residual = subspace.directions @ delta
        results.append(
            SweepResult(
                "ridge",
                float(lambda_preserve),
                rank_before,
                rank_from_scores(after, target_id),
                rank_from_scores(after, target_id) <= top_k,
                locality,
                float(np.linalg.norm(delta)),
                float(np.max(np.abs(residual))) if residual.size else 0.0,
            )
        )
    return results
