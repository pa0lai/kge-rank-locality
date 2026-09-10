import numpy as np
import pytest

from kge_rank_locality import (
    KGEEmbeddings,
    ProtectionSubspace,
    candidate_column_locality,
    direct_promotion_delta,
    rank_truncated_delta,
    ridge_preservation_delta,
    run_frontier,
)


def test_distmult_scores_and_rank() -> None:
    model = KGEEmbeddings(
        np.asarray([[1.0, 2.0], [2.0, 0.0], [-1.0, 1.0]]),
        np.asarray([[3.0, 1.0]]),
        "distmult",
    )
    np.testing.assert_allclose(model.scores(0, 0), [7.0, 6.0, -1.0])
    assert model.tail_rank(0, 0, 2) == 3


def test_complex_query_matches_manual_score() -> None:
    entities = np.asarray([[1.0, 2.0, 0.5, -1.0], [0.2, 1.1, -0.3, 0.7]])
    relations = np.asarray([[0.4, -0.2, 0.6, 0.3]])
    model = KGEEmbeddings(entities, relations, "complex")
    h_re, h_im = np.split(entities[0], 2)
    r_re, r_im = np.split(relations[0], 2)
    t_re, t_im = np.split(entities[1], 2)
    expected = np.sum((h_re * r_re - h_im * r_im) * t_re + (h_re * r_im + h_im * r_re) * t_im)
    assert model.scores(0, 0)[1] == pytest.approx(expected)


def test_rank_truncation_endpoints() -> None:
    query = np.asarray([2.0, 1.0, -1.0])
    protection = ProtectionSubspace.from_directions(np.asarray([[1.0, 0.0, 0.0]]))
    direct = direct_promotion_delta(query, 2.5)
    zero, requested_zero, used_zero = rank_truncated_delta(query, 2.5, protection, 0.0)
    strict, requested_one, used_one = rank_truncated_delta(query, 2.5, protection, 1.0)
    np.testing.assert_allclose(zero, direct)
    assert (requested_zero, used_zero) == (0, 0)
    assert (requested_one, used_one) == (3, 1)
    assert protection.directions @ strict == pytest.approx([0.0], abs=1e-12)
    assert query @ strict == pytest.approx(2.5)


def test_rank_fraction_uses_frozen_half_up_rounding() -> None:
    protection = ProtectionSubspace.from_directions(np.eye(8))
    _delta, requested, used = rank_truncated_delta(np.ones(8), 1.0, protection, 0.125)
    assert (requested, used) == (1, 1)


def test_ridge_reduces_protected_residual_as_lambda_increases() -> None:
    query = np.asarray([1.0, 1.0])
    protection = ProtectionSubspace.from_directions(np.asarray([[1.0, 0.0]]))
    low = ridge_preservation_delta(query, 1.0, protection, 1e-4)
    high = ridge_preservation_delta(query, 1.0, protection, 100.0)
    assert abs(high[0]) < abs(low[0])


def test_candidate_column_locality_detects_a_drop() -> None:
    model = KGEEmbeddings(
        np.asarray([[1.0], [0.9], [0.8], [0.0]]),
        np.asarray([[1.0]]),
        "distmult",
    )
    protected = np.asarray([[0, 0, 1]], dtype=np.int64)
    result = candidate_column_locality(model, protected, 2, np.asarray([0.2]), top_k=2)
    assert result.protected_count == 1
    assert result.damage_count == 1


def test_frontier_shape_and_definitions() -> None:
    rng = np.random.default_rng(4)
    model = KGEEmbeddings(rng.normal(size=(32, 8)), rng.normal(size=(2, 8)), "distmult")
    train = np.asarray([(h, 0, int(np.argmax(model.scores(h, 0)))) for h in range(8)])
    scores = model.scores(20, 0)
    case = np.asarray([20, 0, int(np.argmin(scores))])
    results = run_frontier(model, train, case)
    assert len(results) == 15
    assert all(
        result.safe_success == (result.success and result.locality.no_damage)
        for result in results
    )


@pytest.mark.parametrize("model_name,width", [("distmult", 3), ("complex", 4)])
def test_embedding_validation_accepts_supported_models(model_name: str, width: int) -> None:
    KGEEmbeddings(np.zeros((2, width)), np.zeros((1, width)), model_name)  # type: ignore[arg-type]
