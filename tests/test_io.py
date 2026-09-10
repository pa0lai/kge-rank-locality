from pathlib import Path

import numpy as np
import pytest

from kge_rank_locality import ExperimentBundle, KGEEmbeddings, load_bundle, save_bundle


def test_bundle_round_trip_without_pickle(tmp_path: Path) -> None:
    model = KGEEmbeddings(np.eye(3), np.ones((1, 3)), "distmult")
    triples = np.asarray([[0, 0, 1]], dtype=np.int64)
    bundle = ExperimentBundle(model, triples, triples, triples)
    path = tmp_path / "bundle.npz"
    save_bundle(path, bundle)
    restored = load_bundle(path)
    np.testing.assert_array_equal(restored.model.entity, model.entity)
    np.testing.assert_array_equal(restored.train_triples, triples)


def test_bundle_rejects_out_of_range_ids() -> None:
    model = KGEEmbeddings(np.eye(2), np.ones((1, 2)), "distmult")
    with pytest.raises(ValueError, match="out-of-range"):
        ExperimentBundle(model, np.asarray([[0, 0, 3]]), np.empty((0, 3)), np.empty((0, 3)))
