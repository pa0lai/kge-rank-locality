"""Portable, pickle-free experiment bundle I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kge_rank_locality.core import IntArray, KGEEmbeddings


@dataclass(frozen=True)
class ExperimentBundle:
    model: KGEEmbeddings
    train_triples: IntArray
    valid_triples: IntArray
    test_triples: IntArray

    def __post_init__(self) -> None:
        for name in ("train_triples", "valid_triples", "test_triples"):
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"{name} must have shape (n, 3)")
            if value.size and (
                value[:, [0, 2]].min() < 0
                or value[:, [0, 2]].max() >= self.model.entity.shape[0]
                or value[:, 1].min() < 0
                or value[:, 1].max() >= self.model.relation.shape[0]
            ):
                raise ValueError(f"{name} contains an out-of-range identifier")
            object.__setattr__(self, name, value)


def save_bundle(path: str | Path, bundle: ExperimentBundle) -> None:
    """Write a compressed bundle without object arrays or pickle payloads."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(1, dtype=np.int64),
        model=np.asarray(bundle.model.model),
        entity_embeddings=bundle.model.entity,
        relation_embeddings=bundle.model.relation,
        train_triples=bundle.train_triples,
        valid_triples=bundle.valid_triples,
        test_triples=bundle.test_triples,
    )


def load_bundle(path: str | Path) -> ExperimentBundle:
    """Load and validate a portable bundle with pickle explicitly disabled."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "schema_version",
            "model",
            "entity_embeddings",
            "relation_embeddings",
            "train_triples",
            "valid_triples",
            "test_triples",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"bundle is missing keys: {sorted(missing)}")
        version = int(archive["schema_version"])
        if version != 1:
            raise ValueError(f"unsupported bundle schema version: {version}")
        model_name = str(archive["model"])
        if model_name not in {"distmult", "complex"}:
            raise ValueError(f"unsupported model in bundle: {model_name!r}")
        model = KGEEmbeddings(
            archive["entity_embeddings"],
            archive["relation_embeddings"],
            model_name,  # type: ignore[arg-type]
        )
        return ExperimentBundle(
            model,
            archive["train_triples"],
            archive["valid_triples"],
            archive["test_triples"],
        )


def validate_cases(cases: NDArray[np.integer], bundle: ExperimentBundle) -> IntArray:
    value = np.asarray(cases, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("cases must have shape (n, 3)")
    # Reuse the bundle's range validation without duplicating the contract.
    empty = np.empty((0, 3), dtype=np.int64)
    ExperimentBundle(bundle.model, value, empty, empty)
    return value
