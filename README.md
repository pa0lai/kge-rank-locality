# kge-rank-locality

[![CI](https://github.com/pa0lai/kge-rank-locality/actions/workflows/ci.yml/badge.svg)](https://github.com/pa0lai/kge-rank-locality/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reproducible rank/locality frontiers for **linear knowledge graph embeddings**.
The package implements the rank-truncated null-space and ridge-preservation
updates used to study the tension between successful link-prediction edits and
preservation of existing top-k facts.

## Why this repository

A target-tail edit should improve the requested triple without silently evicting
facts that were already ranked in the top k. This repository makes that tradeoff
explicit and auditable:

- deterministic, float64 NumPy implementations;
- DistMult and ComplEx support through a shared linear-tail interface;
- frozen rank-fraction rounding, SVD tolerance, margin, and ridge jitter;
- candidate-column locality with success, no-damage, and safe-success metrics;
- pickle-free portable bundles and append-only output directories;
- tests, quality gates, hashes, and CI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest

kge-rank-locality demo --out-dir demo_run
column -s, -t < demo_run/results/summary.csv
```

The demo generates a deterministic synthetic DistMult bundle, four deliberately
hard edit cases, and both complete frontiers.

## Run on your embeddings

Prepare a portable NPZ bundle and a case CSV:

```python
import numpy as np
from kge_rank_locality import ExperimentBundle, KGEEmbeddings, save_bundle

model = KGEEmbeddings(entity_embeddings, relation_embeddings, "distmult")
bundle = ExperimentBundle(model, train_triples, valid_triples, test_triples)
save_bundle("bundle.npz", bundle)
```

```csv
h_id,r_id,t_id
12,3,91
48,7,203
```

Then run:

```bash
kge-rank-locality sweep \
  --bundle bundle.npz \
  --cases cases.csv \
  --out-dir results
```

`results/` contains `per_case.csv`, `summary.csv`, `run_config.json`, and
`quality_gates.json`. Existing output directories are rejected to prevent
accidental mixing or overwrite.

## Methods in one line

For edit direction `q`, desired score increase `b`, and protected-direction
matrix `C`:

- direct promotion: `delta = b q / (q^T q)`;
- rank-truncated projection: remove the first `m` right-singular directions of
  `C`, then rescale the projected `q` to meet `q^T delta = b`;
- ridge preservation: penalize `||C delta||^2` using the frozen regularized
  linear solve.

See [docs/METHOD.md](docs/METHOD.md) for exact equations and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for data contracts.

## Scope

The exact projection methods require a score linear in the candidate tail
embedding. DistMult and ComplEx satisfy that condition. Nonlinear models such as
RotatE are intentionally rejected rather than silently approximated.

The reported locality intervention changes the edited entity's **candidate-tail
column only**. It does not claim to measure a full row-and-column entity update;
the distinction is part of the API and documentation.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest --cov=kge_rank_locality --cov-report=term-missing
python -m build
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes.

## Citation

Use the metadata in [CITATION.cff](CITATION.cff). Replace the placeholder paper
title and DOI there once the archival publication record is available.

## License

MIT. See [LICENSE](LICENSE).
