# Reproducibility and data contracts

## Portable bundle schema

Bundles are compressed NumPy archives loaded with `allow_pickle=False`.

| Key | Type | Shape | Meaning |
|---|---:|---:|---|
| `schema_version` | int64 scalar | — | Must equal `1` |
| `model` | Unicode scalar | — | `distmult` or `complex` |
| `entity_embeddings` | numeric | `(entities, d)` | Candidate/head embeddings |
| `relation_embeddings` | numeric | `(relations, d)` | Relation embeddings |
| `train_triples` | int64 | `(n_train, 3)` | `(head, relation, tail)` IDs |
| `valid_triples` | int64 | `(n_valid, 3)` | Validation triples |
| `test_triples` | int64 | `(n_test, 3)` | Test triples |

All arrays are range-checked and converted to float64 or int64 after loading.
No checkpoint executable code or Python object deserialization is used.

## Case schema

The case CSV requires `h_id,r_id,t_id`. Select hyperparameters on validation
cases, freeze the decision, and report once on disjoint test cases. This package
does not automate selection because the scientific selection rule belongs to the
experiment protocol, not the numerical solver.

## Protected universe

For each edit relation, the runner scans training triples in their stored order,
retains facts whose original tail rank is at most k, and applies the optional
`max_protected` cap. The resulting query directions define both the preservation
subspace and relation-scope locality universe.

## Output safety

The CLI refuses an existing output directory. Every run records input SHA-256
hashes, platform/Python metadata, frozen grids, and executable quality gates.
Keep raw output directories immutable; create a new directory for every protocol
revision.
