# Release audit

Date: 2026-09-11

## Scope

This is a whitelist-built sibling repository. It contains the reusable numerical
core for the rank/locality experiments, not a copy of the original research
workspace.

Included:

- rank-truncated projection and ridge-preservation solvers;
- DistMult and ComplEx linear-tail scoring;
- candidate-column locality metrics;
- portable input contracts, CLI, tests, CI, and documentation.

Excluded:

- pretrained checkpoints and third-party datasets;
- raw or derived experiment artifacts;
- cluster scripts, host paths, logs, caches, and temporary files;
- paper drafts, PDFs, figures, and reviewer material.

## Source provenance

The formulas were extracted from `pa0lai/KnowledgeEditonKG` at Git commit
`0fbcb18448cb25fbe4105f12b68ace999b3df652`. Because one camera-ready runner
was not tracked in that workspace snapshot, exact source hashes are recorded in
`RELEASE_MANIFEST.json` rather than claiming the commit alone identifies every
input.

## Verification record

- Ruff: passed.
- Tests: 12 passed on local Python 3.10.18.
- Coverage: 89% total.
- Packaging: sdist and universal wheel built successfully.
- Source equivalence: 100 randomized rank-truncated trials, maximum absolute
  difference `1.395e-10`.
- Source equivalence: 100 randomized ridge trials, maximum absolute difference
  `9.770e-15`.
- No dataset, checkpoint, PDF, log, or artifact file is distributed.
- The release verifier checks the exact whitelist and hashes, file size,
  credential-like patterns, banned suffixes, and machine-local absolute paths.

The equivalence audit establishes numerical agreement for the extracted solver
formulas. It does not claim that the synthetic demo reproduces paper results;
real results require the frozen data, case manifests, and checkpoints from the
original experiment protocol.
