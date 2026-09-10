# Contributing

1. Create a focused branch and keep generated results out of Git.
2. Install with `python -m pip install -e '.[dev]'`.
3. Run `ruff check .`, `pytest`, and `python -m build` before opening a PR.
4. Add a regression test for every numerical or protocol change.
5. Do not change frozen tolerances, grids, ranking semantics, or locality
   definitions without documenting a new protocol version.

Please report security-sensitive issues privately as described in SECURITY.md.
