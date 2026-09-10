"""Command-line interface for reproducible frontier sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kge_rank_locality import DEFAULT_FRACTIONS, DEFAULT_LAMBDAS, KGEEmbeddings, run_frontier
from kge_rank_locality.io import ExperimentBundle, load_bundle, save_bundle, validate_cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_cases(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"h_id", "r_id", "t_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("case CSV must contain h_id,r_id,t_id columns")
        rows = [(int(row["h_id"]), int(row["r_id"]), int(row["t_id"])) for row in reader]
    if not rows:
        raise ValueError("case CSV is empty")
    return np.asarray(rows, dtype=np.int64)


def _synthetic_bundle(seed: int) -> tuple[ExperimentBundle, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = KGEEmbeddings(rng.normal(size=(64, 16)), rng.normal(size=(6, 16)), "distmult")
    train = []
    cases = []
    for relation_id in range(4):
        for head_id in range(12):
            scores = model.scores(head_id, relation_id)
            train.append((head_id, relation_id, int(np.argmax(scores))))
        case_head = 20 + relation_id
        cases.append((case_head, relation_id, int(np.argmin(model.scores(case_head, relation_id)))))
    triples = np.asarray(train, dtype=np.int64)
    return ExperimentBundle(model, triples, triples[:4], triples[4:8]), np.asarray(cases)


def _run(
    bundle_path: Path,
    cases_path: Path,
    out_dir: Path,
    top_k: int,
    max_protected: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    bundle = load_bundle(bundle_path)
    cases = validate_cases(_load_cases(cases_path), bundle)
    out_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        results = run_frontier(
            bundle.model,
            bundle.train_triples,
            case,
            top_k=top_k,
            max_protected=max_protected,
        )
        case_id = f"{int(case[0])}:{int(case[1])}:{int(case[2])}"
        for result in results:
            rows.append(
                {
                    "case_index": case_index,
                    "case_id": case_id,
                    "h_id": int(case[0]),
                    "r_id": int(case[1]),
                    "t_id": int(case[2]),
                    "method": result.method,
                    "parameter": result.parameter,
                    "rank_before": result.rank_before,
                    "rank_after": result.rank_after,
                    "success_at_k": int(result.success),
                    "protected_n": result.locality.protected_count,
                    "damage_count": result.locality.damage_count,
                    "no_damage": int(result.locality.no_damage),
                    "safe_success": int(result.safe_success),
                    "update_norm": result.update_norm,
                    "residual_max_abs": result.residual_max_abs,
                    "requested_rank": result.requested_rank,
                    "used_rank": result.used_rank,
                }
            )
    _write_csv(out_dir / "per_case.csv", rows)
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), float(row["parameter"]))].append(row)
    summary = []
    for (method, parameter), group in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "parameter": parameter,
                "N": len(group),
                "success_at_k": float(np.mean([int(row["success_at_k"]) for row in group])),
                "no_damage": float(np.mean([int(row["no_damage"]) for row in group])),
                "safe_success": float(np.mean([int(row["safe_success"]) for row in group])),
                "damage_mean": float(np.mean([int(row["damage_count"]) for row in group])),
                "update_norm_mean": float(np.mean([float(row["update_norm"]) for row in group])),
            }
        )
    _write_csv(out_dir / "summary.csv", summary)
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "bundle": str(bundle_path.resolve()),
        "bundle_sha256": _sha256(bundle_path),
        "cases": str(cases_path.resolve()),
        "cases_sha256": _sha256(cases_path),
        "top_k": top_k,
        "max_protected": max_protected,
        "fractions": DEFAULT_FRACTIONS,
        "lambdas": DEFAULT_LAMBDAS,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    expected = len(cases) * (len(DEFAULT_FRACTIONS) + len(DEFAULT_LAMBDAS))
    unique_keys = {(r["case_id"], r["method"], r["parameter"]) for r in rows}
    success_valid = all(
        int(row["success_at_k"]) == int(int(row["rank_after"]) <= top_k) for row in rows
    )
    safe_success_valid = all(
        int(row["safe_success"])
        == int(bool(row["success_at_k"]) and bool(row["no_damage"]))
        for row in rows
    )
    gates = {
        "row_count_exact": len(rows) == expected,
        "unique_case_method_parameter": len(unique_keys) == expected,
        "success_definition": success_valid,
        "safe_success_definition": safe_success_valid,
    }
    gates["passed"] = all(gates.values())
    (out_dir / "quality_gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    if not gates["passed"]:
        raise RuntimeError("one or more output quality gates failed")


def _demo(out_dir: Path, seed: int) -> None:
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    bundle, cases = _synthetic_bundle(seed)
    bundle_path = out_dir / "bundle.npz"
    cases_path = out_dir / "cases.csv"
    save_bundle(bundle_path, bundle)
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("h_id", "r_id", "t_id"))
        writer.writerows(cases.tolist())
    _run(bundle_path, cases_path, out_dir / "results", top_k=10, max_protected=5000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kge-rank-locality", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep", help="run both frontiers from a portable bundle")
    sweep.add_argument("--bundle", type=Path, required=True)
    sweep.add_argument("--cases", type=Path, required=True)
    sweep.add_argument("--out-dir", type=Path, required=True)
    sweep.add_argument("--top-k", type=int, default=10)
    sweep.add_argument("--max-protected", type=int, default=5000)
    demo = subparsers.add_parser("demo", help="run a deterministic synthetic quickstart")
    demo.add_argument("--out-dir", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "sweep":
        if args.top_k < 1 or args.max_protected < 0:
            raise SystemExit("--top-k must be positive and --max-protected non-negative")
        _run(args.bundle, args.cases, args.out_dir, args.top_k, args.max_protected)
    else:
        _demo(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
