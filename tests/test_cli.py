import csv
import json
from pathlib import Path

from kge_rank_locality.cli import main


def test_demo_writes_complete_outputs(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    main(["demo", "--out-dir", str(output), "--seed", "9"])
    results = output / "results"
    gates = json.loads((results / "quality_gates.json").read_text())
    assert gates["passed"] is True
    with (results / "per_case.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 15
