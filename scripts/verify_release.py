#!/usr/bin/env python3
"""Fail closed when the release tree differs from its manifest."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
MAX_FILE_BYTES = 1_000_000
IGNORED_PARTS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}
IGNORED_FILES = {".coverage"}
BANNED_SUFFIXES = {".env", ".log", ".pdf", ".pt", ".pth", ".ckpt", ".parquet"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> set[str]:
    files = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.name in IGNORED_FILES or path == MANIFEST:
            continue
        files.add(path.relative_to(ROOT).as_posix())
    return files


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = release_files()
    errors: list[str] = []
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        errors.append(f"tree mismatch: missing={missing}, extra={extra}")
    token_patterns = [
        re.compile("gh" + r"p_[A-Za-z0-9]{20,}"),
        re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}"),
        re.compile("sk" + r"-[A-Za-z0-9]{20,}"),
        re.compile("AKIA" + r"[A-Z0-9]{16}"),
        re.compile("BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY"),
    ]
    for relative in sorted(actual & set(expected)):
        path = ROOT / relative
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"oversized file: {relative}")
        if path.suffix.lower() in BANNED_SUFFIXES:
            errors.append(f"banned release suffix: {relative}")
        if sha256(path) != expected[relative]:
            errors.append(f"hash mismatch: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in token_patterns):
            errors.append(f"credential-like token: {relative}")
        local_prefixes = ("/ho" + "me/", "/wo" + "rk/")
        if any(prefix in text for prefix in local_prefixes):
            errors.append(f"machine-local absolute path: {relative}")
    if errors:
        print("RELEASE VERIFICATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"RELEASE VERIFICATION PASSED ({len(actual)} manifested files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
