"""Fail if obvious secret material is committed."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", ".venv", "data", "logs", "models", "__pycache__"}
PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(
        r"(?im)(api[_-]?secret|api[_-]?key|password|token)[^\S\r\n]*="
        r"[^\S\r\n]*['\"]?[^'\"\s\r\n]+"
    ),
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return files


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PATTERNS:
            if pattern.search(text):
                failures.append(str(path.relative_to(ROOT)))
                break
    if failures:
        print("Potential secrets detected:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("no_secrets_check=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
