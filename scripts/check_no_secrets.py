"""Scan the working tree for accidentally committed credentials.

The naive approach -- grep for ``api_key\\s*=`` -- flags every line of ordinary
code (``api_key=None``, ``self.api_key = config.api_key``) and quickly gets
ignored, which is worse than no scanner at all. This checks the *value*
instead: a real key is long, high-entropy and not made of code punctuation.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Shorter than this and it is not a Binance key (they are 64 chars).
MIN_SECRET_LENGTH = 20

#: Shannon entropy per character. English prose and code identifiers sit
#: around 2.0-2.5; random keys are above 3.5.
MIN_ENTROPY_BITS_PER_CHAR = 2.5

#: Characters that mean "this is an expression, not a literal secret".
CODE_CHARACTERS = set("()[]{}.,+ \t")

# Note the leading (?:^|[^A-Za-z0-9]) rather than \b: a word boundary does not
# match between "_" and "A", so \bapi_key\b would silently skip the single most
# important name in this project, BINANCE_API_KEY.
SECRET_KEY_PATTERN = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])"
    r"(?:api[_-]?key|api[_-]?secret|secret[_-]?key|password|passwd|token|private[_-]?key)"
    r"\s*[:=]\s*(?P<value>.+)"
)

SKIP_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "models", "state", "logs", ".ruff_cache", ".mypy_cache",
}

SKIP_FILES = {".env.example"}

SCAN_SUFFIXES = {".py", ".toml", ".txt", ".md", ".yml", ".yaml", ".json", ".ps1", ".sh", ".cfg", ".ini"}

#: Placeholders that appear in docs and templates.
PLACEHOLDERS = {
    "none", "null", "true", "false", "your_api_key", "your_api_secret",
    "changeme", "xxx", "todo", "example", "placeholder", "redacted",
}


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def looks_like_a_real_secret(value: str, *, quoted: bool) -> bool:
    value = value.strip()
    if not value:
        return False

    # Strip a trailing comment and surrounding quotes.
    for comment in (" #", "  //"):
        if comment in value:
            value = value.split(comment)[0].strip()

    if quoted and value[-1:] == value[:1]:
        # A well-formed literal: "abc123...". Strip the quotes and judge the
        # contents.
        value = value[1:-1]
    elif any(char in CODE_CHARACTERS for char in value):
        # Either unquoted, or opened a quote without closing it on this line.
        # Both mean we are looking at an expression -- os.getenv("X"),
        # config.api_key, or a string concatenation -- rather than a literal
        # secret. Judging the entropy of source code produces only noise.
        return False

    if len(value) < MIN_SECRET_LENGTH:
        return False
    if value.lower() in PLACEHOLDERS:
        return False
    if value.startswith("${") or value.startswith("os.") or value.startswith("<"):
        return False
    if set(value) <= {"x", "X", "*", "0", "-", "_"}:
        return False

    return shannon_entropy(value) >= MIN_ENTROPY_BITS_PER_CHAR


def scan_file(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return findings

    for number, line in enumerate(lines, start=1):
        # Escape hatch for known-safe lines.
        if "# no-secret-scan" in line:
            continue
        match = SECRET_KEY_PATTERN.search(line)
        if not match:
            continue
        raw_value = match.group("value")
        quoted = raw_value.lstrip()[:1] in {'"', "'"}
        if looks_like_a_real_secret(raw_value, quoted=quoted):
            findings.append((number, line.strip()[:120]))
    return findings


def main() -> int:
    problems: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES and path.name not in {".env"}:
            continue

        for number, line in scan_file(path):
            problems.append(f"{path.relative_to(ROOT)}:{number}: {line}")

    if (ROOT / ".env").exists():
        problems.append(".env exists in the repository root -- make sure it is gitignored")

    if problems:
        print("Potential secrets found:\n")
        for problem in problems:
            print(f"  {problem}")
        print("\nIf a finding is a false positive, append '# no-secret-scan' to the line.")
        return 1

    print("No secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
