"""Fail if obvious secret material is committed.

The point is to catch a real key that someone pasted into the source tree, not
every line that happens to contain the word "key". A naive
``api_key\\s*=\\s*.+`` regex flags ``api_key=None``, ``api_key=os.getenv(...)``
and ``api_key="key"`` in tests, which trains people to ignore the check. So we
only report assignments whose *value* actually looks like credential material:
a long, mixed-character literal that is not an obvious placeholder.
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "data",
    "logs",
    "models",
    "state",
    "runs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
IGNORED_FILES = {".env.example"}
TEXT_SUFFIXES = {
    ".py", ".txt", ".md", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini",
    ".ps1", ".sh", ".env", ".bat",
}

# Vendor-specific tokens are unambiguous: report them wherever they appear.
HARD_PATTERNS = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

# Generic ``name = "value"`` assignments; the value is judged separately.
# ``(?:^|[^A-Za-z0-9])`` rather than ``\b`` so prefixed names such as
# ``BINANCE_API_SECRET`` are matched -- ``\b`` fails after an underscore.
ASSIGNMENT = re.compile(
    r"""(?ix)
    (?:^|[^A-Za-z0-9])
    (api[_-]?secret|api[_-]?key|secret[_-]?key|access[_-]?token|
     auth[_-]?token|password|passwd|bearer)
    \s*[:=]\s*
    (?P<quote>['"])?(?P<value>[^'"\s\n]*)(?(quote)(?P=quote))
    """
)

PLACEHOLDERS = {
    "", "none", "null", "todo", "changeme", "change_me", "placeholder", "example",
    "your_api_key", "your_api_secret", "your-key-here", "xxx", "test", "testing",
    "dummy", "fake", "secret", "key", "apikey", "api_key", "api_secret", "password",
    "token", "abc", "abc123", "foo", "bar", "baz", "sample", "redacted",
}

MIN_SECRET_LENGTH = 20
MIN_ENTROPY_BITS_PER_CHAR = 2.5


def shannon_entropy(value: str) -> float:
    """Bits of entropy per character. Real keys score high; words score low."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


# Characters that mean "this is a code expression, not a literal credential"
# -- e.g. ``api_secret = self._require_credentials()``.
CODE_CHARACTERS = set("()[]{}.,+ \t")


def looks_like_a_real_secret(value: str, *, quoted: bool) -> bool:
    stripped = value.strip()
    if stripped.lower() in PLACEHOLDERS:
        return False
    if len(stripped) < MIN_SECRET_LENGTH:
        return False
    # Env-var interpolation and format placeholders are not secrets.
    if stripped.startswith(("$", "{", "%", "<")) or "os.getenv" in stripped:
        return False
    # An unquoted value in source code is an expression (a variable, a call),
    # not a pasted key. Only .env-style bare values should reach the entropy test.
    if not quoted and CODE_CHARACTERS & set(stripped):
        return False
    return shannon_entropy(stripped) >= MIN_ENTROPY_BITS_PER_CHAR


def iter_files() -> list[Path]:
    files: list[Path] = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in IGNORED_DIRS]
        for name in names:
            path = Path(directory) / name
            if name in IGNORED_FILES:
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or name.startswith(".env"):
                files.append(path)
    return files


def scan(path: Path) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings

    for line_number, line in enumerate(text.splitlines(), start=1):
        if "no-secret-scan" in line:
            continue
        for label, pattern in HARD_PATTERNS:
            if pattern.search(line):
                findings.append(f"{path.relative_to(ROOT)}:{line_number}: {label}")
        for match in ASSIGNMENT.finditer(line):
            if looks_like_a_real_secret(
                match.group("value"), quoted=bool(match.group("quote"))
            ):
                findings.append(
                    f"{path.relative_to(ROOT)}:{line_number}: "
                    f"hard-coded {match.group(1).lower()}"
                )
    return findings


def main() -> int:
    findings = [finding for path in iter_files() for finding in scan(path)]

    if findings:
        print("Potential secrets detected:")
        for finding in findings:
            print(f"- {finding}")
        print("\nMove secrets to .env.local (git-ignored) and read them via Settings.")
        return 1

    print("no_secrets_check=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
