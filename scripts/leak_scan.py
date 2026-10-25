"""Check that nothing personal is in the repository.

The repository is public. What must not be in it:

- **Personal paths.** A path with a user name in it says whose
  machine the code was written on, and the name is not the point of
  the example.
- **Keys.** A token in a file is a token in the history, and the
  history is public too.

**The same script runs locally and in the CI.** A scan that only
exists in one place is a scan that gets out of step - and the first
version of this one missed four paths because it looked for
``C:/Users/`` while Windows writes ``C:\\Users\\``.

The examples in the repository use paths that are obviously examples:
``D:\\Modelle\\...`` instead of a path under a user directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The file types that are scanned.
TYPES = ("*.py", "*.md", "*.json", "*.yml", "*.yaml", "*.html", "*.js",
         "*.css", "*.txt", "*.ps1", "*.sh")

#: The directories that are not scanned: they are not part of the
#: repository.
SKIP = {".git", "dist", "build", ".venv", "venv", "__pycache__",
        ".pytest_cache", ".ruff_cache", "results"}

#: What must not be in the repository.
#:
#: The path pattern needs both separators, because Windows paths are
#: written with backslashes and the first version of this scan only
#: looked at the slash.
PATTERNS = [
    (re.compile(r"[A-Za-z]:\\\\?Users\\\\?[a-z][a-z0-9_-]*", re.IGNORECASE),
     "a path under a user directory"),
    (re.compile(r"/home/[a-z][a-z0-9_-]*/"), "a path under a home directory"),
    (re.compile(r"/Users/[a-z][a-z0-9_-]*/"), "a path under a user directory"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "an API key"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "a GitHub token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "an AWS key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
]


def files() -> list[Path]:
    """The files that are scanned."""
    found: list[Path] = []
    for pattern in TYPES:
        for path in ROOT.rglob(pattern):
            if any(part in SKIP for part in path.parts):
                continue
            found.append(path)
    return sorted(set(found))


def scan() -> list[str]:
    """The findings. An empty list means nothing was found."""
    problems: list[str] = []
    for path in files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern, what in PATTERNS:
                if pattern.search(line):
                    problems.append(
                        f"{path.relative_to(ROOT)}:{number}: {what}\n"
                        f"    {line.strip()[:120]}")
    return problems


def main() -> int:
    problems = scan()
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print(f"\n{len(problems)} Treffer. Das darf nicht in ein "
              f"oeffentliches Repository.")
        return 1

    print(f"clean - {len(files())} Dateien geprueft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
