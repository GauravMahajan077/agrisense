"""Find \n-based line numbers (as Notepad displays) for citation verification.

Usage: python kb/find_line.py "<file>" "<regex>" [...more patterns]
"""
import re
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def find(path: Path, pattern: str) -> None:
    rx = re.compile(pattern, re.I)
    for i, line in enumerate(read_text(path).split("\n"), 1):
        if rx.search(line):
            print(f"{i}: {line.strip()[:160]}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    for pat in sys.argv[2:]:
        print(f"--- {pat} ---")
        find(path, pat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
