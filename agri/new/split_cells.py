"""Split a notebook source file into one file per cell.

Why this exists: pasting a multi-cell notebook from a single file means eyeballing the
boundaries and copying a range. One file per cell means Ctrl+A, Ctrl+C, paste.

The combined notebook source stays the single source of truth. Edit it (or edit the
matching file in cells/, then re-run this to resync) and re-run:

    python split_cells.py                       # default: agrisense_notebook.py (3 cells)
    python split_cells.py ../old/agrisense_kaggle.py   # legacy 23-cell notebook

The default notebook is a thin 3-cell wrapper around `agrisense.py`. Its MODULE cell is
generated from the module via the `# %% include:agrisense.py` directive: the directive line
is replaced with the module's content before splitting, so the module stays the single
source of truth for the logic and the notebook stays self-contained when pasted.

Two cell kinds are emitted because Kaggle needs both:
    *.md   -> paste into a MARKDOWN cell
    *.py   -> paste into a CODE cell

Filenames are derived from each cell's own "# CELL N - title" header, so they stay readable
and stable. Re-running is idempotent and clears stale files.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# The repo lives under a path with non-ASCII characters, so printing OUT on a cp1252 console
# raises UnicodeEncodeError. Harmless to the files we write; fatal to the progress print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "agrisense_notebook.py"
if not SRC.is_absolute():
    SRC = HERE / SRC
OUT = HERE / "cells"

# A cell starts at a "# %%" line. "# %% [markdown]" is a prose cell, plain "# %%" is code.
MARKER = re.compile(r"^# %%(?: \[(markdown)\])?\s*$", re.M)
# A "# %% include:<path>" line is replaced with the referenced file's content before parsing.
INCLUDE = re.compile(r"^#\s*%%\s*include:([^\s]+)\s*$", re.M)
HEADER = re.compile(r"^#\s*(CELL\s+(\d+)\s*[—\-–:]\s*(.+?))\s*$")
# Keep titles ASCII-safe for Windows filenames, without mangling readability.
UNSAFE = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 44) -> str:
    s = UNSAFE.sub("_", text.lower()).strip("_")
    return (s[:limit].rstrip("_") or "part")


def resolve_includes(src: str) -> str:
    """Replace `# %% include:<path>` lines with the referenced file's content.

    The directive is a comment line in the notebook source; the included file becomes part
    of the same cell. Only one level of include is supported (the module includes nothing).
    """
    def repl(m: re.Match) -> str:
        path = HERE / m.group(1)
        if not path.exists():
            raise SystemExit(f"include target missing: {path}")
        return path.read_text(encoding="utf-8").rstrip("\n")
    return INCLUDE.sub(repl, src)


def parse(src: str) -> list[dict]:
    """Return [{kind, body, title, num}] in file order. body keeps its own header line."""
    marks = list(MARKER.finditer(src))
    cells: list[dict] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        body = src[m.end():end]
        # The separator banner we insert between cells is not part of any cell.
        body = re.sub(r"\n*={70,}\n(?:=#{2,}.*\n)+={70,}\n", "\n", body)
        body = body.strip("\n")
        kind = (m.group(1) or "code").lower()
        head = HEADER.match(body.split("\n", 1)[0]) if body else None
        cells.append(
            {
                "kind": "md" if kind == "markdown" else "py",
                "body": body,
                "title": head.group(1).strip() if head else "",
                "num": int(head.group(2)) if head else None,
                "raw_head": body.split("\n", 1)[0].lstrip("# ").strip() if body else "",
            }
        )
    return cells


def verify(cells: list[dict], src: str) -> None:
    """Round-trip: every non-empty line of every cell must appear in the original, in order.

    Marker lines are consumed by the parser, so they are excluded from the expected side.
    Separator-tolerant on purpose — it compares the non-empty line sequence, so it stays valid
    no matter how much whitespace we pad between cells with.
    """
    want = [ln.strip() for ln in src.splitlines() if ln.strip() and not MARKER.fullmatch(ln)]
    got: list[str] = []
    for c in cells:
        got += [ln.strip() for ln in c["body"].splitlines() if ln.strip()]
    if want != got:
        for i, (a, b) in enumerate(zip(want, got)):
            if a != b:
                raise SystemExit(f"round-trip mismatch at line {i + 1}:\n  file: {a!r}\n  cell: {b!r}")
        raise SystemExit(f"round-trip length mismatch: {len(want)} lines in file, {len(got)} in cells")


def label_for(c: dict, i: int) -> str:
    """Readable stem for a cell: its own header if it has one, else a positional label."""
    if c["raw_head"]:
        return c["raw_head"]
    if c["num"] is not None:
        return f"cell{c['num']:02d}"
    return f"part{i:02d}"


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    src = SRC.read_text(encoding="utf-8")
    src = resolve_includes(src)
    cells = parse(src)
    if not cells:
        raise SystemExit("no '# %%' markers found — is this the notebook file?")
    verify(cells, src)

    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():
        if old.is_file() and old.name != "README.md":
            old.unlink()

    names, used = [], set()
    for i, c in enumerate(cells):
        base = f"{i:02d}_{slug(label_for(c, i), 52)}"
        # Guarantee uniqueness even if two cells share a header.
        stem, n = base, 2
        while stem in used:
            stem = f"{base}_{n}"; n += 1
        used.add(stem)
        name = f"{stem}.{c['kind']}"
        (OUT / name).write_text(c["body"] + "\n", encoding="utf-8")
        names.append((i, name, c))

    md = [
        "# Notebook cells — paste order",
        "",
        f"Generated by `split_cells.py` from `{SRC.name}`. **Do not hand-edit here** —",
        "edit the combined file and re-run `python split_cells.py`.",
        "",
        f"{len(cells)} cells. Paste top to bottom into a fresh Kaggle notebook.",
        "`.md` → **Markdown** cell. `.py` → **Code** cell.",
        "",
        "| # | File | Kind | Cell |",
        "|---|---|---|---|",
    ]
    for i, name, c in names:
        kind = "markdown" if c["kind"] == "md" else "code"
        md.append(f"| {i} | [`{name}`]({name}) | {kind} | {c['title'] or '—'} |")
    md += [
        "",
        "## Notes",
        "",
        "- Cell 1 (CONFIG) is the only cell that normally needs editing. Cell 2 (MODULE) is",
        "  generated from `agrisense.py` via the `# %% include:agrisense.py` directive — do not",
        "  hand-edit it. Cell 3 (RUN) calls `run(CFG)`.",
        "- Add Input is far less error-prone than pasting cell 1's defaults blindly: if you",
        "  change `CFG['sources']` here, change it in Kaggle too.",
        "- The legacy 23-cell notebook is `../old/agrisense_kaggle.py`; regenerate its cells with",
        "  `python split_cells.py ../old/agrisense_kaggle.py` (this overwrites cells/).",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(f"{len(cells)} cells -> {OUT}")
    for i, name, c in names:
        kind = "md " if c["kind"] == "md" else "py "
        n = len(c["body"].splitlines())
        print(f"  {i:2d}  [{kind}] {name:<58} {n:4d} lines")
    print(f"\npaste order written to {OUT / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
