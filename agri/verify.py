"""Single source of truth for verifying the canonical notebook + generated cells.

Run from the agri/ directory:  python -B verify.py
"""
import ast
import pathlib
import re
import sys

s = pathlib.Path("agrisense_kaggle.py").read_text(encoding="utf-8")
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


print("=== canonical source ===")
try:
    ast.parse(s)
    check("combined source parses", True)
except SyntaxError as e:
    check("combined source parses", False, str(e))

marks = [m.start() for m in re.finditer(r"(?m)^# %%", s)]
check("22 cells (1 markdown + 21 code)", len(marks) == 22, f"got {len(marks)}")

gaps = []
for a, b in zip(marks, marks[1:]):
    lines = s[a:b].split("\n")[:-1]
    n = 0
    for ln in reversed(lines):
        if ln.strip() == "":
            n += 1
        else:
            break
    gaps.append(n)
check("21 boundaries", len(gaps) == 21, f"got {len(gaps)}")
check("every boundary is exactly 10 blank lines", all(g == 10 for g in gaps), str(sorted(set(gaps))))

print("\n=== generated cells ===")
cells = sorted(pathlib.Path("cells").glob("*.py"))
check("21 code cells generated", len(cells) == 21, f"got {len(cells)}")
for f in cells:
    try:
        ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        check(f"{f.name} parses", False, str(e))
check("all cells parse", not fails or not any("parses" in x for x in fails))

# The generated cells must contain the CURRENT source, not a stale copy. This is the check
# that would have caught the itertuples fix landing after the last split_cells run.
c5 = next((f for f in cells if "cell_5" in f.name), None)
check("cell 5 exists", c5 is not None)
if c5:
    t = c5.read_text(encoding="utf-8")
    check("cell 5 is not stale: itertuples bug is gone", "itertuples()" not in t)
    check("cell 5 has the source_alias pair report", "check the source_alias mapping" in t)
    check("cell 5 has the cross-source counter", "clusters spanning >1 source" in t)
    check("cell 5 has the pairs-merged label", "key pairs merged" in t)
    check("cell 5 has largest-cluster composition", "largest cluster composition" in t)

c2 = next((f for f in cells if "cell_2" in f.name), None)
if c2:
    t2 = c2.read_text(encoding="utf-8")
    check("cell 2 has source_alias guard", "shadow" in t2)

print("\n=== README ===")
rm = pathlib.Path("README.md").read_text(encoding="utf-8")
check("README documents 6 classes", "Taxonomy: 6 classes" in rm)
check("README documents the source_alias rationale", "source_alias" in rm)
check("README shows Sheath_Blight as single-source", "| **1** |" in rm)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
