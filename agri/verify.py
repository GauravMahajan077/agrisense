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
check("23 cells (1 markdown + 22 code)", len(marks) == 23, f"got {len(marks)}")

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
check("22 boundaries", len(gaps) == 22, f"got {len(gaps)}")
check("every boundary is exactly 10 blank lines", all(g == 10 for g in gaps), str(sorted(set(gaps))))

print("\n=== generated cells ===")
cells = sorted(pathlib.Path("cells").glob("*.py"))
check("22 code cells generated", len(cells) == 22, f"got {len(cells)}")
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
    check("cell 5 has the pairs-merged label", "same-class key pairs merged" in t)
    check("cell 5 has largest-cluster composition", "largest cluster composition" in t)
    check("cell 5 has the brute-force D4 matrix", "d4_dist_matrix" in t)
    check("cell 5 has brute_clusters", "brute_clusters" in t)
    check("cell 5 has cross_class_mask", "cross_class_mask" in t)
    check("cell 5 has the cross-class counter", "cross-class near-duplicates" in t)

c1 = next((f for f in cells if "cell_1" in f.name), None)
if c1:
    t1 = c1.read_text(encoding="utf-8")
    check("cell 1 has held_out_source", "held_out_source" in t1)
    check("cell 1 has abstain_threshold", "abstain_threshold" in t1)
    check("cell 1 has bootstrap_iters", "bootstrap_iters" in t1)

c4 = next((f for f in cells if "cell_4" in f.name), None)
if c4:
    t4 = c4.read_text(encoding="utf-8")
    check("cell 4 routes held-out source", "man_held" in t4 and "held_out_source" in t4)

c6 = next((f for f in cells if "cell_6" in f.name), None)
if c6:
    t6 = c6.read_text(encoding="utf-8")
    check("cell 6 moves cross-class to train", "moved from val/test to train" in t6)
    check("cell 6 uses D-based leak_scan", "leak_scan(D, man" in t6)

c165 = next((f for f in cells if "cell_16_5" in f.name), None)
check("cell 16.5 exists", c165 is not None)
if c165:
    t165 = c165.read_text(encoding="utf-8")
    check("cell 16.5 has held-out eval", "SOURCE-HELD-OUT EVAL" in t165)
    check("cell 16.5 has bootstrap CI", "bootstrap 95% CI" in t165)
    check("cell 16.5 has abstain", "abstain@" in t165)
    check("cell 16.5 writes source_held_out.csv", "source_held_out.csv" in t165)

c17 = next((f for f in cells if "cell_17" in f.name), None)
if c17:
    t17 = c17.read_text(encoding="utf-8")
    check("cell 17 has abstain line", "abstain@" in t17)

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
