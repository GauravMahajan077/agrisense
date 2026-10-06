"""Phase 3 unit tests for the brute-force D4 dedupe helpers.

Extracts the SHIPPED implementations (d4_dist_matrix, brute_clusters, cross_class_mask,
leak_scan) straight out of agrisense.py via AST, so the tests exercise exactly what runs on
Kaggle — not a copy. The helpers are Pipeline methods but never touch `self`, so they are
bound to a dummy instance and called as plain functions.

Run from agri/:  python -B test_phase3_dedupe.py
"""
import ast
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

SRC = pathlib.Path("agrisense.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)

FUNCS = ["d4_dist_matrix", "brute_clusters", "cross_class_mask", "leak_scan"]

ns = {"np": np, "defaultdict": defaultdict}
for name in FUNCS:
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.get_source_segment(SRC, node)
    exec(compile(code, f"<{name}>", "exec"), ns)


class _Dummy:
    pass


_dummy = _Dummy()
d4_dist_matrix = lambda *a, **k: ns["d4_dist_matrix"](_dummy, *a, **k)
brute_clusters = lambda *a, **k: ns["brute_clusters"](_dummy, *a, **k)
cross_class_mask = lambda *a, **k: ns["cross_class_mask"](_dummy, *a, **k)
leak_scan = lambda *a, **k: ns["leak_scan"](_dummy, *a, **k)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ---- d4_dist_matrix ----
print("=== d4_dist_matrix ===")
rng = np.random.RandomState(0)
keys = rng.randint(0, 2**64, (5, 8), dtype=np.uint64)
D = d4_dist_matrix(keys)
check("identity diagonal is 0", np.all(np.diag(D) == 0))
check("symmetric", np.all(D == D.T))
Dc = d4_dist_matrix(keys, chunk=2)
check("chunked == full", np.array_equal(D, Dc))


def rot90(k):
    """Rotate the 64-bit pHash grid 90 degrees (bit i = row-major cell i)."""
    bits = [(k >> i) & 1 for i in range(64)]
    grid = np.array(bits, np.uint8).reshape(8, 8)
    out = 0
    for i, b in enumerate(np.rot90(grid, 1).ravel()):
        out |= int(b) << i
    return np.uint64(out)


def pack(g):
    v = 0
    for i, b in enumerate(g.ravel()):
        v |= int(b) << i
    return v


def d4_variants(k):
    """All 8 D4 transforms of the 64-bit pHash grid (row-major bit packing)."""
    bits = np.array([(k >> i) & 1 for i in range(64)], np.uint8).reshape(8, 8)
    out = []
    for t in range(4):
        g = np.rot90(bits, t)
        out.append(pack(g))
        out.append(pack(np.fliplr(g)))
    return [np.uint64(v) for v in out]


# D4-min: two images whose variant orbits overlap must be distance 0
k0 = np.uint64(0x0123456789ABCDEF)
V = np.array([d4_variants(k0), d4_variants(rot90(k0)), d4_variants(k0 ^ 1)], np.uint64)
D2 = d4_dist_matrix(V)
check("D4-min: shared orbit variant -> distance 0", D2[0, 1] == 0)
# D4-min: one bit off a shared variant -> distance 1 (proves the min is taken over all
# 8x8 variant pairs, not just the identity pair)
check("D4-min: 1 bit off a shared variant -> distance 1", D2[0, 2] == 1)
D2c = d4_dist_matrix(V, chunk=1)
check("chunked == full (D4 orbits)", np.array_equal(D2, D2c))

# ---- brute_clusters ----
print("=== brute_clusters ===")
# 4 images: 0,1 same class near; 2 cross-class near to 0; 3 far from everyone
D4 = np.array([
    [0, 2, 3, 20],
    [2, 0, 9, 21],
    [3, 9, 0, 22],
    [20, 21, 22, 0],
], np.uint8)
labels = np.array([0, 0, 1, 0], np.int32)
cid, merged = brute_clusters(D4, 4, labels)
check("same-class near pair merged", cid[0] == cid[1])
check("cross-class near pair NOT merged", cid[0] != cid[2])
check("far pair not merged", cid[0] != cid[3])
check("merged count == 1", merged == 1)

# ---- cross_class_mask ----
print("=== cross_class_mask ===")
mask = cross_class_mask(D4, 4, labels)
check("cross-class near pair flagged", mask[0] and mask[2])
check("same-class near pair not flagged", not mask[1])
check("far image not flagged", not mask[3])

# ---- leak_scan ----
print("=== leak_scan ===")
man = pd.DataFrame({"split": ["train", "train", "val", "test"]})
leaks = leak_scan(D4, man, tight=4)
n_pairs = sum(len(v) for v in leaks.values())
check("same-split near pair not a leak", ("train", "train") not in leaks)
check("cross-split near pair reported", n_pairs == 1)
check("leak keys are split pairs", set(leaks) == {("train", "val")})

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)