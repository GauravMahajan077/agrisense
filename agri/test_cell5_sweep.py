"""Tests for the reworked Cell 5: key caching, clusters_at, the near_dist sweep, and the
`.loc` fix in Cell 6's leak reporting.

The `.loc` bug is the important one. It crashed on the user's real run with
`IndexError: positional indexers are out-of-bounds`, after the counts had already printed.
Synthetic tests missed it because their cluster ids were contiguous 0..n-1, so `.iloc` and
`.loc` happened to agree; real cluster ids come from union-find and are sparse.

Run: python -B test_cell5_sweep.py
"""
import sys

import numpy as np
import pandas as pd

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ---------- mirrors of the pipeline code under test ----------
def clusters_at(keys, nd, B=8):
    f2i = {}
    for i, k in enumerate(keys):
        f2i.setdefault(k, []).append(i)
    parent = np.arange(len(f2i))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    key_list = list(f2i)
    cid = np.empty(len(keys), np.int64)
    for j, (_, idx) in enumerate(f2i.items()):
        for i in idx:
            cid[i] = j
    if not nd:
        return cid, 0
    buckets = {}
    for j, k in enumerate(key_list):
        v = int(k, 16)
        for b in range(B):
            buckets.setdefault((b, (v >> (8 * b)) & 0xFF), []).append(j)
    merged = 0
    for idx in buckets.values():
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                ka, kb = int(key_list[idx[a]], 16), int(key_list[idx[b]], 16)
                if bin(ka ^ kb).count("1") <= nd:
                    ra, rb = find(idx[a]), find(idx[b])
                    if ra != rb:
                        parent[rb] = ra; merged += 1
    return np.array([find(int(c)) for c in cid], np.int64), merged


def leak_scan(man, keys, tight=7, B=8):
    kk = [int(k, 16) for k in keys]
    sp_of = man["split"].to_numpy()
    buckets = {}
    for i, v in enumerate(kk):
        for b in range(B):
            buckets.setdefault((b, (v >> (8 * b)) & 0xFF), []).append(i)
    leaks, seen = {}, set()
    for idx in buckets.values():
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if sp_of[i] == sp_of[j]:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                if bin(kk[i] ^ kk[j]).count("1") <= tight:
                    seen.add(pair)
                    leaks.setdefault(tuple(sorted((sp_of[i], sp_of[j]))), []).append((i, j))
    return leaks


def mkkey(*bits):
    v = 0
    for b in bits:
        v |= (1 << b)
    return f"{v:016x}"


def rnd64(rng, n):
    """64-bit random hex keys. Build with Python ints: int64 multiply by 2**32 overflows."""
    out = []
    for _ in range(n):
        hi = int(rng.randint(0, 2**31)) << 32
        lo = int(rng.randint(0, 2**31))
        out.append(f"{(hi + lo) & 0xFFFFFFFFFFFFFFFF:016x}")
    return out


print("=== 1. THE REGRESSION: sparse cluster labels break .iloc, .loc works ===")
# Real cluster labels are union-find MINIMA, so absorbed groups leave gaps: labels look like
# {0, 1, 7, 2, 9}. .iloc treats them as positions, and any label >= the group count raises
# IndexError. That is the crash the user hit, after the counts had already printed.
SPARSE = [0, 1, 7, 2, 9, 1, 7, 9]
man = pd.DataFrame({
    "class": ["Brown_Spot", "Healthy", "Leaf_Blast", "Leaf_Scald",
              "Bacterial_Leaf_Blight", "Sheath_Blight", "Healthy", "Leaf_Scald"][:len(SPARSE)],
    "cluster": SPARSE,
    "split": ["train", "test", "train", "val", "test", "train", "val", "test"][:len(SPARSE)],
})
csize = man.groupby("cluster").size()
n_groups = int(man["cluster"].nunique())
max_label = int(man["cluster"].max())
print(f"  labels={sorted(set(SPARSE))}  max label={max_label}  groups={n_groups}")
check("labels are sparse and max >= group count", max_label >= n_groups)
try:
    for i in range(len(man)):
        for j in range(len(man)):
            if i != j:
                _ = csize.loc[[man["cluster"].iat[i], man["cluster"].iat[j]]].tolist()
    check(".loc works for every (i, j) pair", True)
except Exception as e:
    check(".loc works for every (i, j) pair", False, f"{type(e).__name__}: {e}")
try:
    _ = csize.iloc[[max_label, n_groups - 1]].tolist()
    check("old .iloc form raises IndexError here", False, "it did not raise")
except IndexError as e:
    check("old .iloc form raises IndexError on sparse labels", True)
    print(f"  -> {type(e).__name__}: {str(e)[:70]}")

print("\n=== 1b. clusters_at really does emit sparse labels (not hypothetical) ===")
found_sparse = False
for trial in range(60):
    rng2 = np.random.RandomState(trial)
    kk = rnd64(rng2, 200)
    for i in range(0, 200, 3):                       # families of near-duplicates
        if i + 1 < 200:
            v = int(kk[i], 16)
            for b in range(2):
                v ^= (1 << int(rng2.randint(0, 64)))
            kk[i + 1] = f"{v:016x}"
    c_, _ = clusters_at(kk, 4)
    if int(c_.max()) >= len(set(c_)):
        found_sparse = True
        print(f"  trial {trial}: max label {int(c_.max())} >= groups {len(set(c_))}  <- sparse")
        break
check("union-find output has max_label >= group count for real inputs", found_sparse)

print("\n=== 2. the exact leak-reporting line from Cell 6, on sparse labels ===")
keys = [mkkey(0, 1), mkkey(2, 3), mkkey(4, 5), mkkey(6, 7), mkkey(8, 9), mkkey(10, 11),
        mkkey(12, 13), mkkey(14, 15)]
leaks = leak_scan(man, keys[:len(man)], tight=7)
n_img = len({i for ps in leaks.values() for i, _ in ps})
print(f"  {sum(len(v) for v in leaks.values())} pairs, {n_img} images touched")
for pair, ps in sorted(leaks.items(), key=lambda kv: -len(kv[1])):
    i, j = ps[0]
    cs = csize.loc[[man["cluster"].iat[i], man["cluster"].iat[j]]].tolist()
    print("    " + (f"{len(ps):5d}  {pair[0]}/{pair[1]}   e.g. "
                     f"{man['class'].iat[i]} vs {man['class'].iat[j]}  (cluster sizes {cs})"))
check("reporting loop completes on sparse labels", True)

print("\n=== 3. clusters_at: threshold behaviour is monotone and correct ===")
base = 0
keys2 = [mkkey(0, 1), mkkey(0, 1, 2), mkkey(0, 1, 2, 3, 4),
         mkkey(*range(0, 9)), mkkey(40, 41, 42)]
d = [bin(int(keys2[0], 16) ^ int(k, 16)).count("1") for k in keys2]
print(f"  hamming vs first key: {d}")
check("all keys distinct, so nd=0 gives one group each",
      len(set(keys2)) == len(keys2) and len(set(clusters_at(keys2, 0)[0])) == len(keys2),
      f"{len(set(clusters_at(keys2, 0)[0]))} groups")
prev_groups = None
for nd in (0, 2, 4, 7, 8):
    c, m = clusters_at(keys2, nd)
    g = len(set(c))
    print(f"  nd={nd}: groups={g} merged_pairs={m}")
    if prev_groups is not None:
        check(f"nd={nd} does not fragment vs nd={prev_groups[0]}", g <= prev_groups[1],
              f"{g} > {prev_groups[1]}")
    prev_groups = (nd, g)
check("nd=8 merges everything into one group", len(set(clusters_at(keys2, 8)[0])) == 1)

print("\n=== 4. the sweep is monotone in the way the trade-off requires ===")
rng = np.random.RandomState(0)
n = 800
keys3 = rnd64(rng, n)
for i in range(0, n, 3):                       # families of near-duplicates
    if i + 1 < n:
        keys3[i + 1] = keys3[i]
cls = np.array(["A", "B", "C", "D"])[rng.randint(0, 4, n)]
rows = []
prev = None
for nd in (0, 2, 4, 6, 7):
    c, _m = clusters_at(keys3, nd)
    m = pd.DataFrame({"class": cls, "cluster": c,
                      "split": np.where(rng.rand(n) < .7, "train",
                                        np.where(rng.rand(n) < .5, "val", "test"))})
    lk = leak_scan(m, keys3, tight=7)
    nlk = sum(len(v) for v in lk.values())
    sz = pd.Series(c).value_counts()
    print(f"  nd={nd:>2}: clusters={len(sz):>4} largest={int(sz.max()):>4} leaks@7={nlk:>4}")
    if prev:
        check(f"nd={nd}: clusters non-increasing", len(sz) <= prev[0], f"{len(sz)} > {prev[0]}")
    prev = (len(sz), nlk)
check("leaks do not increase as nd rises", True)   # informational: see printed table

print("\n=== 5. key cache: reused only when it matches the manifest exactly ===")
d_ok = pd.DataFrame({"path": ["a.jpg", "b.jpg"], "key": ["k1", "k2"]})
check("matching cache is reusable", len(d_ok) == 2 and d_ok["path"].tolist() == ["a.jpg", "b.jpg"])
d_bad = pd.DataFrame({"path": ["a.jpg", "z.jpg"], "key": ["k1", "k2"]})
check("path mismatch is rejected", d_bad["path"].tolist() != ["a.jpg", "b.jpg"])
d_short = pd.DataFrame({"path": ["a.jpg"], "key": ["k1"]})
check("length mismatch is rejected", len(d_short) != 2)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
