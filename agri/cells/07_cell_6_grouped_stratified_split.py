# CELL 6 — grouped stratified split
# `grouped_split`, `leak_scan` and `clusters_at` are defined in CELL 5, so this cell uses the
# exact same splitter. Do not re-define them here: a second copy is how the two halves drift
# apart.

sp = CFG["split"]
man, got = grouped_split(man, sp, SEED)          # defined in Cell 5, shared with this cell
print("split sizes: " + ", ".join(f"{k}={int(got[k])}" for k in sp)
      + f"  (target {int(sp['train']*len(man))}/{int(sp['val']*len(man))}/{int(sp['test']*len(man))})")
tab = man.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=list(sp))
print("\nper-class x per-split counts:")
print(tab.to_string())

# The check that would have caught the un-stratified split: every class must land near its
# own target ratio, not merely sum to the right total.
print("\nper-class deviation from target ratio:")
worst = 0.0
for c in tab.index:
    n = int(tab.loc[c].sum())
    dev = max(abs(int(tab.loc[c, s]) - n * sp[s]) / n for s in sp)
    worst = max(worst, dev)
    flag = ""
    if dev > 0.05:
        flag = "   <- OFF TARGET"
    dead = [s for s in sp if int(tab.loc[c, s]) == 0]
    if dead:
        flag += f"   <- NO SAMPLES in {dead}: its metric is undefined"
    thin = [s for s in sp if 0 < int(tab.loc[c, s]) < 50]
    if thin:
        flag += f"   <- thin: {thin}"
    print(f"  {c:24s} n={n:5d}  max dev {dev:5.1%}{flag}")
print(f"  worst per-class deviation: {worst:.1%}")
if worst > 0.05:
    print("  WARNING: per-class ratios are off. Do not trust val/test metrics - they are")
    print("           measuring the split, not the model.")
if any(int(tab.loc[c, s]) == 0 for c in tab.index for s in sp):
    print("  WARNING: a class has no val or no test samples. Grouping won over stratification,")
    print("           which is correct (splitting the cluster would leak), but that class's")
    print("           macro-F1 contribution is undefined and the run must be redone with a")
    print("           different seed.")

sets = {k: set(man.loc[man["split"] == k, "cluster"]) for k in sp}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = sets[a] & sets[b]
    print(f"  cluster overlap {a}/{b}: {len(overlap)}")
    assert not overlap, f"LEAK: {len(overlap)} clusters in both {a} and {b}"
if man["cluster"].nunique() == len(man):
    print("  NOTE: every cluster is a singleton - grouping was inert, the leak assert is weak")
print("  NOTE: the overlap check above is STRUCTURAL. Split is assigned per cluster, so")
print("        overlap is impossible by construction - that assert cannot fail and proves")
print("        nothing on its own. The scan below is the one that can actually fail.")

# ---- the leakage check that can actually fail ----
# Cell 5 merged near-duplicates at dedupe.near_dist on the D4-min pHash. Images that are
# visually near-identical but pHash-different (re-cropped, re-encoded, rotated off-grid) never
# got merged, so they CAN straddle train and val/test. Scan for exactly that at a deliberately
# TIGHTER threshold. Reuses the canonical keys, so there is no re-hashing cost.
#
# Completeness: a pair differing in <= T bits has at most T dirty bands, so with B bands it
# shares >= B - T fully-matching bands. B=8 => complete (no false negatives) for T <= 7.
TIGHT = 7
_csize = man.groupby("cluster").size()
if CFG["dedupe"]["enable"] and "keys" in globals() and len(keys) == len(man):
    leaks = leak_scan(man, keys, tight=TIGHT, B=CFG["dedupe"]["bands"])
    n_pairs = sum(len(v) for v in leaks.values())
    print(f"\ncross-split near-duplicate scan (hamming <= {TIGHT}, tighter than the merge "
          f"threshold of {CFG['dedupe']['near_dist']}):")
    if n_pairs:
        n_img = len({i for ps in leaks.values() for i, _ in ps})
        print(f"  {n_pairs} near-duplicate pair(s) straddle splits, touching {n_img} images "
              f"({n_img/len(man):.1%} of the manifest):")
        for pair, ps in sorted(leaks.items(), key=lambda kv: -len(kv[1])):
            i, j = ps[0]
            # .loc, NOT .iloc: this Series is indexed by CLUSTER LABEL, and after union-find
            # merging those labels are sparse (0, 1, 5, 17, ...) — passing them positionally
            # raises IndexError as soon as a label exceeds the group count. Synthetic tests with
            # contiguous ids hid this completely; real cluster ids do not.
            cs = _csize.loc[[man["cluster"].iat[i], man["cluster"].iat[j]]].tolist()
            print(f"    {len(ps):5d}  {pair[0]}/{pair[1]}   e.g. "
                  f"{man['class'].iat[i]} vs {man['class'].iat[j]}  (cluster sizes {cs})")
    else:
        print("  0 - no train image is a near-duplicate of any val/test image  OK")
else:
    print("\ncross-split near-duplicate scan SKIPPED (no canonical keys available)")
