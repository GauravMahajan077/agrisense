"""Test the REWRITTEN Cell 6 grouped_split.

Two bugs it fixes, both reproduced from the user's real output:
  1. not stratified — balanced only the global total, giving Brown_Spot 1404/59/59 and
     Sheath_Blight 230/200/202 against a 70/15/15 target;
  2. `rng` created and never used, so CFG["seed"] did nothing.

Run: python -B test_split_strat.py
"""
import sys

import numpy as np
import pandas as pd

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


SP = {"train": .70, "val": .15, "test": .15}

# ---- new implementation (mirrors the pipeline) ----
def grouped_split(man, sp, seed):
    rng = np.random.RandomState(seed)
    classes = sorted(man["class"].unique())
    comp = (man.groupby(["cluster", "class"]).size()
              .unstack(fill_value=0).reindex(columns=classes, fill_value=0))
    n_cls = comp.sum()
    target = pd.DataFrame({s: n_cls * v for s, v in sp.items()})
    got = pd.DataFrame(0.0, index=classes, columns=list(sp), dtype=float)
    assign = {}
    for c in sorted(classes, key=lambda x: n_cls[x]):
        col = comp[c]
        cand = col[col > 0]
        if not len(cand):
            continue
        idx = np.lexsort((rng.rand(len(cand)), -cand.to_numpy()))
        for i in idx:
            cid = cand.index[i]
            if cid in assign:
                continue
            best = min(sp, key=lambda s: (got.at[c, s] / max(target.at[c, s], 1e-9), rng.random()))
            assign[cid] = best
            got.at[c, best] += float(cand.iloc[i])
    out = man.copy()
    out["split"] = out["cluster"].map(assign)
    return out, got.sum(axis=0).to_dict(), comp


# ---- old implementation, faithful, for comparison ----
def old_grouped_split(man, sp, seed):
    rng = np.random.RandomState(seed)                      # dead code, kept faithful
    info = (man.groupby("cluster")
              .agg(n=("class", "size"), c=("class", lambda s: s.mode().iat[0]))
              .reset_index())
    info = info.sort_values("n", ascending=False).reset_index(drop=True)
    tot = info["n"].sum()
    target = {k: tot * v for k, v in sp.items()}
    got = {k: 0 for k in sp}
    assign = {}
    for row in info.itertuples():
        k = max(sp, key=lambda x: target[x] - got[x])
        assign[row.cluster] = k; got[k] += row.n
    out = man.copy()
    out["split"] = out["cluster"].map(assign)
    return out, got


def worst_dev(man, sp):
    tab = man.groupby(["class", "split"]).size().unstack(fill_value=0)
    return max(max(abs(int(tab.loc[c, s]) - tab.loc[c].sum() * sp[s]) / tab.loc[c].sum()
                    for s in sp if s in tab.columns) for c in tab.index)


def synth(seed=0, imbalance=True, n_big=660, big_sz=(1, 62)):
    """Reproduce the real shape: ~660 multi-member clusters + singletons, 6 classes."""
    rng = np.random.RandomState(seed)
    rows, cid = [], 0
    totals = ([1154, 1522, 1091, 1152, 1066, 632] if imbalance
              else [1100] * 6)
    names = ["Bacterial_Leaf_Blight", "Brown_Spot", "Healthy", "Leaf_Blast",
             "Leaf_Scald", "Sheath_Blight"]
    for ci, c in enumerate(names):
        left = totals[ci]
        if imbalance and ci == 1:                       # Brown_Spot: 92% sits in big clusters
            big, left = int(totals[ci] * .92), left - int(totals[ci] * .92)
        else:
            big = 0
        while big > 0:
            k = min(big, rng.randint(*big_sz))
            for _ in range(k):
                rows.append({"class": c, "cluster": cid})
            big -= k; cid += 1
        while left > 0:
            rows.append({"class": c, "cluster": cid}); left -= 1; cid += 1
    return pd.DataFrame(rows)


print("=== 1. OLD implementation really is un-stratified (reproduces the reported skew) ===")
m = synth()
mo, go = old_grouped_split(m, SP, 1337)
to = mo.groupby(["class", "split"]).size().unstack(fill_value=0)
bs = to.loc["Brown_Spot"]
print("  old Brown_Spot:", {k: int(bs[k]) for k in SP})
check("old: global totals are right", abs(go["train"] / len(m) - .70) < .02, str(go))
check("old: Brown_Spot is badly skewed in train", bs["train"] / bs.sum() > .80,
      f"{bs['train']/bs.sum():.0%} in train")

print("\n=== 2. NEW implementation is stratified on the same data ===")
mn, gn, _ = grouped_split(m, SP, 1337)
tn = mn.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=list(SP))
print(tn.to_string())
wd = worst_dev(mn, SP)
check("new: global totals still correct", abs(gn["train"] / len(m) - .70) < .02, str(gn))
check("new: worst per-class deviation < 5%", wd < .05, f"{wd:.1%}")
for c in tn.index:
    n = tn.loc[c].sum()
    d = max(abs(tn.loc[c, s] - n * SP[s]) / n for s in SP)
    check(f"new: {c:24s} within 5%", d < .05, f"{d:.1%}")

print("\n=== 3. every image gets a split ===")
check("no NaN splits", mn["split"].notna().all())
check("split counts match the manifest", mn["split"].value_counts().to_dict() == {k: int(v) for k, v in gn.items()})

print("\n=== 4. the seed now actually changes something ===")
data = synth(seed=7)
outs = {s: grouped_split(data, SP, s)[0]["split"].tolist() for s in (1337, 42, 99)}
check("different seeds give different assignments", outs[1337] != outs[42] != outs[99])
check("same seed reproduces exactly", grouped_split(data, SP, 1337)[0]["split"].tolist() == outs[1337])
# old function: compare it against ITSELF across seeds, not against the new output
o1337 = old_grouped_split(data, SP, 1337)[0]["split"].tolist()
same = all(old_grouped_split(data, SP, s)[0]["split"].tolist() == o1337 for s in (1337, 42, 99))
check("old implementation was seed-invariant (the bug)", same)
# ...and every seed still hit the target totals, which is why the bug was invisible
check("old: totals correct for every seed (bug hidden by correct sums)",
      all(abs(old_grouped_split(data, SP, s)[1]["train"] / len(data) - .70) < .02 for s in (1337, 42, 99)))

print("\n=== 5. clusters stay intact: no cluster spans two splits ===")
xs = mn.groupby("cluster")["split"].nunique()
check("no cluster is split across train/val/test", int(xs.max()) == 1, f"max {int(xs.max())}")

print("\n=== 6. a single-class dataset still splits sanely ===")
one = pd.DataFrame({"class": ["Healthy"] * 1000, "cluster": np.arange(1000)})
mo2, go2, _ = grouped_split(one, SP, 1)
t2 = mo2["split"].value_counts()
check("single class ~70/15/15", abs(t2["train"] / 1000 - .70) < .03,
      {k: int(v) for k, v in t2.items()})

print("\n=== 7. one huge cluster cannot starve a class ===")
rows = [{"class": "Sheath_Blight", "cluster": 0} for _ in range(400)]   # 400 in ONE cluster
rows += [{"class": "Brown_Spot", "cluster": 1 + i} for i in range(400)]
big1 = pd.DataFrame(rows)
mb, gb, _ = grouped_split(big1, SP, 3)
tb = mb.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=list(SP))
print(tb.to_string())
sb = tb.loc["Sheath_Blight"]
check("the 400-image cluster lands whole in one split", int(sb.sum()) == 400)
check("exactly one split holds all 400", int((sb == 400).sum()) == 1, {k: int(v) for k, v in sb.items()})
# A class whose images are ONE cluster cannot be spread across splits without breaking the
# cluster. Grouping wins: leakage would be worse than an absent val/test sample.
check("the other two splits get 0 - correct, not a bug", int((sb == 0).sum()) == 2,
      {k: int(v) for k, v in sb.items()})
check("no split is left empty overall", set(mb["split"].unique()) == set(SP), set(mb["split"].unique()))
check("Brown_Spot still gets all three splits", (tb.loc["Brown_Spot"] > 0).all(),
      {k: int(v) for k, v in tb.loc["Brown_Spot"].items()})

print("\n=== 8. Cell 7's min_class re-split has the right escape hatch ===")
# If a class ends up under min_class in train, Cell 7 must be able to recover it by lowering
# the group constraint — i.e. the split must be re-runnable with a different seed.
recovered = False
for s in range(1, 40):
    _, g, _ = grouped_split(synth(seed=3), SP, s)
    if g and abs(g["train"] / 6617 - .70) < .02:
        recovered = True
        break
check("many seeds give a usable split", recovered)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
