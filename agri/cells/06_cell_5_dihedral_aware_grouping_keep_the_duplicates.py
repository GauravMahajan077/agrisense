# CELL 5 — dihedral-aware grouping (KEEP the duplicates)
# pHash is not flip/rotate invariant, and `Rice_Leaf_AUG` is exactly flips and rotations.
# So we hash all 8 dihedral (D4) variants and use the MINIMUM as the canonical key: two
# images that are rotations/reflections of each other produce the same key.
#
# We then KEEP every image and assign each one its cluster id. Dropping duplicates here
# would collapse every cluster to a singleton and make the grouped split in Cell 6 vacuous.
import imagehash

def _dihedral(im):
    yield im
    for t in (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270):
        yield im.transpose(t)
    fl = im.transpose(Image.FLIP_LEFT_RIGHT)
    yield fl
    for t in (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270):
        yield fl.transpose(t)

def _canon_key(p, dihedral=True, probe=256):
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if probe:
                # draft() lets libjpeg do a SCALED decode — this is what makes preloading the
                # 2.4 MB shayanriyaz files tolerable at all.
                try: im.draft("RGB", (probe, probe))
                except Exception: pass
                im = im.resize((probe, probe), Image.BILINEAR)
            vs = [int(str(imagehash.phash(v)), 16) for v in (_dihedral(im) if dihedral else (im,))]
            return f"{min(vs):016x}"
    except Exception:
        return None

def clusters_at(keys, nd, B=8):
    """Per-row cluster id for a given near-duplicate threshold.

    Exact canonical-key matches are always clustered. `nd` additionally merges keys within
    Hamming distance `nd`, using B bands of 8 bits — complete (no false negatives) for
    nd <= B-1, because a pair differing in <= nd bits has at most nd dirty bands.
    """
    f2i = defaultdict(list)
    for i, k in enumerate(keys):
        f2i[k].append(i)
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
        return cid
    buckets = defaultdict(list)
    for j, k in enumerate(key_list):
        v = int(k, 16)
        for b in range(B):
            buckets[(b, (v >> (8 * b)) & 0xFF)].append(j)
    merged = 0
    for idx in buckets.values():
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                ka, kb = int(key_list[idx[a]], 16), int(key_list[idx[b]], 16)
                if bin(ka ^ kb).count("1") <= nd:
                    ra, rb = find(idx[a]), find(idx[b])
                    if ra != rb:
                        parent[rb] = ra; merged += 1
    out = np.array([find(int(c)) for c in cid], np.int64)
    return out, merged

def grouped_split(man, sp, seed):
    """Stratified group split. Defined here, not in Cell 6, so the threshold sweep below can
    measure leak counts for every candidate `near_dist` before one of them is chosen.

    Two properties matter, and the previous version had neither it claimed:
      * per-CLASS ratios, not just the global total. Global bin packing alone gave
        Brown_Spot 1404/59/59 and Sheath_Blight 230/200/202 on the real data, while the global
        sums looked perfect — so macro-F1 described the split, not the model.
      * the seed must do something. `rng` used to be created and never used.
    """
    rng = np.random.RandomState(seed)
    classes = sorted(man["class"].unique())
    comp = (man.groupby(["cluster", "class"]).size()
              .unstack(fill_value=0).reindex(columns=classes, fill_value=0))
    n_cls = comp.sum()
    target = pd.DataFrame({s: n_cls * v for s, v in sp.items()})
    got = pd.DataFrame(0.0, index=classes, columns=list(sp), dtype=float)
    assign = {}
    for c in sorted(classes, key=lambda x: n_cls[x]):          # rarest class first
        cand = comp[c][comp[c] > 0]
        if not len(cand):
            continue
        idx = np.lexsort((rng.rand(len(cand)), -cand.to_numpy()))   # biggest cluster first
        for i in idx:
            cid = cand.index[i]
            if cid in assign:                 # already claimed by a rarer class
                continue
            best = min(sp, key=lambda s: (got.at[c, s] / max(target.at[c, s], 1e-9), rng.random()))
            assign[cid] = best
            got.at[c, best] += float(cand.iloc[i])
    out = man.copy()
    out["split"] = out["cluster"].map(assign)
    assert out["split"].notna().all(), "some cluster was never assigned a split"
    return out, got.sum(axis=0).to_dict()

def leak_scan(man, keys, tight=7, B=8):
    """Count near-duplicate pairs that ended up in DIFFERENT splits.

    Cell 5 merges at `near_dist`; this looks at `tight` (> near_dist) so it finds the twins the
    merge missed. Reuses the canonical keys, so there is no re-hashing cost. A pair within
    `tight` bits shares up to `tight` byte-bands, so pairs are de-duplicated — otherwise every
    leak is reported once per shared band, inflating counts several-fold.
    """
    kk = [int(k, 16) for k in keys]
    sp_of = man["split"].to_numpy()
    buckets = defaultdict(list)
    for i, v in enumerate(kk):
        for b in range(B):
            buckets[(b, (v >> (8 * b)) & 0xFF)].append(i)
    leaks = defaultdict(list)
    seen = set()
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
                    leaks[tuple(sorted((sp_of[i], sp_of[j])))].append((i, j))
    return leaks

if CFG["dedupe"]["enable"]:
    # Cache the canonical keys. Hashing is the only expensive step (84 s for 6617 images);
    # the threshold sweep below re-runs it zero times, so trying several `near_dist` values is
    # nearly free. Without this, every threshold experiment cost a full re-hash and nobody
    # would try more than one.
    cache = OUT / "canon_keys.csv"
    keys = None
    if cache.exists() and not CFG["dedupe"].get("rehash", False):
        d = pd.read_csv(cache)
        if len(d) == len(man) and d["path"].tolist() == man["path"].tolist():
            keys = d["key"].tolist()
            print(f"reused cached canonical keys for {len(keys)} images "
                  f"(set dedupe.rehash=True, or delete {cache.name}, to recompute)")
    if keys is None:
        t = time.time()
        dia = CFG["dedupe"]["dihedral"]
        with ThreadPoolExecutor(CFG["workers"]) as ex:
            keys = list(ex.map(lambda p: _canon_key(p, dia), man["path"].tolist()))
        ok = np.array([k is not None for k in keys])
        print(f"hashed {ok.sum()}/{len(ok)} in {time.time()-t:.0f}s "
              f"(dihedral={dia}, {8 if dia else 1} variants/img)")
        man = man[ok].reset_index(drop=True)
        keys = [k for k, m in zip(keys, ok) if m]
        pd.DataFrame({"path": man["path"], "key": keys}).to_csv(cache, index=False)

    B = CFG["dedupe"]["bands"]
    nd = CFG["dedupe"]["near_dist"]

    # ---- the threshold sweep: measure the trade-off instead of guessing it ----
    # Merging harder drives cross-split leaks toward zero but grows clusters, which distorts
    # class ratios and folds more label noise together. Both sides are printed so the choice
    # is made on numbers. Previously this was a guess, and one guess already cost a full run.
    sweep = CFG["dedupe"].get("sweep") or [nd]
    sweep = [x for x in sweep if x >= 0]
    if len(sweep) > 1:
        print(f"\nnear_dist sweep  ({len(keys)} images, split {CFG['split']}, seed {SEED})")
        print(f"  {'nd':>3} | {'clusters':>8} | {'largest':>7} | {'multi-img':>9} | "
              f"{'x-class':>7} | {'x-src':>5} | {'leaks@7':>8} | {'worst dev':>9}")
        for n_ in sorted(sweep):
            c_, _m = clusters_at(keys, n_, B)
            m2 = man.copy()
            m2["cluster"] = c_
            sz = pd.Series(c_).value_counts()
            mm = sz[sz > 1]
            xc = int((m2.groupby("cluster")["class"].nunique() > 1).sum())
            xs_ = int((m2.groupby("cluster")["source"].nunique() > 1).sum())
            sp_, _g = grouped_split(m2, CFG["split"], SEED)
            lk = leak_scan(sp_, keys, tight=7, B=B)
            nlk = sum(len(v) for v in lk.values())
            tb = sp_.groupby(["class", "split"]).size().unstack(fill_value=0)
            wd = max(max(abs(tb.loc[c, s] - tb.loc[c].sum() * CFG["split"][s]) / tb.loc[c].sum()
                          for s in CFG["split"] if s in tb.columns) for c in tb.index)
            mark = "  <= chosen" if n_ == nd else ""
            print(f"  {n_:>3} | {len(sz):>8} | {int(sz.max()):>7} | {int(mm.sum()):>9} | "
                  f"{xc:>7} | {xs_:>5} | {nlk:>8} | {wd:>8.1%}{mark}")
        print("  leaks@7 = near-duplicate pairs straddling splits, detected at Hamming <= 7.")
        print("  Read the row you want and set CFG['dedupe']['near_dist'] to it.")

    man["cluster"], _merged = clusters_at(keys, nd, B)
    print(f"\nchosen near_dist {nd}: {_merged} key pairs merged within Hamming <= {nd} "
          f"(B={B}, complete for nd <= {B - 1})")
else:
    man["cluster"] = np.arange(len(man))

# ---- diagnostics that matter ----
sizes = man.groupby("cluster").size()
multi = sizes[sizes > 1]
xclass = man.groupby("cluster")["class"].nunique()
xsrc = man.groupby("cluster")["source"].nunique()
n_before = len(man)
print(f"\nclusters: {man['cluster'].nunique()}  |  multi-member: {len(multi)}  "
      f"|  images inside multi-member clusters: {int(multi.sum())} "
      f"({multi.sum()/len(man):.0%})")
print(f"images folded into an existing cluster: {n_before - man['cluster'].nunique()} "
      f"({(n_before - man['cluster'].nunique())/n_before:.0%} of the manifest)")
print(f"largest cluster: {int(sizes.max())} images  |  median {int(sizes.median())}")
print(f"clusters spanning >1 class: {int((xclass>1).sum())}"
      + ("   <- LABEL NOISE warning" if (xclass > 1).any() else ""))
print(f"clusters spanning >1 source: {int((xsrc>1).sum())}  "
      f"(cross-source copies = the datasets overlap)")

# A cluster of ~60 that is single-linkage chaining would show here. Largest is the number to
# watch: if it climbs into the hundreds, lower CFG["dedupe"]["near_dist"] to 0.
if sizes.max() > 10:
    big = man[man["cluster"] == sizes.idxmax()]
    comp = ", ".join(f"{k}={v}" for k, v in big["class"].value_counts().items())
    srcs = ", ".join(f"{k}={v}" for k, v in big["source"].value_counts().items())
    print(f"largest cluster composition: {comp}   (sources: {srcs})")

# WHERE the label noise comes from. This is also the empirical check on CFG["source_alias"]:
# if indo3's `blight` folder were really Leaf Blast, its images would land in clusters with
# Leaf_Blast images and that pair would show up here. Absence is not proof, but a cluster full
# of indo3:blight <-> <something else> would be a red flag worth acting on.
bad = xclass[xclass > 1].index
xs = man[man["cluster"].isin(bad)]
if len(xs):
    pairs = defaultdict(int)
    for _cid, grp in xs.groupby("cluster"):
        # NB: NOT itertuples — a column literally named "class" is not a valid Python
        # identifier, so pandas renames it to "_1" and tuple indexing raises TypeError.
        # That would land here, after 84 s of hashing. String concat sidesteps it entirely.
        combos = sorted(set(grp["source"].astype(str) + ":" + grp["class"].astype(str)))
        for a in range(len(combos)):
            for b in range(a + 1, len(combos)):
                pairs[(combos[a], combos[b])] += 1
    print(f"\nimages inside cross-class clusters: {len(xs)} ({len(xs)/len(man):.1%}) "
          f"across {len(bad)} clusters")
    print("which source:class pairs collide:")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]:
        flag = ""
        if "indo3" in a and "indo3" in b:
            flag = "   <- both from indo3: within-source label disagreement"
        elif ("indo3" in a) != ("indo3" in b):
            flag = "   <- involves indo3 (check the source_alias mapping)"
        print(f"   {n:4d} {a}  <->  {b}{flag}")
man.to_csv(OUT / "manifest_grouped.csv", index=False)
