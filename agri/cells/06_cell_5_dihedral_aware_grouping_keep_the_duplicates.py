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

def _d4_keys(p, probe=256):
    # All 8 D4-variant pHashes, not the min: the brute-force merge below takes the MIN over
    # the 8x8 variant pairs, which is strictly more information than a single canonical key.
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if probe:
                # draft() lets libjpeg do a SCALED decode — this is what makes preloading the
                # 2.4 MB shayanriyaz files tolerable at all.
                try: im.draft("RGB", (probe, probe))
                except Exception: pass
                im = im.resize((probe, probe), Image.BILINEAR)
            return tuple(int(str(imagehash.phash(v)), 16) for v in _dihedral(im))
    except Exception:
        return None

def d4_dist_matrix(keys8, chunk=256):
    """Full pairwise min-over-D4-variants Hamming matrix, (n, n) uint8.

    keys8: (n, 8) uint64. D[i, j] = min over a, b of popcount(keys8[i,a] ^ keys8[j,b]).
    Chunked so the working set is (chunk, n) uint8, not (n, n) at once.
    """
    n = len(keys8)
    D = np.zeros((n, n), np.uint8)
    for q0 in range(0, n, chunk):
        q = keys8[q0:q0 + chunk]
        best = np.full((len(q), n), 255, np.uint8)
        for a in range(8):
            qa = q[:, a][:, None]
            for b in range(8):
                d = np.bitwise_count(np.bitwise_xor(qa, keys8[:, b][None, :]))
                np.minimum(best, d, out=best)
        D[q0:q0 + chunk] = best
    return D

def brute_clusters(D, nd, labels):
    """Union-find over SAME-CLASS pairs with D <= nd. Returns (cluster_ids, n_merged).

    Cross-class near-duplicates are NOT merged: they are label noise, and merging them would
    fold contradictory labels into one cluster. They are flagged separately by cross_class_mask.
    """
    n = len(D)
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    merged = 0
    for i in range(n):
        js = np.nonzero((D[i] <= nd) & (labels == labels[i]))[0]
        for j in js:
            if j <= i:
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri; merged += 1
    out = np.array([find(i) for i in range(n)], np.int64)
    return out, merged

def cross_class_mask(D, nd, labels):
    """True for images within nd of a DIFFERENT-class image (ambiguous / mislabelled)."""
    n = len(D)
    mask = np.zeros(n, bool)
    for i in range(n):
        mask[i] = bool(((D[i] <= nd) & (labels != labels[i])).any())
    return mask

def grouped_split(man, sp, seed):
    """Stratified group split. Defined here, not in Cell 6, so Cell 5 and Cell 6 share one
    implementation (a second copy is how the two halves drift apart).

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

def leak_scan(D, man, tight=7):
    """Count near-duplicate pairs that ended up in DIFFERENT splits.

    Uses the brute-force D4 distance matrix from Cell 5 (no re-hashing). A pair within `tight`
    bits that straddles splits is a leak the merge missed.
    """
    sp_of = man["split"].to_numpy()
    leaks = defaultdict(list)
    for i in range(len(man)):
        js = np.nonzero((D[i] <= tight) & (sp_of != sp_of[i]))[0]
        for j in js:
            if j <= i:
                continue
            leaks[tuple(sorted((sp_of[i], sp_of[j])))].append((i, j))
    return leaks

if CFG["dedupe"]["enable"]:
    # Hash all 8 D4 variants fresh every run. The old on-disk cache was removed on purpose:
    # it could silently serve stale keys after a manifest change, and hashing is only ~84 s
    # for the full set (far less in smoke).
    t = time.time()
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        keys8 = list(ex.map(_d4_keys, man["path"].tolist()))
    ok = np.array([k is not None for k in keys8])
    print(f"hashed {ok.sum()}/{len(ok)} in {time.time()-t:.0f}s (8 D4 variants/img)")
    man = man[ok].reset_index(drop=True)
    keys8 = np.array([k for k, m in zip(keys8, ok) if m], np.uint64)

    nd = CFG["dedupe"]["near_dist"]
    labels = man["class"].astype("category").cat.codes.to_numpy(np.int32)
    t = time.time()
    D = d4_dist_matrix(keys8)
    man["cluster"], _merged = brute_clusters(D, nd, labels)
    cross_mask = cross_class_mask(D, nd, labels)
    print(f"\nbrute-force D4 dedupe in {time.time()-t:.0f}s: {_merged} same-class key pairs merged "
          f"within Hamming <= {nd} (min over 8 D4 variants)")
    print(f"cross-class near-duplicates: {int(cross_mask.sum())} images "
          f"({cross_mask.mean():.1%}) — excluded from val/test in Cell 6")
else:
    man["cluster"] = np.arange(len(man))
    D = None
    cross_mask = None

# ---- diagnostics that matter ----
sizes = man.groupby("cluster").size()
multi = sizes[sizes > 1]
xsrc = man.groupby("cluster")["source"].nunique()
n_before = len(man)
print(f"\nclusters: {man['cluster'].nunique()}  |  multi-member: {len(multi)}  "
      f"|  images inside multi-member clusters: {int(multi.sum())} "
      f"({multi.sum()/len(man):.0%})")
print(f"images folded into an existing cluster: {n_before - man['cluster'].nunique()} "
      f"({(n_before - man['cluster'].nunique())/n_before:.0%} of the manifest)")
print(f"largest cluster: {int(sizes.max())} images  |  median {int(sizes.median())}")
# Same-class merge means clusters can never span classes; cross-class collisions live in
# cross_mask instead (reported below).
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
# if indo3's `blight` folder were really Leaf Blast, its images would be near-duplicates of
# Leaf_Blast images and that pair would show up here. Absence is not proof, but a pile of
# indo3:blight <-> <something else> pairs would be a red flag worth acting on.
if cross_mask is not None and cross_mask.any():
    pairs = defaultdict(int)
    for i in np.nonzero(cross_mask)[0]:
        js = np.nonzero((D[i] <= nd) & (labels != labels[i]))[0]
        for j in js:
            if j <= i:
                continue
            a = man["source"].iat[i] + ":" + man["class"].iat[i]
            b = man["source"].iat[j] + ":" + man["class"].iat[j]
            pairs[tuple(sorted((a, b)))] += 1
    print(f"\ncross-class near-duplicates: {int(cross_mask.sum())} images "
          f"({cross_mask.mean():.1%}) — excluded from val/test in Cell 6")
    print("which source:class pairs collide:")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]:
        flag = ""
        if "indo3" in a and "indo3" in b:
            flag = "   <- both from indo3: within-source label disagreement"
        elif ("indo3" in a) != ("indo3" in b):
            flag = "   <- involves indo3 (check the source_alias mapping)"
        print(f"   {n:4d} {a}  <->  {b}{flag}")
man.to_csv(OUT / "manifest_grouped.csv", index=False)
