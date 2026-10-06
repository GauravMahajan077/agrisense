# CELL 7 — enforce min_class on TRAIN, re-splitting if a class is starved
# Must run AFTER the split: a class can clear the raw-count bar and still vanish from a split.
def starved_classes(m, classes, min_class):
    tr = m[m["split"] == "train"]["class"].value_counts()
    bad = []
    for c in classes:
        n = int(tr.get(c, 0))
        missing = [k for k in sp if c not in set(m.loc[m["split"] == k, "class"])]
        if n < min_class or missing:
            bad.append((c, n, missing))
    return bad

CLASSES = list(CFG["classes"])
for attempt in range(4):
    keep = [c for c in CLASSES if c in set(man["class"])]
    gone = [c for c in CLASSES if c not in keep]
    if gone: print(f"dropped (no images at all): {gone}")
    man = man[man["class"].isin(keep)]
    CLASSES, CIDX, NC = keep, {c: i for i, c in enumerate(CLASSES)}, len(keep)
    man, _ = grouped_split(man, sp, SEED)
    bad = starved_classes(man, CLASSES, CFG["min_class"])
    if not bad:
        print(f"\nmin_class OK: every class has >= {CFG['min_class']} TRAIN images "
              f"and appears in all {len(sp)} splits")
        break
    print(f"\nattempt {attempt+1}: starving -> " + ", ".join(
        f"{c} (train={n}{', missing from ' + ','.join(ms) if ms else ''})" for c, n, ms in bad))
    drop = {c for c, _, _ in bad}
    CLASSES = [c for c in CLASSES if c not in drop]
else:
    raise SystemExit("could not satisfy min_class — lower it or add a data source")

tr, va, te = (man[man["split"] == k].reset_index(drop=True) for k in ("train", "val", "test"))
print(f"\nfinal: train {len(tr)} | val {len(va)} | test {len(te)}")
print(tr.groupby("class").size().reindex(CLASSES).to_frame("train_n").to_string())
man.to_csv(OUT / "manifest_split.csv", index=False)
