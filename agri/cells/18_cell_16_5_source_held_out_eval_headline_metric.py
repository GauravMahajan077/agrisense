# CELL 16.5 — source-held-out eval (HEADLINE METRIC)
# The number that matters: train on anshul6+indo3, test on dedeikh across the 5 shared classes
# (Sheath_Blight is single-source in anshul6, so it stays in training but is excluded here).
# In-source val/test (Cells 15-16) is secondary and expected to be much higher.
if man_held is not None and len(man_held):
    shared = sorted(set(CLASSES) & set(man_held["class"].unique()))
    print(f"\n=== SOURCE-HELD-OUT EVAL ===  held-out source: {CFG['held_out_source']}")
    print(f"shared classes: {shared} ({len(shared)} of {len(CLASSES)})")
    if len(shared) < 2:
        print("  <2 shared classes — held-out eval undefined, skipping")
    else:
        # Preprocess exactly like the field stress test (Cell 17): 0-255 float, resize to SIZE.
        # The parity check (Cell 19) proves the deployed graph agrees with `model` at these sizes.
        keep = np.array([c in shared for c in man_held["class"]])
        mh = man_held[keep].reset_index(drop=True)
        t = time.time()
        Xh = np.stack([np.asarray(Image.open(p).convert("RGB")
                                   .resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
                       for p in mh["path"]])
        print(f"preprocessed {len(Xh)} held-out images in {time.time()-t:.0f}s")
        Ph = model.predict(Xh, verbose=0)                       # (n, NC)
        yh = np.array([CLASSES.index(c) for c in mh["class"]], np.int32)
        shared_idx = {c: i for i, c in enumerate(shared)}
        yh_c = np.array([shared_idx[CLASSES[y]] for y in yh], np.int32)
        Ph_c = Ph[:, [CLASSES.index(c) for c in shared]]
        pred_c = Ph_c.argmax(1)
        cm = confusion(yh_c, pred_c, len(shared))
        f1_ho, rec_ho = report(cm, f"SOURCE-HELD-OUT ({CFG['held_out_source']}, "
                               f"{len(shared)} classes)", classes=shared)

        # bootstrap 95% CI on macro-F1 (resample images with replacement)
        def boot_macro_f1(y, p, k, iters=CFG["bootstrap_iters"], seed=0):
            rng = np.random.RandomState(seed)
            n = len(y); scores = np.empty(iters)
            for it in range(iters):
                idx = rng.randint(0, n, n)
                cmb = confusion(y[idx], p[idx], k)
                tp = np.diag(cmb).astype(float)
                prec = np.divide(tp, cmb.sum(0), out=np.zeros_like(tp), where=cmb.sum(0) > 0)
                recb = np.divide(tp, cmb.sum(1), out=np.zeros_like(tp), where=cmb.sum(1) > 0)
                f1b = np.divide(2*prec*recb, prec+recb, out=np.zeros_like(tp), where=(prec+recb) > 0)
                scores[it] = np.nanmean(f1b)
            return np.percentile(scores, [2.5, 97.5])
        lo, hi = boot_macro_f1(yh_c, pred_c, len(shared))
        print(f"macro-F1 bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")

        # abstain rule: skip predictions below the confidence threshold
        thr = CFG["abstain_threshold"]
        conf = Ph_c.max(1)
        abstain = conf < thr
        cov = (~abstain).mean()
        if cov > 0:
            cm_a = confusion(yh_c[~abstain], pred_c[~abstain], len(shared))
            f1a, _ = report(cm_a, f"HELD-OUT with abstain@{thr}", classes=shared)
            acc_a = np.trace(cm_a) / max(cm_a.sum(), 1)
        else:
            f1a, acc_a = np.nan, 0.0
        print(f"abstain@{thr}: coverage {cov:.1%} | acc on covered {acc_a:.4f} | "
              f"macro-F1 on covered {np.nanmean(f1a):.4f}")

        # per-source macro-F1 (general: works if the held-out set spans several sources)
        print("\nper-source macro-F1 (held-out set):")
        for src, grp in mh.groupby("source"):
            idx = np.nonzero(mh["source"].to_numpy() == src)[0]
            cm_s = confusion(yh_c[idx], pred_c[idx], len(shared))
            tp = np.diag(cm_s).astype(float)
            prec = np.divide(tp, cm_s.sum(0), out=np.zeros_like(tp), where=cm_s.sum(0) > 0)
            recb = np.divide(tp, cm_s.sum(1), out=np.zeros_like(tp), where=cm_s.sum(1) > 0)
            f1s = np.divide(2*prec*recb, prec+recb, out=np.zeros_like(tp), where=(prec+recb) > 0)
            print(f"  {src:12s} n={len(idx):5d}  macro-F1 {np.nanmean(f1s):.4f}")

        # in-source test per-source macro-F1 (uses yte_pred/yte from Cell 16)
        print("\nper-source macro-F1 (in-source TEST):")
        for src, grp in te.groupby("source"):
            idx = np.nonzero(te["source"].to_numpy() == src)[0]
            cm_s = confusion(yte[idx], yte_pred[idx], NC)
            tp = np.diag(cm_s).astype(float)
            prec = np.divide(tp, cm_s.sum(0), out=np.zeros_like(tp), where=cm_s.sum(0) > 0)
            recb = np.divide(tp, cm_s.sum(1), out=np.zeros_like(tp), where=cm_s.sum(1) > 0)
            f1s = np.divide(2*prec*recb, prec+recb, out=np.zeros_like(tp), where=(prec+recb) > 0)
            print(f"  {src:12s} n={len(idx):5d}  macro-F1 {np.nanmean(f1s):.4f}")

        pd.DataFrame({"class": shared, "f1": f1_ho.round(4),
                      "recall": rec_ho.round(4)}).to_csv(OUT / "source_held_out.csv", index=False)
else:
    print("source-held-out eval SKIPPED (CFG['held_out_source'] is None or empty)")
