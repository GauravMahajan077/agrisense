# CELL 16 — test set, touched once
yte_pred = predict_idx(test_ds, len(yte))
f1_te, rec_te = report(confusion(yte, yte_pred, NC), "TEST")
gap = np.nanmean(f1_va) - np.nanmean(f1_te)
print(f"\nGAP (val - test macro-F1) = {gap:+.4f}")
print("  > +0.05 : split still leaky or too small to trust")
print("  <  0.00 : val was pessimistic, test is the better number")
pd.DataFrame(confusion(yte, yte_pred, NC), index=CLASSES,
             columns=CLASSES).to_csv(OUT / "test_confusion.csv")
