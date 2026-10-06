# CELL 15 — validation report
def predict_idx(ds, n):
    out = []
    for b in ds:
        out.append(np.asarray(model.predict(b, verbose=0)).argmax(1))
    return np.concatenate(out) if out else np.zeros(0, int)

def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred): cm[int(t), int(p)] += 1
    return cm

def report(cm, title):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec  = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(tp), where=(prec+rec) > 0)
    df = pd.DataFrame({"class": CLASSES, "support": cm.sum(1), "precision": prec.round(3),
                       "recall": rec.round(3), "f1": f1.round(3)}).sort_values("recall")
    print(f"\n=== {title} ===\n{df.to_string(index=False)}")
    print(f"MACRO-F1 {np.nanmean(f1):.4f} | ACC {np.trace(cm)/max(cm.sum(),1):.4f} | "
          f"WORST-CLASS RECALL {rec.min():.4f}")
    fig, ax = plt.subplots(figsize=(0.62*NC+3, 0.62*NC+2.5))
    im = ax.imshow(cm/np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NC), CLASSES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(NC), CLASSES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} (row-normalised)")
    for i in range(NC):
        for j in range(NC):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i,j]/max(cm.sum(1)[i],1) > .5 else "black")
    plt.colorbar(im); plt.tight_layout(); plt.show()
    return f1, rec

yva_pred = predict_idx(val_ds, len(yva))
f1_va, rec_va = report(confusion(yva, yva_pred, NC), "VAL")
pd.DataFrame({"class": CLASSES, "support": np.bincount(yva, minlength=NC),
              "recall": rec_va.round(3)}).to_csv(OUT / "val_report.csv", index=False)
