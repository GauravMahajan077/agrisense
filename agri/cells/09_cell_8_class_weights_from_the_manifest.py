# CELL 8 — class weights from the manifest
# NB: never touch a prefetched dataset for this — that is what raised
# `'_PrefetchDataset' object has no attribute 'class_names'`.
cnt = tr["class"].value_counts().reindex(CLASSES).to_numpy(float)
N = cnt.sum()
mode = CFG["imbalance"]["mode"]
assert mode == "class_weight", f"only class_weight is supported (got {mode!r})"

w = N / (NC * cnt)
w = w / w.mean()
CLS_W = tf.constant(w, dtype=tf.float32)
print(f"imbalance mode: {mode}  |  majority/minority {cnt.max()/max(cnt[cnt>0].min(),1):.1f}x")
print(pd.DataFrame({"class": CLASSES, "train_n": cnt.astype(int), "weight": w.round(3)}).to_string())

LOSS = lambda: tf.keras.losses.CategoricalCrossentropy()
