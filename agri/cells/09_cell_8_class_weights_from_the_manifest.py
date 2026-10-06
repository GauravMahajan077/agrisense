# CELL 8 — class weights from the manifest
# NB: never touch a prefetched dataset for this — that is what raised
# `'_PrefetchDataset' object has no attribute 'class_names'`.
cnt = tr["class"].value_counts().reindex(CLASSES).to_numpy(float)
N = cnt.sum()
mode = CFG["imbalance"]["mode"]

if mode == "class_weight":
    w = N / (NC * cnt)
elif mode == "effective":
    b = CFG["imbalance"]["effective_beta"]; w = (1 - b) / (1 - b ** cnt)
else:
    w = np.ones(NC)
w = w / w.mean()
CLS_W = tf.constant(w, dtype=tf.float32)
print(f"imbalance mode: {mode}  |  majority/minority {cnt.max()/max(cnt[cnt>0].min(),1):.1f}x")
print(pd.DataFrame({"class": CLASSES, "train_n": cnt.astype(int), "weight": w.round(3)}).to_string())

LOSS = {
    "class_weight": lambda: tf.keras.losses.CategoricalCrossentropy(),
    "effective":    lambda: tf.keras.losses.CategoricalCrossentropy(),
    "oversample":   lambda: tf.keras.losses.CategoricalCrossentropy(),
    "none":         lambda: tf.keras.losses.CategoricalCrossentropy(),
}[mode]
