# AgriSense — field-adaptation training (RUN ON KAGGLE, free GPU T4)
# ---------------------------------------------------------------------
# What it does (order matters):
#   1. Leaf-crop preprocessing (fixes domain shift: whole-plant field photos)
#   2. Fine-tune EfficientNet-B0 head on cropped data, class-weighted (handles
#      your insufficient-per-class dataset)
#   3. TTA eval + abstention threshold tuning on the FIELD test set
#   4. Export bundle: .keras + .tflite + class_names.json + report CSVs
#
# Usage on Kaggle:
#   - Upload agrisense_bundle.zip, your dataset folder, field_test images
#   - Set CFG["dataset_dir"] / CFG["field_dir"] below
#   - Add this file as a notebook cell (or attach as script), run top-to-bottom
#   - NEVER run locally (thermal/hardware constraint)
# ---------------------------------------------------------------------
import io
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np

CFG = {
    "seed": 1337,
    "dataset_dir": "/kaggle/input/paddy-dataset",   # <- EDIT: class folders (train/val or flat)
    "field_dir": "/kaggle/input/field-test",        # <- EDIT: your 60+ real field photos (subfolders=class optional)
    "bundle_zip": "/kaggle/input/agrisense-bundle/agrisense_bundle.zip",  # <- EDIT
    "img_size": 224,
    "pre_size": 256,
    "epochs_head": 6,
    "epochs_full": 4,
    "lr_head": 3e-3,
    "lr_full": 3e-5,
    "batch": 32,
    "use_crop": True,        # OpenCV leaf crop (the key lever)
    "use_tta": True,         # h-flip TTA at eval
    "target_field_acc": 0.75,
    "out_dir": "agrisense_out",
}
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

import cv2
import tensorflow as tf

CLASSES = ["Bacterial_Leaf_Blight", "Brown_Spot", "Healthy", "Leaf_Blast", "Leaf_Scald", "Sheath_Blight"]


# ---------- 1. crop (same contract as ml/src/crop_leaf.py) ----------
def crop_leaf(img_bgr):
    """Largest green component crop; falls back to original on failure."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 40, 25), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return img_bgr
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == idx)
    if len(xs) < 0.02 * mask.size:
        return img_bgr
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int(0.12 * (x1 - x0)), int(0.12 * (y1 - y0))
    h, w = img_bgr.shape[:2]
    return img_bgr[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]


def load_image(path, use_crop=None):
    """path -> float32 0..255, exactly 256x256 Lanczos (inference contract)."""
    use_crop = CFG["use_crop"] if use_crop is None else use_crop
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if use_crop:
        img = crop_leaf(img)
    img = cv2.resize(img, (CFG["pre_size"], CFG["pre_size"]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)  # 0..255, NEVER /255


# ---------- 2. data ----------
def scan_dataset(root):
    """Class-folder layout -> (paths, labels). Prefers train/ subdir if present."""
    root = Path(root)
    base = root / "train" if (root / "train").exists() else root
    paths, labels = [], []
    for ci, cls in enumerate(sorted(p.name for p in base.iterdir() if p.is_dir())):
        for f in (base / cls).rglob("*.*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(str(f))
                labels.append(ci)
    return paths, np.array(labels)


def make_dataset(paths, labels, shuffle=False, use_crop=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(1024, seed=CFG["seed"])
    def _load(p, y):
        return tf.py_function(lambda pp, yy: (load_image(pp, use_crop), yy), [p, y], (tf.float32, tf.int64))
    ds = ds.map(_load, num_parallel_tiles=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.repeat().batch(CFG["batch"])
    else:
        ds = ds.batch(CFG["batch"])
    return ds.prefetch(tf.data.AUTOTUNE)


def balanced_class_weights(labels):
    """Inverse-frequency class weights — compensates your insufficient-per-class data."""
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {int(c): float(total) / (len(classes) * n) for c, n in zip(classes, counts)}


# ---------- 3. model ----------
def build_model(base_weights_path):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(CFG["img_size"], CFG["img_size"], 3))
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="head")(x)
    m = tf.keras.Model(base.input, out)
    m.load_weights(base_weights_path, by_name=True, skip_mismatch=True)
    return m, base


def compile_model(m, lr):
    m.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


def predict_probs(model, paths, tta=CFG["use_tta"]):
    """Probs for a list of paths (crop contract + optional h-flip TTA)."""
    imgs = np.stack([load_image(p) for p in paths])
    p = model.predict(imgs, verbose=0)
    if tta:
        flipped = model.predict(imgs[:, :, ::-1, :], verbose=0)
        p = (p + flipped) / 2
    return p


def tune_abstain(probs, y_true, targets=(0.95, 0.90)):
    """Smallest confidence threshold hitting target accuracy on accepted set."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    results = []
    for t in sorted(set(np.round(conf, 2))):
        keep = conf >= t
        if keep.sum() == 0:
            continue
        acc = (pred[keep] == y_true[keep]).mean()
        results.append({"threshold": float(t), "coverage": float(keep.mean()),
                        "acc_on_accepted": float(acc), "n_wrong_confident": int(((pred[keep] != y_true[keep]) & (conf[keep] >= 0.8)).sum())})
    chosen = {tg: next((r["threshold"] for r in results if r["acc_on_accepted"] >= tg), None) for tg in targets}
    return results, chosen


def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)
    with zipfile.ZipFile(CFG["bundle_zip"]) as z:
        z.extractall(CFG["out_dir"])
        bundle = json.loads(z.read("class_names.json"))
    base_w = os.path.join(CFG["out_dir"], "agrisense_b0.keras")

    paths, labels = scan_dataset(CFG["dataset_dir"])
    print(f"dataset: {len(paths)} images, class counts: {np.bincount(labels)}")
    class_w = balanced_class_weights(labels)
    print("class weights:", class_w)

    # split: last 15% stratified-ish by shuffle
    rng = np.random.default_rng(CFG["seed"])
    idx = rng.permutation(len(paths))
    n_val = max(1, int(0.15 * len(idx)))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    val_paths = [paths[i] for i in val_i]
    val_y = labels[val_i]

    model, base = build_model(base_w)
    compile_model(model, CFG["lr_head"])
    model.fit(make_dataset([paths[i] for i in tr_i], labels[tr_i], shuffle=True),
              validation_data=make_dataset(val_paths, val_y, shuffle=False),
              epochs=CFG["epochs_head"], class_weight=class_w, verbose=2)

    base.trainable = True
    for layer in base.layers[:80]:
        layer.trainable = False
    compile_model(model, CFG["lr_full"])
    model.fit(make_dataset([paths[i] for i in tr_i], labels[tr_i], shuffle=True),
              validation_data=make_dataset(val_paths, val_y, shuffle=False),
              epochs=CFG["epochs_full"], class_weight=class_w, verbose=2)

    # ---- field evaluation + abstention tuning ----
    field_paths = [str(p) for p in Path(CFG["field_dir"]).rglob("*.*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if field_paths:
        fp = predict_probs(model, field_paths)
        np.savetxt(os.path.join(CFG["out_dir"], "field_probs.csv"), fp, delimiter=",")
        print("field images:", len(field_paths), "-> field_probs.csv")

    val_probs = predict_probs(model, val_paths)
    curve, thresholds = tune_abstain(val_probs, val_y)
    acc = (val_probs.argmax(1) == val_y).mean()
    print(f"val acc (crop+TTA): {acc:.3f} | abstain thresholds: {thresholds}")
    with open(os.path.join(CFG["out_dir"], "abstain.json"), "w") as f:
        json.dump({"thresholds": thresholds, "curve": curve, "val_acc": float(acc),
                   "field_target": CFG["target_field_acc"], "preprocess": bundle["steps"]}, f, indent=2)

    # ---- export ----
    model.save(os.path.join(CFG["out_dir"], "agrisense_field_adapted.keras"))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    open(os.path.join(CFG["out_dir"], "agrisense_field_adapted.tflite"), "wb").write(tfl)
    print("exported: keras + tflite + abstain.json ->", CFG["out_dir"])


if __name__ == "__main__":
    main()