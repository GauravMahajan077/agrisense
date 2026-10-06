"""Phase 4 unit tests for the field-test loader (agri/new/field_test.py).

Imports the SHIPPED loader directly (it is the artifact, not a copy) and tests the
preprocessing contract, photo discovery, metrics, abstain, bootstrap, and an end-to-end run
on a tiny locally-built TFLite model.

Run from agri/:  python -B test_phase4_field_test.py
"""
import json
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

import field_test

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ---- preprocess: the Cell 18 contract ----
print("=== preprocess (contract) ===")
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    img = Image.new("RGB", (400, 300), (10, 200, 30))
    p = td / "leaf.jpg"
    img.save(p)
    x = field_test.preprocess(str(p), 256)
    check("shape (256,256,3)", x.shape == (256, 256, 3), str(x.shape))
    check("dtype float32", x.dtype == np.float32, str(x.dtype))
    check("value range 0-255 (no /255)", x.min() >= 0 and x.max() <= 255, f"max {x.max()}")
    check("pixel values preserved", x[0, 0].tolist() == [10, 200, 30], str(x[0, 0]))
    img2 = Image.new("RGB", (256, 256), (1, 2, 3))
    p2 = td / "leaf2.png"   # PNG: lossless, so the pixel survives the round-trip
    img2.save(p2)
    x2 = field_test.preprocess(str(p2), 256)
    check("256x256 input not resized", x2[0, 0].tolist() == [1, 2, 3])

# ---- discover_photos ----
print("=== discover_photos ===")
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    (td / "A").mkdir(); (td / "B").mkdir(); (td / "unknown").mkdir()
    (td / "A" / "1.jpg").write_bytes(b"x")
    (td / "A" / "2.png").write_bytes(b"x")
    (td / "B" / "1.jpg").write_bytes(b"x")
    (td / "unknown" / "1.jpg").write_bytes(b"x")
    rows = field_test.discover_photos(str(td), classes=["A", "B"])
    check("subfolder discovery", len(rows) == 3, str(len(rows)))
    check("unknown class filtered", all(c in ("A", "B") for _, c in rows))
    csv_path = td / "labels.csv"
    csv_path.write_text("path,class\nA/1.jpg,A\nB/1.jpg,B\n", encoding="utf-8")
    rows2 = field_test.discover_photos(str(td), labels_csv=str(csv_path), classes=["A", "B"])
    check("labels CSV discovery", len(rows2) == 2, str(len(rows2)))
    rows3 = field_test.discover_photos(str(td), classes=["A", "B"], limit=1)
    check("limit caps per class", len(rows3) == 2, str(len(rows3)))

# ---- metrics ----
print("=== metrics ===")
cm = np.array([[10, 2], [3, 5]], np.int64)
m = field_test.metrics(cm)
check("macro-F1 = 11/15", abs(m["macro_f1"] - 11 / 15) < 1e-9, f"{m['macro_f1']:.4f}")
check("accuracy = 0.75", abs(m["accuracy"] - 0.75) < 1e-9, f"{m['accuracy']:.4f}")
check("worst recall = 5/8", abs(m["worst_recall"] - 5 / 8) < 1e-9, f"{m['worst_recall']:.4f}")

# ---- abstain ----
print("=== abstain ===")
y = np.array([0, 0, 1, 1], np.int32)
p = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.49, 0.48]], np.float32)
ab = field_test.abstain_report(y, p, 2, 0.5)
check("coverage 0.75", abs(ab["coverage"] - 0.75) < 1e-9, f"{ab['coverage']:.2f}")
check("acc on covered 1.0", abs(ab["accuracy"] - 1.0) < 1e-9, f"{ab['accuracy']:.2f}")

# ---- bootstrap ----
print("=== bootstrap ===")
lo, hi = field_test.bootstrap_macro_f1(y, p.argmax(1), 2, iters=200)
check("CI ordered", lo <= hi, f"[{lo:.3f}, {hi:.3f}]")

# ---- end-to-end on a tiny TFLite model ----
print("=== end-to-end (tiny TFLite) ===")
try:
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input((256, 256, 3)),
        tf.keras.layers.Conv2D(2, 3, padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "model.tflite").write_bytes(tfl)
        (td / "class_names.json").write_text(json.dumps({
            "classes": ["A", "B"], "graph_input_size": 224, "recommended_input_size": 256,
            "value_range": [0, 255], "dtype": "float32", "abstain_threshold": 0.5,
        }), encoding="utf-8")
        (td / "photos" / "A").mkdir(parents=True)
        (td / "photos" / "B").mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (300, 300), (i * 100, 50, 200)).save(td / "photos" / "A" / f"{i}.jpg")
            Image.new("RGB", (300, 300), (200, i * 100, 50)).save(td / "photos" / "B" / f"{i}.jpg")
        classes, size, abstain = field_test.load_contract(str(td / "class_names.json"))
        predict, (h, w) = field_test.load_model(str(td / "model.tflite"))
        photos = field_test.discover_photos(str(td / "photos"), classes=classes)
        summary = field_test.run_eval(photos, predict, classes, size, abstain, str(td / "out"))
        check("e2e: 4 images evaluated", summary["n"] == 4, str(summary["n"]))
        check("e2e: predictions.csv written", (td / "out" / "predictions.csv").exists())
        check("e2e: confusion.csv written", (td / "out" / "confusion.csv").exists())
        check("e2e: macro-F1 is a float", isinstance(summary["macro_f1"], float))
except ImportError:
    print("  SKIP end-to-end (tensorflow not installed)")

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)