"""field_test.py — evaluate the deployed Agrisense model on real field photos (eval only).

Loads the exported bundle (agrisense_bundle.zip) or a folder with model.tflite +
class_names.json, and runs the model on a folder of field photos. Applies the exact
preprocessing contract shipped in class_names.json (Cell 18 of the training notebook):
decode RGB, resize to 256x256 with a high-quality filter, emit (1,256,256,3) float32 in
0..255 — never divide by 255.

Usage:
    python field_test.py --bundle agrisense_bundle.zip --photos /path/to/photos
    python field_test.py --model model.tflite --classes class_names.json --photos /path/to/photos
    python field_test.py --bundle agrisense_bundle.zip --photos /path/to/photos --labels labels.csv

Photos layout (default): one subfolder per class, class/*.jpg — the same layout the crawler
(Cell 17) writes. With --labels, a CSV with columns path,class for flat folders.

Outputs: console report + predictions.csv + confusion.csv in --out (default: the photos dir).
"""
import argparse
import csv
import json
import pathlib
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_contract(classes_path):
    """Read the preprocessing contract shipped with the model."""
    with open(classes_path, encoding="utf-8") as f:
        c = json.load(f)
    classes = list(c["classes"])
    size = int(c.get("recommended_input_size", c.get("graph_input_size", 256)))
    abstain = float(c.get("abstain_threshold", 0.5))
    return classes, size, abstain


def load_model(model_path):
    """Load a TFLite model; return (predict, (h, w)).

    predict(img) takes one (h, w, 3) float32 image in 0..255 and returns (K,) float32 probs.
    """
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])
    if inp["dtype"] != np.float32:
        print(f"WARNING: model input dtype is {inp['dtype']} (expected float32) — the shipped "
              f"bundle is float32; a quantised model needs different preprocessing")
    def predict(img):
        interp.set_tensor(inp["index"], img[None].astype(inp["dtype"]))
        interp.invoke()
        return interp.get_tensor(out["index"])[0].astype(np.float32)
    return predict, (h, w)


def discover_photos(photos_dir, labels_csv=None, classes=None, limit=None):
    """Return [(path, class), ...]. Subfolder-per-class by default; --labels CSV overrides."""
    photos_dir = pathlib.Path(photos_dir)
    rows = []
    if labels_csv:
        with open(labels_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((str(photos_dir / r["path"]), r["class"]))
        if limit:
            seen, kept = {}, []
            for p, c in rows:
                if seen.get(c, 0) < limit:
                    kept.append((p, c)); seen[c] = seen.get(c, 0) + 1
            rows = kept
    else:
        for d in sorted(p for p in photos_dir.iterdir() if p.is_dir()):
            cls = d.name
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            if limit:
                files = files[:limit]
            for p in files:
                rows.append((str(p), cls))
    if classes is not None:
        known = set(classes)
        unknown = sorted({c for _, c in rows if c not in known})
        if unknown:
            print(f"WARNING: {len(unknown)} class label(s) not in the model: {unknown} — skipped")
        rows = [(p, c) for p, c in rows if c in known]
    return rows


def preprocess(path, size):
    """Decode RGB, LANCZOS-resize to (size, size), return float32 0-255 (size, size, 3)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32)


def confusion(y, p, k):
    cm = np.zeros((k, k), np.int64)
    for t, pr in zip(y, p):
        cm[t, pr] += 1
    return cm


def metrics(cm):
    """Per-class precision/recall/F1 + macro-F1 + accuracy + worst recall from a confusion matrix."""
    k = len(cm)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "macro_f1": float(np.nanmean(f1)),
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "worst_recall": float(rec.min()) if k else 0.0,
    }


def bootstrap_macro_f1(y, p, k, iters=2000, seed=0):
    """Bootstrap 95% CI on macro-F1 (resample images with replacement)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    scores = np.empty(iters)
    for it in range(iters):
        idx = rng.randint(0, n, n)
        scores[it] = metrics(confusion(y[idx], p[idx], k))["macro_f1"]
    return np.percentile(scores, [2.5, 97.5])


def abstain_report(y, p, k, threshold):
    """Coverage + accuracy + macro-F1 on predictions at/above the confidence threshold."""
    conf = p.max(1)
    covered = conf >= threshold
    cov = float(covered.mean())
    if cov > 0:
        m = metrics(confusion(y[covered], p[covered].argmax(1), k))
        return {"coverage": cov, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    return {"coverage": cov, "accuracy": float("nan"), "macro_f1": float("nan")}


def run_eval(photos, predict, classes, size, abstain_threshold, out_dir):
    """Run inference on photos, compute metrics, write CSVs. Returns a summary dict."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, preds, confs, margins, paths, probs = [], [], [], [], [], []
    for path, cls in photos:
        try:
            x = preprocess(path, size)
        except Exception as e:
            print(f"  skip {path}: {type(e).__name__}: {str(e)[:60]}")
            continue
        pr = predict(x)
        y.append(classes.index(cls))
        preds.append(int(pr.argmax()))
        confs.append(float(pr.max()))
        s = np.sort(pr)
        margins.append(float(s[-1] - s[-2]))
        paths.append(path)
        probs.append(pr)
    if not y:
        raise SystemExit("no images could be read")
    y = np.array(y, np.int32)
    preds = np.array(preds, np.int32)
    confs = np.array(confs, np.float32)
    margins = np.array(margins, np.float32)
    probs = np.array(probs, np.float32)
    k = len(classes)
    cm = confusion(y, preds, k)
    m = metrics(cm)
    lo, hi = bootstrap_macro_f1(y, preds, k)
    ab = abstain_report(y, probs, k, abstain_threshold)
    support = np.bincount(y, minlength=k)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "conf", "margin", "abstain"])
        for i in range(len(y)):
            w.writerow([paths[i], classes[y[i]], classes[preds[i]],
                        f"{confs[i]:.4f}", f"{margins[i]:.4f}",
                        "yes" if confs[i] < abstain_threshold else "no"])
    with open(out_dir / "confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + classes)
        for i in range(k):
            w.writerow([classes[i]] + [int(cm[i, j]) for j in range(k)])

    print(f"\n=== FIELD TEST (eval only) ===")
    print(f"photos: {len(y)} images, {len(set(classes[i] for i in y))} class(es) | "
          f"input {size}x{size} float32 0-255 | abstain@{abstain_threshold:.2f}")
    print(f"\n  {'class':<24} {'support':>7} {'precision':>9} {'recall':>7} {'f1':>6}")
    for i, c in enumerate(classes):
        print(f"  {c:<24} {support[i]:>7d} {m['precision'][i]:>9.3f} "
              f"{m['recall'][i]:>7.3f} {m['f1'][i]:>6.3f}")
    print(f"\nMACRO-F1 {m['macro_f1']:.4f} | ACC {m['accuracy']:.4f} | "
          f"WORST-CLASS RECALL {m['worst_recall']:.4f}")
    print(f"macro-F1 bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"abstain@{abstain_threshold:.2f}: coverage {ab['coverage']:.1%} | "
          f"acc on covered {ab['accuracy']:.4f} | macro-F1 on covered {ab['macro_f1']:.4f}")
    print(f"\nwrote {out_dir / 'predictions.csv'} and {out_dir / 'confusion.csv'}")
    return {"n": int(len(y)), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "worst_recall": m["worst_recall"], "ci": (float(lo), float(hi)),
            "abstain": ab, "classes": classes}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Evaluate the deployed Agrisense model on real field photos (eval only).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", help="agrisense_bundle.zip (model.tflite + class_names.json inside)")
    src.add_argument("--model", help="path to model.tflite (use with --classes)")
    ap.add_argument("--classes", help="path to class_names.json (required with --model)")
    ap.add_argument("--photos", required=True, help="folder of field photos (subfolder per class, "
                                                   "or flat with --labels)")
    ap.add_argument("--labels", help="CSV with columns path,class for flat photo folders")
    ap.add_argument("--abstain", type=float, default=None,
                    help="override the shipped abstain_threshold")
    ap.add_argument("--limit", type=int, default=None, help="cap images per class")
    ap.add_argument("--out", default=None,
                    help="output directory for predictions.csv + confusion.csv (default: photos dir)")
    args = ap.parse_args(argv)

    if args.bundle:
        with zipfile.ZipFile(args.bundle) as z:
            names = z.namelist()
            model_name = next((n for n in names if n.endswith(".tflite")), None)
            contract_name = next((n for n in names if n.endswith("class_names.json")), None)
            if not model_name or not contract_name:
                raise SystemExit(f"bundle missing model.tflite or class_names.json (found: {names})")
            tmp = pathlib.Path(tempfile.mkdtemp(prefix="agrisense_field_"))
            z.extract(model_name, tmp)
            z.extract(contract_name, tmp)
            model_path, contract_path = tmp / model_name, tmp / contract_name
    else:
        model_path, contract_path = pathlib.Path(args.model), pathlib.Path(args.classes)
        if not model_path.exists() or not contract_path.exists():
            raise SystemExit("--model and --classes must both exist")

    classes, contract_size, shipped_abstain = load_contract(contract_path)
    abstain = args.abstain if args.abstain is not None else shipped_abstain
    predict, (h, w) = load_model(model_path)
    size = h
    if (h, w) != (contract_size, contract_size):
        print(f"WARNING: TFLite input is {h}x{w} but the contract says {contract_size}x"
              f"{contract_size} — preprocessing to the interpreter's {h}x{w}")

    photos = discover_photos(args.photos, args.labels, classes, args.limit)
    if not photos:
        raise SystemExit("no photos found")
    print(f"field photos: {len(photos)} images, "
          f"{len(set(c for _, c in photos))} class(es)")

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(args.photos)
    run_eval(photos, predict, classes, size, abstain, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())