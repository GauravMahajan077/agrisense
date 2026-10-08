
# CELL 5 TESTSINGS

import io, os, json, random, tempfile, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from IPython.display import display
import tensorflow as tf

BUNDLE = "/kaggle/working/agrisense_bundle.zip"      # same session as training
# BUNDLE = "/kaggle/input/datasets/<you>/<your-dataset>/agrisense_bundle.zip"

# Your own photos: a file path or a folder path. Leave "" to skip.
MY_PHOTOS = ""    # e.g. "/kaggle/input/datasets/<you>/my-leaves"
SHOW_IMAGES = True

# ---- load the bundle ----
if not os.path.exists(BUNDLE):
    raise FileNotFoundError(f"{BUNDLE} not found. If the session restarted, attach the bundle "
                            f"as a Dataset and point BUNDLE at it.")
tmp = tempfile.mkdtemp(prefix="agrisense_test_")
with zipfile.ZipFile(BUNDLE) as z:
    names = z.namelist()
    tfl = next(n for n in names if n.endswith(".tflite"))
    cj  = next(n for n in names if n.endswith("class_names.json"))
    z.extract(tfl, tmp); z.extract(cj, tmp)
contract = json.load(open(os.path.join(tmp, cj), encoding="utf-8"))
classes = contract["classes"]
abstain = float(contract.get("abstain_threshold", 0.5))

interp = tf.lite.Interpreter(model_path=os.path.join(tmp, tfl))
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
H, W = int(inp["shape"][1]), int(inp["shape"][2])
print(f"model input {H}x{W} float32 0-255 | classes: {classes} | abstain@{abstain}")

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def predict_image(img, name, true=None, show=SHOW_IMAGES):
    img = ImageOps.exif_transpose(img).convert("RGB")      # phone photos: fix rotation
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)            # contract: high-quality resize
    x = np.asarray(img, dtype=np.float32)[None]            # 0-255, NEVER /255
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    pr = interp.get_tensor(out["index"])[0].astype(np.float32)
    top = int(pr.argmax()); s = np.sort(pr)
    conf, margin = float(pr[top]), float(s[-1] - s[-2])
    flag = "ABSTAIN" if conf < abstain else "ok"
    tag = f"  (folder label: {true})" if true else ""
    print(f"\n{name}{tag}\n  -> {classes[top]}  conf {conf:.3f}  margin {margin:.3f}  [{flag}]")
    for i, c in enumerate(classes):
        print(f"     {c:<24} {pr[i]:.3f} {'#' * int(round(pr[i] * 30))}")
    if show:
        display(img)
    return classes[top], conf

# ---- 1) sanity test on held-out dedeikh images ----
roots = [p for pat in ("/kaggle/input/*/rice-leafs-disease-dataset",
                       "/kaggle/input/datasets/*/rice-leafs-disease-dataset",
                       "/kaggle/input/rice-leafs-disease-dataset")
         for p in map(str, Path("/").glob(pat.lstrip("/")))]
if roots:
    files = [p for p in Path(roots[0]).rglob("*") if p.suffix.lower() in EXT]
    random.seed(0)
    print(f"\n=== SANITY: 8 random images from {roots[0]} ===")
    hits = 0; sample = random.sample(files, min(8, len(files)))
    for p in sample:
        pred, _ = predict_image(Image.open(p), p.name, true=p.parent.name, show=False)
        hits += int(pred.lower().replace("_", "") == p.parent.name.lower().replace("_", ""))
    print(f"\nname-match {hits}/{len(sample)} (folders like narrow_brown_spot/tungro won't match by design)")

# ---- 2) your own photos (file or folder) ----
if MY_PHOTOS:
    p = Path(MY_PHOTOS)
    mine = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in EXT)
    print(f"\n=== YOUR PHOTOS: {len(mine)} ===")
    for q in mine:
        predict_image(Image.open(q), q.name)

# ---- 3) optional upload button (ipywidgets 7 AND 8) ----
try:
    from ipywidgets import FileUpload, Button, VBox, Output
    upload = FileUpload(accept="image/*", multiple=True, description="Choose photo(s)")
    btn = Button(description="Predict", button_style="primary")
    log = Output()

    def _files(w):
        v = w.value
        if isinstance(v, dict):                                    # ipywidgets 7
            return [(n, bytes(i["content"])) for n, i in v.items()]
        return [(f["name"], bytes(f["content"])) for f in v]       # ipywidgets 8

    def on_click(_):
        with log:
            log.clear_output()
            fs = _files(upload)
            if not fs:
                print("No photo selected. Click 'Choose photo(s)' first."); return
            for name, data in fs:
                predict_image(Image.open(io.BytesIO(data)), name)

    btn.on_click(on_click)
    display(VBox([upload, btn, log]))
except Exception as e:
    print(f"(upload widget unavailable here: {type(e).__name__}) - use MY_PHOTOS instead")
