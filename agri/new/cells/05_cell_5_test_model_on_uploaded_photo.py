# ============================================================
# CELL 5 — TEST the model on a photo (upload from your files)
# ============================================================
# 1) Click "Choose a photo" -> your file explorer opens -> pick a photo.
# 2) Click "Predict" -> the model runs and prints the prediction.
import io, json, tempfile, zipfile
import numpy as np
from PIL import Image
from IPython.display import display
import tensorflow as tf
from ipywidgets import FileUpload, Button, VBox

# ---- where the model lives ----
BUNDLE = "/kaggle/working/agrisense_bundle.zip"   # same session as training
# BUNDLE = "/kaggle/input/<your-dataset>/agrisense_bundle.zip"  # attached as a Dataset

# ---- load the bundle (tflite + class_names.json) ----
tmp = tempfile.mkdtemp(prefix="agrisense_test_")
with zipfile.ZipFile(BUNDLE) as z:
    names = z.namelist()
    tfl = next(n for n in names if n.endswith(".tflite"))
    cj  = next(n for n in names if n.endswith("class_names.json"))
    z.extract(tfl, tmp); z.extract(cj, tmp)
contract = json.load(open(f"{tmp}/{cj}", encoding="utf-8"))
classes = contract["classes"]
abstain = float(contract.get("abstain_threshold", 0.5))

interp = tf.lite.Interpreter(model_path=f"{tmp}/{tfl}")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
SIZE = int(inp["shape"][1])
print(f"model: {SIZE}x{SIZE} float32 0-255 | classes: {classes} | abstain@{abstain}")

def predict(img):
    interp.set_tensor(inp["index"], img[None].astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out["index"])[0].astype(np.float32)

def run_inference(data, filename):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32)          # 0-255, NEVER divide by 255
    pr = predict(x)
    top = int(pr.argmax())
    s = np.sort(pr)
    conf, margin = float(pr[top]), float(s[-1] - s[-2])
    abst = "ABSTAIN (low confidence)" if conf < abstain else "OK"
    print(f"\n{filename}")
    print(f"  -> {classes[top]}  (conf {conf:.3f}, margin {margin:.3f})  [{abst}]")
    for i, c in enumerate(classes):
        print(f"     {c:<24} {pr[i]:.3f} {'#' * int(round(pr[i] * 30))}")
    display(img)

# ---- upload button + Predict button (reliable on Kaggle) ----
upload = FileUpload(accept="image/*", multiple=False, description="Choose a photo")
predict_btn = Button(description="Predict", button_style="primary")

def on_predict(b):
    if not upload.value:
        print("No photo selected yet — click 'Choose a photo' first.")
        return
    for name, info in upload.value.items():
        run_inference(info["content"], name)

predict_btn.on_click(on_predict)
display(VBox([upload, predict_btn]))
print("1) Choose a photo   2) Click Predict")