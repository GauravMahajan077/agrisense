# CELL 19 — TFLite + parity check + bundle
parity = None
if CFG["export"]["tflite"]:
    conv = tf.lite.TFLiteConverter.from_keras_model(clean)
    conv.optimizations = []                 # no quantisation -> stays in builtin ops
    tfl = conv.convert()
    (WORK / "agrisense_b0.tflite").write_bytes(tfl)
    print(f"tflite {(WORK/'agrisense_b0.tflite').stat().st_size/1e6:.1f} MB")

    # PARITY CHECK: Keras vs the TFLite interpreter on real val images. Without this you
    # are shipping a converted model on faith.
    interp = tf.lite.Interpreter(model_content=tfl)
    inp_d = interp.get_input_details()[0]; out_d = interp.get_output_details()[0]
    interp.allocate_tensors()
    # the exported graph has a dynamic H,W input; if the converter pinned it, use that
    shp = inp_d["shape"]
    h, w = (int(shp[1]), int(shp[2])) if int(shp[1]) > 0 else (PRE, PRE)
    rng_p = np.random.RandomState(0)
    agree, maxdiff, nprobe = 0, 0.0, min(16, len(yva))
    for i in rng_p.choice(len(yva), size=nprobe, replace=False):
        probe = (Xva[i][None] if Xva is not None
                 else np.asarray(Image.open(va["path"].iloc[i]).convert("RGB")
                                 .resize((PRE, PRE), Image.BILINEAR))[None])
        probe = probe.astype(np.float32)
        if (h, w) != (PRE, PRE):
            probe = tf.image.resize(tf.convert_to_tensor(probe), (h, w)).numpy()
        k_out = model.predict(probe, verbose=0)[0]
        interp.set_tensor(inp_d["index"], probe.astype(inp_d["dtype"]))
        interp.invoke()
        t_out = interp.get_tensor(out_d["index"])[0]
        agree += int(np.argmax(k_out) == np.argmax(t_out))
        maxdiff = max(maxdiff, float(np.abs(k_out - t_out).max()))
    parity = agree / nprobe
    print(f"PARITY keras-vs-tflite: {parity:.0%} argmax agreement over {nprobe} images "
          f"at {h}x{w}, max |delta prob| = {maxdiff:.4f}")
    if parity < 1.0:
        print("!! parity < 100% — investigate before shipping")

import zipfile
with zipfile.ZipFile(WORK / "agrisense_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(WORK / "agrisense_b0.keras", "agrisense_b0.keras")
    z.write(WORK / "class_names.json", "class_names.json")
    if CFG["export"]["tflite"]: z.write(WORK / "agrisense_b0.tflite", "agrisense_b0.tflite")
    for f in ("val_report.csv", "test_confusion.csv", "field_stress_test.csv"):
        if (OUT / f).exists(): z.write(OUT / f, f)
    if parity is not None:
        z.writestr("parity.txt", f"keras-vs-tflite argmax agreement {parity:.0%}\n")
print(f"bundle {(WORK/'agrisense_bundle.zip').stat().st_size/1e6:.1f} MB — download from "
      f"Kaggle Output, then attach it as a Dataset for future runs")
print(f"total notebook wall clock {elapsed():.1f} min")
print("TFJS: separate notebook — pip install tensorflow==2.15.1 tensorflowjs numpy==1.26.4")
