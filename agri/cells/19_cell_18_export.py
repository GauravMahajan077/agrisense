# CELL 18 — export
# PREPROCESSING CONTRACT (this is what the frontend must do):
#   1. decode to RGB
#   2. downscale the LONG EDGE to about 256 px using a HIGH-QUALITY filter
#      (PIL LANCZOS, canvas drawImage with imageSmoothingQuality='high', or cv2.INTER_AREA)
#   3. feed raw 0-255 floats, shape (1, 256, 256, 3), channels_last
# The graph then does 256 -> 224 bilinear, which at that ratio is nearly 1:1 and aliases
# negligibly. Do NOT feed a 4000px photo straight in: in-graph non-antialiased bilinear
# would alias exactly the spot texture this model exists to read, and TFLite cannot do an
# antialiased resize in-graph. Training used PIL BILINEAR, which DOES antialias, so the
# step-2 requirement is what removes the train/serve skew.
inp = tf.keras.Input((None, None, 3), dtype=tf.float32)
x   = tf.keras.layers.Resizing(SIZE, SIZE, interpolation="bilinear")(inp)
clean = tf.keras.Model(inp, model(x), name="agrisense_infer")

# The functional API REUSES layer objects, so `clean` already holds the trained weights.
shared = any(l is model for l in clean.layers)
if shared:
    print(f"weights shared by construction ({model.name} is a layer of {clean.name})")
else:
    tgt = {l.name: l for l in clean.get_layer(model.name).layers}
    bad = []
    for l in model.layers:
        if not l.get_weights(): continue
        if l.name not in tgt: bad.append(l.name); continue
        try: tgt[l.name].set_weights(l.get_weights())
        except ValueError: bad.append(l.name)
    print(f"name-matched transfer {len(model.layers)-len(bad)}/{len(model.layers)}")
    if bad: print("MISMATCH:", bad)
clean.summary(line_length=110)
clean.save(str(WORK / "agrisense_b0.keras"))

contract = {
    "classes": CLASSES, "graph_input_size": SIZE, "recommended_input_size": PRE,
    "channels_last": True, "value_range": [0, 255], "dtype": "float32",
    "steps": ["decode RGB",
              f"downscale long edge to ~{PRE}px with a high-quality filter "
              "(Lanczos / imageSmoothingQuality=high / INTER_AREA)",
              "emit (1, H, W, 3) float32 in 0..255; graph resizes to 224"],
    "do_not": ["nearest-neighbour resize", "skip the pre-downscale step"],
    "val_macro_f1": round(float(np.nanmean(f1_va)), 4),
    "test_macro_f1": round(float(np.nanmean(f1_te)), 4),
    "worst_class_recall": round(float(rec_te.min()), 4),
}
(WORK / "class_names.json").write_text(json.dumps(contract, indent=2))
print(f"\nDECLARED {NC} == OUTPUT {clean.output_shape[-1]}  "
      f"{'OK' if clean.output_shape[-1]==NC else 'MISMATCH — DO NOT SHIP'}")
