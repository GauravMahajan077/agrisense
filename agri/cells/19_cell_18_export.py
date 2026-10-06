# CELL 18 — export
# PREPROCESSING CONTRACT (this is what the frontend must do):
#   1. decode to RGB
#   2. resize to EXACTLY 256 x 256 using a HIGH-QUALITY filter
#      (PIL LANCZOS, canvas drawImage with imageSmoothingQuality='high', or cv2.INTER_AREA)
#   3. feed raw 0-255 floats, shape (1, 256, 256, 3), channels_last
# The input is FIXED at 256x256, so any other size is rejected outright instead of being
# silently mis-shape-checked. The graph then does 256 -> 224 bilinear, which at that ratio is
# nearly 1:1 and aliases negligibly. Do NOT feed a 4000px photo straight in: in-graph
# non-antialiased bilinear would alias exactly the spot texture this model exists to read, and
# TFLite cannot do an antialiased resize in-graph. Training used PIL BILINEAR, which DOES
# antialias, so step 2 is what removes the train/serve skew.
#
# EXPORT MUST BE FLOAT32. Training runs mixed_float16 (CFG["amp"]) and TFLite cannot legalise
# f16 Conv2D/MatMul: conversion dies with ConverterError "'tf.Conv2D' op is neither a custom
# op nor a flex op". Wrapping `model` does NOT fix it — a layer's dtype policy is baked in at
# build time and set_global_policy() afterwards never touches existing layers. So the export
# graph is REBUILT under float32 and the trained weights are copied across.
# NOTE the policy stays float32 until `clean` exists: `Resizing` is a NEW layer, so building
# it while mixed_float16 is active gives it an f16 compute policy -> tf.ResizeBilinear on f16
# -> ConverterError "'tf.ResizeBilinear' op is neither a custom op nor a flex op".
_prev_policy = tf.keras.mixed_precision.global_policy()
tf.keras.mixed_precision.set_global_policy("float32")
export_base = build_model(NC)
_w_tr, _w_ex = model.get_weights(), export_base.get_weights()
assert [a.shape for a in _w_tr] == [a.shape for a in _w_ex], "rebuild changed the weight layout"
export_base.set_weights(_w_tr)
_w_chk = export_base.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(_w_tr, _w_chk)), "trained weights did not land"

inp = tf.keras.Input((PRE, PRE, 3), batch_size=1, dtype=tf.float32)
x   = tf.keras.layers.Resizing(SIZE, SIZE, interpolation="bilinear")(inp)
clean = tf.keras.Model(inp, export_base(x), name="agrisense_infer")
tf.keras.mixed_precision.set_global_policy(_prev_policy)   # restore only after clean exists
print(f"export rebuilt as float32 from {model.name}: {len(_w_ex)} tensors copied and verified "
      f"(max |delta| vs {model.name} = "
      f"{max(float(np.abs(a - b).max()) for a, b in zip(_w_tr, _w_chk)):.1e})")
clean.summary(line_length=110)
clean.save(str(WORK / "agrisense_b0.keras"))

contract = {
    "classes": CLASSES, "graph_input_size": SIZE, "recommended_input_size": PRE,
    "input_shape": [1, PRE, PRE, 3],
    "channels_last": True, "value_range": [0, 255], "dtype": "float32",
    "steps": ["decode RGB",
              f"resize to exactly {PRE}x{PRE} with a high-quality filter "
              "(Lanczos / imageSmoothingQuality=high / INTER_AREA)",
              f"emit (1, {PRE}, {PRE}, 3) float32 in 0..255; graph resizes to {SIZE}"],
    "do_not": ["nearest-neighbour resize", "feed any other HxW (the input is fixed)",
               "divide by 255"],
    "val_macro_f1": round(float(np.nanmean(f1_va)), 4),
    "test_macro_f1": round(float(np.nanmean(f1_te)), 4),
    "worst_class_recall": round(float(rec_te.min()), 4),
}
(WORK / "class_names.json").write_text(json.dumps(contract, indent=2))
print(f"\nDECLARED {NC} == OUTPUT {clean.output_shape[-1]}  "
      f"{'OK' if clean.output_shape[-1]==NC else 'MISMATCH — DO NOT SHIP'}")
