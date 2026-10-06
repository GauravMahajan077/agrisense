# # Agrisense — paddy disease classification, Kaggle pipeline
#
# **Read `README.md` first.** `agrisense_kaggle.py`, cells marked `# %%`.
# Cell 1 (`CFG`) is the only cell you edit.
#
# ## What this version fixes relative to the previous one
#
# Hard crashes that were present before:
#   1. `tf.image.random_resized_crop` does not exist -> use `sample_distorted_bounding_box`.
#   2. Color ops ran on 0-255 (`adjust_hue`/`adjust_saturation` assume 0-1) and
#      `random_contrast`/`random_saturation` were missing `lower`/`upper`.
#   3. `import keras.callbacks as K` was shadowed by `N, K = ...` -> module renamed `KC`.
#   4. Labels were int32 while the loss/metric needed one-hot -> one-hot now emitted in-pipeline.
#   5. Eval images were 256px but the model input is 224px -> eval resizes to SIZE.
#
# Silent failures that were present before (these mattered more):
#   6. `set_trainable` matched layer names against "efficientnet", but Keras names EfficientNet
#      internals `stem_conv` / `block1a_dwconv`. `core` was the single nested base model, so
#      ALL THREE STAGES trained only the final softmax on top of a random frozen Dense(128).
#      Progressive unfreezing was a no-op. Now it walks `base.layers` directly.
#   7. Dedupe kept ONE image per cluster, which made every cluster a singleton — so the
#      "grouped split" and its leak assert were vacuous. Now duplicates are KEPT and the split
#      is by cluster, which is what the assert is supposed to mean.
#   8. pHash is not flip/rotate invariant, so the `Rice_Leaf_AUG` siblings it was built to
#      cluster did NOT cluster. Now hashing is over the 8 dihedral (D4) variants.
#   9. Checkpoint chosen by file size, and each stage's `best` reset to -1 so Stage C always
#      saved its first epoch. Now one global best, tracked in one file.
#  10. `MacroF1.result()` used numpy inside a traced graph and had no `get_config`.
#  11. `compile()` ran outside `STRATEGY.scope()`.
#  12. `merge_narrow_brown` was never wired to anything.
#  13. `min_class` was checked pre-dedupe/pre-split, so a thin class could vanish later.
#  14. `shuffle(20000)` over 196 KB images is a ~3.9 GB buffer; `from_tensor_slices` on a big
#      array embeds it as a graph constant and trips the 2 GB protobuf limit. Both replaced by
#      a `from_generator` that permutes INDICES in Python.
#  15. Cell 3 extracted the download zip and left it on disk, so peak usage was 2x the dataset
#      instead of 1x. On /kaggle/working (~20 GB) the 8.1 GB shayanriyaz dataset filled the
#      volume and every later cell died with `OSError: [Errno 28]` — including a 3 KB CSV.
#      Now: mount-first resolution, a disk preflight, the zip deleted after extraction, the
#      partial tree removed on failure, and an early raise when < 1 GB is left.
