# AGENTS.md — Agrisense (rice leaf disease, Kaggle, TF 2.20 / Keras 3, 2x T4)
Goal: BETTER, not perfect. Real-world (internet/field) performance matters more than lab F1.
Rules:
- Do not add features. Fix only what is listed in the task or what a real error forces.
- One module `agrisense.py`; notebook = import + run. Print PIPELINE_VERSION in every entry point.
- Must have `smoke` mode (40 img/class, 1 epoch/stage, no crawl/export). Run it before any full run.
- Never report success without pasting real run output. If you can't run it, say so.
- Keep: class_weight, progressive unfreeze + asserts, val_macro_f1 checkpoint, grouped split.
- Removed on purpose: LSH/sweep, hash cache, oversample/effective, finetune-crawl, dynamic TFLite.
- Headline metric = source-held-out (train anshul6+indo3, test dedeikh, 5 shared classes). In-source test is secondary.
- Model input is 0-255 float (EfficientNet has internal preprocessing). Never divide by 255 at inference.
- Dedupe: per-image 8 D4 pHashes, distance = min over variants, brute-force numpy, merge SAME-class pairs only.
Known traps: Keras 3 predict() needs the dataset, not (x,y) tuples; set_shape after dynamic slice; export in float32 with fixed 256x256 input.
