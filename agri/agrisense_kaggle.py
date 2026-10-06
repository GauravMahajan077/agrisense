# %% [markdown]
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










# %%
# CELL 0 — environment probe
# Laptop-safe on purpose: no hard exit, no Kaggle-only paths. You can dry-run this on your
# machine to sanity-check imports. It will refuse to *train* without a GPU, not refuse to start.
import os, sys, time, shutil, tempfile, subprocess
from pathlib import Path

# NOTE: PYTHONHASHSEED is read by the interpreter at startup, so setting it here is a no-op.
# It is set only because Kaggle lets you control it from the environment panel. Nothing in
# this pipeline depends on it (all seeding is explicit).


os.environ.setdefault("PYTHONHASHSEED", "1337")
_NC = max(1, os.cpu_count() or 4)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hides the absl/CUDA banner noise
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(min(8, _NC)))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
T0 = time.time()

# Bumped on every intentional pipeline change. Printed at every entry point (start, train,
# export, bundle) so a stale paste is visible in the log instead of silently shipping.
PIPELINE_VERSION = "2.0.0"

import numpy as np, tensorflow as tf

KAGGLE = Path("/kaggle").is_dir()
WORK = Path("/kaggle/working") if KAGGLE else Path(tempfile.gettempdir()) / "agrisense"
WORK.mkdir(parents=True, exist_ok=True)

def _ram_gb():
    """Cross-platform RAM probe. Never raises — a diagnostic must not be able to kill a run."""
    try:
        import psutil
        v = psutil.virtual_memory(); return v.total / 1e9, v.available / 1e9
    except Exception:
        pass
    try:
        d = {l.split(":")[0].strip(): int(l.split()[1])
             for l in open("/proc/meminfo") if ":" in l}
        return d["MemTotal"] / 1e6, d["MemAvailable"] / 1e6
    except Exception:
        return float("nan"), float("nan")

def _free_gb(p=None):
    """Free bytes on the filesystem holding WORK, in GB. Never raises.

    Disk headroom is the binding constraint on Kaggle: /kaggle/working is ~20 GB and one
    source (shayanriyaz) is an 8 GB zip that expands to another 8 GB. Every download in
    Cell 3 is therefore gated on this number."""
    try:
        return shutil.disk_usage(p or WORK).free / 1e9
    except Exception:
        return float("nan")

RAM_GB, RAM_AVAIL_GB = _ram_gb()
free_gb = _free_gb()

def _tmp_gb():
    """Free space on the system temp dir.

    Reported separately because it is the resource that actually failed: the `kaggle` CLI
    stages its download through temp, and on Kaggle /tmp is a much smaller filesystem than
    /kaggle/working. Cell 3 now redirects TMPDIR at WORK so the two agree — but the sizes are
    worth seeing, because "19.5 GB free on /kaggle/working" coexisting with Errno 28 on an
    8 GB download is otherwise a mystery.
    """
    import tempfile as _tf
    for cand in (_tf.gettempdir(), "/tmp"):
        try:
            g = shutil.disk_usage(cand).free / 1e9
            return f"{g:.1f} GB free on {_tf.gettempdir()}"
        except Exception:
            continue
    return "temp space unknown"

def _dir_gb(d):
    """Recursive on-disk size of a tree, in GB. Used to report what we just freed."""
    try:
        return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e9
    except Exception:
        return 0.0

print(f"PIPELINE_VERSION {PIPELINE_VERSION} | python {sys.version.split()[0]} | cpus {_NC} | "
      f"platform {'kaggle' if KAGGLE else 'local'}")
print(f"tf {tf.__version__} | numpy {np.__version__} | keras {tf.keras.__version__}")
print(f"RAM {RAM_GB:.1f} GB total / {RAM_AVAIL_GB:.1f} GB available | disk {free_gb:.1f} GB free in WORK")
print(f"temp: {_tmp_gb()} (Cell 3 redirects TMPDIR here, so WORK is the only limit)")
print(f"work dir: {WORK}")

gpus = tf.config.list_physical_devices("GPU")
# Best-effort memory growth. One effect that matters here: it stops TF pre-allocating all
# 13.7 GB per T4, which is what turns a late allocation into "OOM at the last epoch" instead
# of a clean failure. Must run before any GPU op; if it fails we carry on.
#
# NOTE: it does NOT enable get_memory_info(). That was my earlier claim and it was wrong —
# TF still raises ValueError("Memory statistics not tracked") because the BFC allocator needs
# separate instrumentation. We read VRAM from nvidia-smi instead, which always works here.
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print(f"  set_memory_growth({g.name}) refused: {type(e).__name__}")

def _vram_lines():
    """Per-GPU total/used VRAM in MB, via nvidia-smi. Empty list if unavailable."""
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,name,memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            return [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    except Exception:
        pass
    return []

_vram = _vram_lines()
if _vram:
    for ln in _vram:
        parts = [p.strip() for p in ln.split(",")]
        # Never let a display probe kill the run: a GPU name containing a comma, or a driver
        # that changes the column set, must degrade to a printed line, not an exception.
        try:
            idx, name, total, used = parts
            print(f"  GPU {idx}: {name}  VRAM {int(total)/1024:.1f} GB total / "
                  f"{int(used)/1024:.2f} GB used")
        except Exception:
            print(f"  GPU (unparsed nvidia-smi line): {ln}")
else:
    print("  (nvidia-smi unavailable — VRAM figures omitted, training unaffected)")
print(f"GPUs detected: {len(gpus)}")
if not gpus:
    print("!! No GPU. Cells 0-9 still run; Cell 14 will be very slow or fail.")
    print("!! On Kaggle: Settings -> Accelerator -> GPU T4 x2 -> Save.")
print(f"disk headroom: {_free_gb():.1f} GB")










# %%
# CELL 1 — CONFIG. The only cell you edit.
CFG = {
    "seed": 1337,

    # smoke = 40 img/class, 1 epoch/stage, no crawl, no export. Run it before any full run.
    "smoke": False,

    # ---------- image / speed ----------
    "img_size": 224,               # model input
    "pre_size": 256,               # preload target; augmentation crops 256 -> 224
    "batch_size_per_replica": 64,  # global batch = this x num_replicas
    "amp": True,                   # mixed_float16
    "jit": True,                   # XLA (auto-skipped when multi_gpu is on)
    "multi_gpu": False,            # single GPU by default: MirroredStrategy adds friction
                                   # (XLA off, batch split) for no accuracy gain. True to try
                                   # both T4s anyway.
    "preload_ram": True,           # pre-decode into a RAM array (the speed win)
    "ram_frac": 0.40,              # preload budget as a fraction of AVAILABLE ram
    "workers": 8,

    # Wall-clock cap on TRAINING ONLY (starts at Cell 14, not at notebook start).
    "train_budget_min": 20,

    # ---------- canonical taxonomy: single source of truth ----------
    # 6 classes, from anshul6 + dedeikh + indo3. Two classes of the original 8 are absent:
    #   Hispa  — only shayanriyaz had it, and that download does not fit (see CFG["sources"]).
    #   Tungro — only indo3 had it, at ~80 images, below min_class.
    # Both are recoverable, but neither is worth blocking on. See README section 3.
    "classes": ["Bacterial_Leaf_Blight", "Brown_Spot", "Healthy",
                "Leaf_Blast", "Leaf_Scald", "Sheath_Blight"],
    # dedeikh also ships "Narrow Brown Spot". At 224px it is not reliably separable from
    # Brown Spot, so by default it folds in and donates its images.
    # Set False AND add "Narrow_Brown_Spot" to "classes" to keep it as its own label.
    "merge_narrow_brown": True,

    # ---------- data sources ----------
    # type "auto"   : Add Input mount first, then a disk-gated CLI download
    # type "local"  : mount only, fail loudly if absent
    # type "kaggle" : CLI download only, still gated on free disk
    #
    # USE "Add Input". Mounted datasets live in /kaggle/input and cost ZERO /kaggle/working
    # space; the CLI path costs 2x the zip size (zip + extracted copy) on a ~20 GB volume.
    # `zip_gb` is only used to preflight the CLI path — measured from Kaggle's dataset
    # metadata, not guessed. Delete a source to drop its classes (see Cell 7 for what happens).
    "allow_cli_download": True,
    "sources": [
        {"name": "anshul6",    "type": "auto", "slug": "anshulm257/rice-disease-dataset",
         "mount": "/kaggle/input/rice-disease-dataset", "zip_gb": 1.1},
        {"name": "dedeikh",    "type": "auto", "slug": "dedeikhsandwisaputra/rice-leafs-disease-dataset",
         "mount": "/kaggle/input/rice-leafs-disease-dataset", "zip_gb": 0.4},
        # shayanriyaz is DISABLED. It is 8.04 GB for 3,355 images and the `kaggle` CLI failed
        # to fetch it three ways (direct, with TMPDIR redirected at WORK, with the zip deleted
        # after extraction) while /kaggle/working still had 19.5 GB free — so the limit is a
        # Kaggle-side quota or CLI staging path we cannot see or size. Its only unique class is
        # Hispa; its author describes it as an aggregation of other web datasets, so much of it
        # probably duplicates anshul6/dedeikh. Enable it only via Add Input (mounted datasets
        # cost no working disk). Re-enable by uncommenting the line and adding
        # "Hispa" to CFG["classes"].
        # {"name": "shayan_cc0", "type": "auto", "slug": "shayanriyaz/riceleafs",
        #  "mount": "/kaggle/input/riceleafs", "zip_gb": 8.1},

        # indo3 IS enabled but contributes little: 240 images across 3 classes, so ~80 per class
        # and ~56 after the train split. That is under min_class, so Cell 7 drops its unique
        # class (Tungro) and keeps only the Brown Spot / Leaf Blast images as extra samples.
        {"name": "indo3",      "type": "auto", "slug": "tedisetiady/leaf-rice-disease-indonesia",
         "mount": "/kaggle/input/leaf-rice-disease-indonesia", "zip_gb": 0.3},
    ],

    # folder name -> canonical class. Keys are matched after stripping non-alphanumerics and
    # lowercasing, so "Leaf Scald", "leaf_scald" and "LeafScald" all hit the same entry.
    #
    # Deliberately ABSENT, because these are guesses that would silently mislabel rather than
    # surface as UNMAPPED for review:
    #   "blight", "brown", "good", "sheath"      -> too generic, could match anything
    #   "bacterialleafspot"                       -> a DIFFERENT disease from leaf blight
    #   "yellowleaf", "leafyellow"                -> not reliably Tungro
    # If one of those appears, Cell 4 lists it under UNMAPPED and you add it deliberately.
    "alias": {
        "bacterialleafblight": "Bacterial_Leaf_Blight",
        "bacterialleaf":       "Bacterial_Leaf_Blight",
        "brownspot":           "Brown_Spot",
        "brownleafspot":       "Brown_Spot",
        "healthy":             "Healthy",
        "healthyriceleaf":     "Healthy",
        "healthyleaf":         "Healthy",
        "normal":              "Healthy",          # standard convention for healthy leaves
        "hispa":               "Hispa",
        "ricehispa":           "Hispa",
        "leafblast":           "Leaf_Blast",
        "blast":               "Leaf_Blast",
        "riceblast":           "Leaf_Blast",
        "leafscald":           "Leaf_Scald",
        "scald":               "Leaf_Scald",
        "sheathblight":        "Sheath_Blight",
        "shb":                 "Sheath_Blight",
        "tungro":              "Tungro",
        "tungrovirus":         "Tungro",
    },

    # ---------- per-source alias overrides ----------
    # Checked BEFORE the global table below. This is how a genuinely ambiguous folder name
    # becomes safe to map: "blight" on its own is untrustworthy — it could mean leaf blight,
    # stem blight or fire blight, which is exactly why it is absent from "alias". But inside
    # indo3 it is unambiguous: that dataset is 240 images in three folders, `leafblast`,
    # `tungro` and `blight`. Two of those three already name themselves, so the third can only
    # be the bacterial disease. Scoping the mapping to the source removes the ambiguity instead
    # of guessing globally.
    "source_alias": {
        "indo3": {"blight": "Bacterial_Leaf_Blight"},
    },

    # ---------- split / dedupe ----------
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    # pHash is not flip/rotate invariant, so plain pHash misses exactly the `Rice_Leaf_AUG`
    # siblings we need to group. dihedral=True hashes all 8 D4 variants per image; the
    # distance between two images is the MIN over the 8x8 variant pairs (brute-force numpy).
    # near_dist is the single merge threshold, and ONLY same-class pairs merge (cross-class
    # near-duplicates are label noise -> excluded from val/test in Cell 6).
    "dedupe": {"enable": True, "dihedral": True, "near_dist": 4},
    "min_class": 120,   # MINIMUM TRAIN IMAGES, enforced AFTER the split (see Cell 7)

    # ---------- imbalance ----------
    # class_weight only. effective/oversample were removed on purpose (audit): class_weight
    # is the one that worked, and the others added config surface without a measured win.
    "imbalance": {"mode": "class_weight"},

    # ---------- honest evaluation (Phase 3) ----------
    # The headline metric is source-held-out: train on anshul6+indo3, test on dedeikh across
    # the 5 shared classes (Sheath_Blight is single-source in anshul6, so it stays in training
    # but is excluded from the held-out eval). dedeikh is the noisy source (README section 6),
    # so this is the honest number. Set to None to disable and train on every source.
    "held_out_source": "dedeikh",
    "abstain_threshold": 0.5,   # held-out/field eval: skip predictions below this confidence
    "bootstrap_iters": 2000,    # bootstrap CI for the headline macro-F1

    # ---------- augmentation (CPU, after preload -> never baked into exports) ----------
    # Applied in 0..1, matching tf.image.adjust_* expectations.
    "aug": {"rrc_scale": (0.70, 1.00), "flip_h": True, "flip_v": False,
            "brightness": 0.25, "contrast": 0.25, "sat": 0.15, "hue": 0.03},

    # ---------- progressive unfreezing ----------
    # n > 0 = first n backbone layers, n < 0 = last n, 0 = backbone frozen.
    # Only the BACKBONE is selected here; the head always trains in every stage.
    "stages": [
        {"name": "A_head",    "unfreeze": 0,   "epochs": 4, "lr": 1e-3},
        {"name": "B_shallow", "unfreeze": -18, "epochs": 6, "lr": 1e-4},
        {"name": "C_deep",    "unfreeze": -70, "epochs": 6, "lr": 3e-5},
    ],

    # ---------- crawled web photos: quarantined ----------
    "crawl": {
        "enable": True,
        "role": "stress_test",       # stress_test only (finetune removed on purpose)
        "providers": ["wikimedia", "ddg"],
        "per_class": 20,
        "min_side": 200,
        "max_bytes": 5_000_000,      # per-image download cap (a 20 MB photo is never useful)
        "sleep": 0.3,
        "max_seconds": 180,
        "queries": {
            "Brown_Spot": ["rice leaf brown spot disease", "oryza sativa brown spot leaf"],
            "Leaf_Blast": ["rice leaf blast disease", "rice blast lesion leaf field"],
            "Healthy":    ["healthy rice leaf close up", "rice plant leaf green"],
        },
    },

    "export": {"tflite": True},
}

if not CFG["merge_narrow_brown"] and "Narrow_Brown_Spot" not in CFG["classes"]:
    raise SystemExit("merge_narrow_brown=False requires 'Narrow_Brown_Spot' in CFG['classes']")










# %%
# CELL 2 — deps, imports, seeds, strategy
# imagehash and ddgs are NOT Kaggle preinstalls.
import importlib.util, subprocess as _sp

def _pip(*pkgs, required=True):
    # Each arg is (import_name, install_spec). The import name is what we probe; the spec is
    # what pip installs, and it is PINNED so a future package release cannot silently change
    # the pipeline. "PIL" is the import name of "Pillow" — pip has no package called "PIL".
    missing = [s for s in pkgs if importlib.util.find_spec(s[0]) is None]
    if not missing:
        print(f"present: {' '.join(s[0] for s in pkgs)}"); return
    names = [s[1] for s in missing]
    print(f"installing: {' '.join(names)} (import name(s): {' '.join(s[0] for s in missing)})")
    try:
        _sp.run([sys.executable, "-m", "pip", "install", "-q", *names], check=True)
        # pip returning 0 is not proof: the install can land yet the module still not import
        # (stale sys.path cache, or a shadowing local file). Re-probe before claiming OK.
        importlib.invalidate_caches()
        for imp, _ in missing:
            if importlib.util.find_spec(imp) is None:
                raise ImportError(f"{imp} still not importable after installing {' '.join(names)}")
        print("  installed and importable")
    except Exception as e:
        if required: raise
        print(f"  optional install failed ({type(e).__name__}) — continuing without it")

# Pinned: imagehash and ddgs are the two packages this pipeline actually installs on Kaggle.
# Pillow is a Kaggle preinstall, so it is left unpinned (only installed if missing).
_pip(("PIL", "Pillow"), ("imagehash", "imagehash==4.3.2"))
try:
    _pip(("ddgs", "ddgs==9.16.0"), required=False)   # DuckDuckGo image search, crawler only
except Exception:
    pass

import random, re, json, time, zipfile, subprocess, io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow.keras.callbacks as KC     # KC, not K — K is used as a class count in Cell 8

SEED = CFG["seed"]
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE
SIZE, PRE = CFG["img_size"], CFG["pre_size"]
OUT = WORK / "out"; OUT.mkdir(exist_ok=True)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}

if CFG["amp"]:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

STRATEGY = None
if CFG["multi_gpu"] and len(gpus) > 1:
    try:
        STRATEGY = tf.distribute.MirroredStrategy()
    except Exception as e:
        print(f"MirroredStrategy unavailable ({type(e).__name__}: {str(e)[:60]}) -> single GPU")
REPLICAS = STRATEGY.num_replicas_in_sync if STRATEGY is not None else 1
PER_REP  = CFG["batch_size_per_replica"]
BS       = PER_REP * REPLICAS
JIT      = bool(CFG["jit"]) and STRATEGY is None      # XLA + MirroredStrategy has friction
print(f"replicas {REPLICAS} | global batch {BS} ({PER_REP}/replica) | XLA {JIT}")

def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

CLASSES = list(CFG["classes"])
if len(set(CLASSES)) != len(CLASSES):
    dupes = sorted({c for c in CLASSES if CLASSES.count(c) > 1})
    raise SystemExit(f"CFG['classes'] has duplicates: {dupes}")
CIDX = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)          # class count. Never shadow the KC module or any other name.

def build_alias():
    a = {norm(k): v for k, v in CFG["alias"].items()}
    # narrow brown spot is handled here so merge_narrow_brown is actually wired up
    a["narrowbrownspot"] = "Brown_Spot" if CFG["merge_narrow_brown"] else "Narrow_Brown_Spot"

    # An alias pointing at an undeclared class is a trap. Cell 4 would label those images with
    # a name that is not in CIDX, and the KeyError only surfaces in Cell 8/9 — far from the
    # cause. This is the same class of bug as the old `return name.title()` fallback, so it is
    # caught here, loudly, instead of downstream.
    orphans = {v: k for k, v in a.items() if v not in CIDX}
    if orphans:
        print(f"!! dropping {len(orphans)} alias entr(y/ies) whose class is not in "
              f"CFG['classes']: {sorted(set(orphans))}")
        print("   any images in those folders will be reported as UNMAPPED in Cell 4, not "
              "mislabelled. To keep them, add the class name to CFG['classes'].")
        a = {k: v for k, v in a.items() if v in CIDX}
    return a

ALIAS = build_alias()
SRC_ALIAS = {}
for _src, _m in CFG.get("source_alias", {}).items():
    _d = {norm(k): v for k, v in _m.items()}
    _orph = {k: v for k, v in _d.items() if v not in CIDX}
    if _orph:
        print(f"!! dropping {len(_orph)} source_alias entr(y/ies) for {_src} whose class is "
              f"not in CFG['classes']: {sorted(set(_orph.values()))}")
        _d = {k: v for k, v in _d.items() if v in CIDX}
    SRC_ALIAS[_src] = _d
for _src, _d in SRC_ALIAS.items():
    print(f"source_alias[{_src}]: {_d or 'empty'}")

print(f"taxonomy ({NC}): {CLASSES}")
print(f"alias: {len(ALIAS)} folder patterns -> {sorted(set(ALIAS.values()))}")
print(f"narrow brown spot -> {'Brown_Spot (merged)' if CFG['merge_narrow_brown'] else 'its own label'}")
# Cross-check: nothing may map to a class that is not declared, and every declared class that
# has an alias must be reachable. Cheap, and it is the exact bug that shipped last time.
assert set(ALIAS.values()) <= set(CLASSES), "alias leaked an undeclared class"
for _d in SRC_ALIAS.values():
    assert set(_d.values()) <= set(CLASSES), "source_alias leaked an undeclared class"
# A source_alias that shadows a global alias with a DIFFERENT class is almost always a typo,
# and it would silently win in Cell 4. Catch it here instead of in a confusion matrix.
for _src, _d in SRC_ALIAS.items():
    for _k, _v in _d.items():
        if _k in ALIAS and ALIAS[_k] != _v:
            raise SystemExit(f"source_alias[{_src}]['{_k}'] -> {_v} shadows the global alias "
                             f"-> {ALIAS[_k]}. Remove one of them.")

def elapsed(): return (time.time() - T0) / 60.0










# %%
# CELL 3 — fetch sources
# Resolution order for type "auto":
#   1. the explicit `mount` path            -> free, costs no /kaggle/working space
#   2. any /kaggle/input dir matching the slug tail
#   3. `kaggle datasets download`, GATED on free disk
#
# Why the gate exists: /kaggle/working is ~20 GB and shayanriyaz is an 8.1 GB zip that
# expands to another ~8 GB. Without the gate the download fills the volume, the partial file
# is left behind, and every later cell dies with Errno 28 — including a 3 KB CSV write.
# A mounted dataset has none of that cost, so Add Input is always the better path here.
def _n_images(d): return sum(1 for _ in d.rglob("*") if _.suffix.lower() in IMG_EXT)

def _first_images(d):
    """Walk down to the first directory that actually holds images.
    We deliberately return the SOURCE ROOT, not the image dir: Cell 4 resolves classes from
    each file's ancestor chain relative to this root, which works for any nesting depth."""
    d = Path(d)
    if not d.is_dir(): return d
    if _n_images(d): return d
    for c in sorted(d.iterdir()):
        if c.is_dir() and _n_images(c): return c
    return d

def _mounted(slug):
    """Find an Add Input mount whose directory name matches the slug tail."""
    want = slug.split("/")[-1].lower()
    base = Path("/kaggle/input")
    if not base.is_dir(): return None
    for d in base.iterdir():
        if d.is_dir() and want in d.name.lower() and _n_images(d):
            print(f"   Add Input mount '{d.name}' ({_n_images(d)} imgs)")
            return _first_images(d)
    return None

def _cli_download(src):
    """Download + extract one slug. Returns the extraction root.

    Three deliberate choices, each a fix for a failure we actually hit:
      * PREFLIGHT — refuse before writing anything if the zip plus its expansion cannot fit;
      * DELETE THE ZIP — peak usage is zip + extracted, so removing the zip right after
        extraction halves peak disk. The old code left the zip sitting next to the copy;
      * CLEAN UP ON FAILURE — remove the partial tree, so one bad download cannot make every
        subsequent cell fail with Errno 28."""
    slug = src["slug"]
    out = WORK / ("_dl_" + slug.split("/")[-1])
    if (out / "_ok").exists():
        return _first_images(out / "x")

    if not CFG.get("allow_cli_download", True):
        raise RuntimeError("CLI download disabled (CFG['allow_cli_download'] = False)")
    zip_gb = float(src.get("zip_gb") or 0.0)
    # Peak = zip on disk + extracted tree. JPEGs barely recompress, so extracted ~= zip.
    need = zip_gb * 2 + 0.5
    have = _free_gb()
    if zip_gb and have < need:
        raise RuntimeError(
            f"refusing to download {zip_gb:.1f} GB into {have:.1f} GB free "
            f"(needs ~{need:.1f} GB for zip + extraction).\n"
            f"  FIX (best): Add Input -> attach '{slug.split('/')[-1]}'. Mounted datasets cost\n"
            f"       no /kaggle/working space at all, and Cell 3 detects it automatically.\n"
            f"  ALTERNATIVELY: drop this source from CFG['sources'] — check Cell 7 for which\n"
            f"       classes it was the only source for (here: Hispa).")

    # Force the CLI to stage through WORK, not the system temp dir. On Kaggle /tmp is a much
    # smaller filesystem than /kaggle/working, and the kaggle client downloads through a temp
    # file — which is why an 8 GB dataset failed with Errno 28 while WORK still had 19.5 GB
    # free. Pointing TMPDIR/TMP/TEMP at WORK makes the preflight above meaningful.
    staging = WORK / "_tmp"
    staging.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"TMPDIR": str(staging), "TMP": str(staging), "TEMP": str(staging)})

    try:
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["kaggle", "datasets", "download", "-d", slug,
                            "-p", str(out), "-q"], capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout or "").strip()[:200])

        zips = list(out.glob("*.zip"))
        if not zips:
            raise RuntimeError("download reported success but produced no .zip")
        (out / "x").mkdir(parents=True, exist_ok=True)
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out / "x")
            z.unlink()                       # <-- halves peak disk; the only copy left is the tree
        (out / "_ok").write_text("ok")
        return _first_images(out / "x")
    except OSError as e:
        shutil.rmtree(out, ignore_errors=True)
        # An OSError from the CLI arrives as a bare "[Errno 28] No space left on device",
        # which reads like the working volume is full. It is not: /kaggle/working still has
        # free space, so the CLI is writing somewhere we cannot see or size. Say so.
        raise RuntimeError(
            f"{type(e).__name__} raised while running the kaggle CLI: {e}\n"
            f"  NOT the working volume: /kaggle/working still has {_free_gb():.1f} GB free,\n"
            f"  and TMPDIR was already redirected at it. The limit is a Kaggle-side quota or a\n"
            f"  CLI staging path outside our control.\n"
            f"  FIX: attach '{slug.split('/')[-1]}' via Add Input. A mounted dataset costs zero\n"
            f"  working disk and bypasses the CLI entirely — Cell 3 finds it automatically.\n"
            f"  ALTERNATIVELY: remove this source from CFG['sources'] to proceed without it.")
    except Exception:
        shutil.rmtree(out, ignore_errors=True)   # never leave a half-written tree behind
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

def fetch(src):
    kind = src.get("type", "auto")
    slug = src.get("slug")

    if kind in ("local", "auto") and src.get("mount"):
        p = Path(src["mount"])
        if p.is_dir() and _n_images(p):
            print(f"   mount '{p}' ({_n_images(p)} imgs)")
            return _first_images(p)
        if kind == "local":
            raise FileNotFoundError(
                f"{p} not found — add it via Kaggle 'Add Input', or set type 'auto'/'kaggle'")

    if not slug:
        raise RuntimeError(f"source {src.get('name')} has neither 'mount' nor 'slug'")
    m = _mounted(slug)
    if m is not None:
        return m
    if kind == "local":
        raise FileNotFoundError(f"no mount for '{slug}' — attach it via Kaggle 'Add Input'")
    return _cli_download(src)

ROOTS, FAILED = [], []
for s in CFG["sources"]:
    t = time.time()
    try:
        r = fetch(s)
        ROOTS.append({**s, "path": r})
        how = "mount" if "/kaggle/input" in str(r) else "cli"
        print(f"[ok]   {s['name']:11s} {_n_images(r):6d} imgs  [{how}]  {r}  ({time.time()-t:.0f}s)")
    except Exception as e:
        FAILED.append(s["name"]); print(f"[FAIL] {s['name']:11s} {e}")
assert ROOTS, "no datasets available — check CFG['sources'], Add Input, and Internet"

left = _free_gb()
print(f"\n{len(ROOTS)} source(s) ready; failed: {FAILED or 'none'}")
print(f"disk left: {left:.1f} GB")
# Catch the state that produced Errno 28 in Cell 4 BEFORE it happens, while we still know
# which download caused it and can still suggest Add Input instead of a full restart.
if left < 1.0:
    which = [s["name"] for s in ROOTS if "/kaggle/input" not in str(s["path"])]
    raise SystemExit(
        f"only {left:.1f} GB free on /kaggle/working — later cells will fail with Errno 28.\n"
        f"  CLI-downloaded sources: {which or 'none'}\n"
        f"  FIX: attach those via Kaggle 'Add Input' (free), or remove them from CFG['sources'].\n"
        f"  Then restart the session to clear the partial downloads.")










# %%
# CELL 4 — manifest
# Class comes from the NEAREST ancestor folder that hits an alias. Works for
# Rice_Leaf_AUG/<Class>/, train/<Class>/, validation/<Class>/, and arbitrary nesting.
#
# Resolution order per folder, first hit wins:
#   1. SRC_ALIAS[source]  — per-source overrides, for names only safe in context
#   2. ALIAS              — the global table
# Unresolvable images are REPORTED under UNMAPPED, never given an invented label.
rows, unmapped = [], defaultdict(int)
via_counts = defaultdict(int)

for r in ROOTS:
    root = r["path"]
    sal = SRC_ALIAS.get(r["name"], {})

    for f in root.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in IMG_EXT): continue
        parts = f.relative_to(root).parts[:-1]
        cls = via = None
        for d in reversed(parts):
            k = norm(d)
            if k in sal:
                cls, via = sal[k], f"source_alias[{r['name']}]"
                break
            if k in ALIAS:
                cls, via = ALIAS[k], "alias"
                break
        if cls is None:
            unmapped[f"{r['name']}:{'/'.join(parts) or '<root>'}"] += 1
        else:
            rows.append({"path": str(f), "class": cls, "source": r["name"], "via": via})
            via_counts[via] += 1

man = pd.DataFrame(rows)
if man.empty:
    raise SystemExit("manifest empty — CFG['alias'] does not match your folder names")
if CFG["smoke"]:
    man = man.groupby("class").head(40).reset_index(drop=True)
    print(f"SMOKE: manifest capped to {len(man)} images (40/class)")
man.to_csv(OUT / "manifest_raw.csv", index=False)

# source-held-out: route the held-out source OUT of training. It is evaluated separately in
# Cell 16.5 (the headline metric) and never touches train/val/test or the dedupe.
man_held = None
if CFG.get("held_out_source"):
    held_name = CFG["held_out_source"]
    mh = man["source"] == held_name
    if mh.any():
        man_held = man[mh].reset_index(drop=True)
        man = man[~mh].reset_index(drop=True)
        print(f"source-held-out: {held_name} -> {len(man_held)} images held out; "
              f"{len(man)} images remain for train/val/test")
    else:
        print(f"WARNING: held_out_source '{held_name}' not in the manifest — nothing held out")

print(f"manifest: {len(man)} images from {man['source'].nunique()} source(s)\n")
if unmapped:
    print("!! UNMAPPED folders — no label was invented. Add the folder to CFG['alias'], or")
    print("   CFG['source_alias'] if the name is only safe for one source:")
    for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:25]:
        print(f"     {v:6d}  {k}")
    print(f"     ({sum(unmapped.values())} images excluded from training)")
    print()
print("resolved by: " + ", ".join(f"{k}={v}" for k, v in sorted(via_counts.items())))
print()
print(man.groupby(["source", "class"]).size().unstack(fill_value=0).to_string())

# Per-class source coverage. A class backed by ONE source is the weakest evidence in the set:
# if that source's labels are wrong the class is wrong, and no metric here would say so.
_cov = man.groupby("class")["source"].nunique()
_single = sorted(_cov[_cov == 1].index)
print(f"\nclasses backed by a single source: {_single or 'none'}")
for c in _single:
    print(f"   {c:<24} <- {sorted(man[man['class'] == c]['source'].unique())}")










# %%
# CELL 5 — dihedral-aware grouping (KEEP the duplicates)
# pHash is not flip/rotate invariant, and `Rice_Leaf_AUG` is exactly flips and rotations.
# So we hash all 8 dihedral (D4) variants and use the MINIMUM as the canonical key: two
# images that are rotations/reflections of each other produce the same key.
#
# We then KEEP every image and assign each one its cluster id. Dropping duplicates here
# would collapse every cluster to a singleton and make the grouped split in Cell 6 vacuous.
import imagehash

def _dihedral(im):
    yield im
    for t in (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270):
        yield im.transpose(t)
    fl = im.transpose(Image.FLIP_LEFT_RIGHT)
    yield fl
    for t in (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270):
        yield fl.transpose(t)

def _d4_keys(p, probe=256):
    # All 8 D4-variant pHashes, not the min: the brute-force merge below takes the MIN over
    # the 8x8 variant pairs, which is strictly more information than a single canonical key.
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if probe:
                # draft() lets libjpeg do a SCALED decode — this is what makes preloading the
                # 2.4 MB shayanriyaz files tolerable at all.
                try: im.draft("RGB", (probe, probe))
                except Exception: pass
                im = im.resize((probe, probe), Image.BILINEAR)
            return tuple(int(str(imagehash.phash(v)), 16) for v in _dihedral(im))
    except Exception:
        return None

def d4_dist_matrix(keys8, chunk=256):
    """Full pairwise min-over-D4-variants Hamming matrix, (n, n) uint8.

    keys8: (n, 8) uint64. D[i, j] = min over a, b of popcount(keys8[i,a] ^ keys8[j,b]).
    Chunked so the working set is (chunk, n) uint8, not (n, n) at once.
    """
    n = len(keys8)
    D = np.zeros((n, n), np.uint8)
    for q0 in range(0, n, chunk):
        q = keys8[q0:q0 + chunk]
        best = np.full((len(q), n), 255, np.uint8)
        for a in range(8):
            qa = q[:, a][:, None]
            for b in range(8):
                d = np.bitwise_count(np.bitwise_xor(qa, keys8[:, b][None, :]))
                np.minimum(best, d, out=best)
        D[q0:q0 + chunk] = best
    return D

def brute_clusters(D, nd, labels):
    """Union-find over SAME-CLASS pairs with D <= nd. Returns (cluster_ids, n_merged).

    Cross-class near-duplicates are NOT merged: they are label noise, and merging them would
    fold contradictory labels into one cluster. They are flagged separately by cross_class_mask.
    """
    n = len(D)
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    merged = 0
    for i in range(n):
        js = np.nonzero((D[i] <= nd) & (labels == labels[i]))[0]
        for j in js:
            if j <= i:
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri; merged += 1
    out = np.array([find(i) for i in range(n)], np.int64)
    return out, merged

def cross_class_mask(D, nd, labels):
    """True for images within nd of a DIFFERENT-class image (ambiguous / mislabelled)."""
    n = len(D)
    mask = np.zeros(n, bool)
    for i in range(n):
        mask[i] = bool(((D[i] <= nd) & (labels != labels[i])).any())
    return mask

def grouped_split(man, sp, seed):
    """Stratified group split. Defined here, not in Cell 6, so Cell 5 and Cell 6 share one
    implementation (a second copy is how the two halves drift apart).

    Two properties matter, and the previous version had neither it claimed:
      * per-CLASS ratios, not just the global total. Global bin packing alone gave
        Brown_Spot 1404/59/59 and Sheath_Blight 230/200/202 on the real data, while the global
        sums looked perfect — so macro-F1 described the split, not the model.
      * the seed must do something. `rng` used to be created and never used.
    """
    rng = np.random.RandomState(seed)
    classes = sorted(man["class"].unique())
    comp = (man.groupby(["cluster", "class"]).size()
              .unstack(fill_value=0).reindex(columns=classes, fill_value=0))
    n_cls = comp.sum()
    target = pd.DataFrame({s: n_cls * v for s, v in sp.items()})
    got = pd.DataFrame(0.0, index=classes, columns=list(sp), dtype=float)
    assign = {}
    for c in sorted(classes, key=lambda x: n_cls[x]):          # rarest class first
        cand = comp[c][comp[c] > 0]
        if not len(cand):
            continue
        idx = np.lexsort((rng.rand(len(cand)), -cand.to_numpy()))   # biggest cluster first
        for i in idx:
            cid = cand.index[i]
            if cid in assign:                 # already claimed by a rarer class
                continue
            best = min(sp, key=lambda s: (got.at[c, s] / max(target.at[c, s], 1e-9), rng.random()))
            assign[cid] = best
            got.at[c, best] += float(cand.iloc[i])
    out = man.copy()
    out["split"] = out["cluster"].map(assign)
    assert out["split"].notna().all(), "some cluster was never assigned a split"
    return out, got.sum(axis=0).to_dict()

def leak_scan(D, man, tight=7):
    """Count near-duplicate pairs that ended up in DIFFERENT splits.

    Uses the brute-force D4 distance matrix from Cell 5 (no re-hashing). A pair within `tight`
    bits that straddles splits is a leak the merge missed.
    """
    sp_of = man["split"].to_numpy()
    leaks = defaultdict(list)
    for i in range(len(man)):
        js = np.nonzero((D[i] <= tight) & (sp_of != sp_of[i]))[0]
        for j in js:
            if j <= i:
                continue
            leaks[tuple(sorted((sp_of[i], sp_of[j])))].append((i, j))
    return leaks

if CFG["dedupe"]["enable"]:
    # Hash all 8 D4 variants fresh every run. The old on-disk cache was removed on purpose:
    # it could silently serve stale keys after a manifest change, and hashing is only ~84 s
    # for the full set (far less in smoke).
    t = time.time()
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        keys8 = list(ex.map(_d4_keys, man["path"].tolist()))
    ok = np.array([k is not None for k in keys8])
    print(f"hashed {ok.sum()}/{len(ok)} in {time.time()-t:.0f}s (8 D4 variants/img)")
    man = man[ok].reset_index(drop=True)
    keys8 = np.array([k for k, m in zip(keys8, ok) if m], np.uint64)

    nd = CFG["dedupe"]["near_dist"]
    labels = man["class"].astype("category").cat.codes.to_numpy(np.int32)
    t = time.time()
    D = d4_dist_matrix(keys8)
    man["cluster"], _merged = brute_clusters(D, nd, labels)
    cross_mask = cross_class_mask(D, nd, labels)
    print(f"\nbrute-force D4 dedupe in {time.time()-t:.0f}s: {_merged} same-class key pairs merged "
          f"within Hamming <= {nd} (min over 8 D4 variants)")
    print(f"cross-class near-duplicates: {int(cross_mask.sum())} images "
          f"({cross_mask.mean():.1%}) — excluded from val/test in Cell 6")
else:
    man["cluster"] = np.arange(len(man))
    D = None
    cross_mask = None

# ---- diagnostics that matter ----
sizes = man.groupby("cluster").size()
multi = sizes[sizes > 1]
xsrc = man.groupby("cluster")["source"].nunique()
n_before = len(man)
print(f"\nclusters: {man['cluster'].nunique()}  |  multi-member: {len(multi)}  "
      f"|  images inside multi-member clusters: {int(multi.sum())} "
      f"({multi.sum()/len(man):.0%})")
print(f"images folded into an existing cluster: {n_before - man['cluster'].nunique()} "
      f"({(n_before - man['cluster'].nunique())/n_before:.0%} of the manifest)")
print(f"largest cluster: {int(sizes.max())} images  |  median {int(sizes.median())}")
# Same-class merge means clusters can never span classes; cross-class collisions live in
# cross_mask instead (reported below).
print(f"clusters spanning >1 source: {int((xsrc>1).sum())}  "
      f"(cross-source copies = the datasets overlap)")

# A cluster of ~60 that is single-linkage chaining would show here. Largest is the number to
# watch: if it climbs into the hundreds, lower CFG["dedupe"]["near_dist"] to 0.
if sizes.max() > 10:
    big = man[man["cluster"] == sizes.idxmax()]
    comp = ", ".join(f"{k}={v}" for k, v in big["class"].value_counts().items())
    srcs = ", ".join(f"{k}={v}" for k, v in big["source"].value_counts().items())
    print(f"largest cluster composition: {comp}   (sources: {srcs})")

# WHERE the label noise comes from. This is also the empirical check on CFG["source_alias"]:
# if indo3's `blight` folder were really Leaf Blast, its images would be near-duplicates of
# Leaf_Blast images and that pair would show up here. Absence is not proof, but a pile of
# indo3:blight <-> <something else> pairs would be a red flag worth acting on.
if cross_mask is not None and cross_mask.any():
    pairs = defaultdict(int)
    for i in np.nonzero(cross_mask)[0]:
        js = np.nonzero((D[i] <= nd) & (labels != labels[i]))[0]
        for j in js:
            if j <= i:
                continue
            a = man["source"].iat[i] + ":" + man["class"].iat[i]
            b = man["source"].iat[j] + ":" + man["class"].iat[j]
            pairs[tuple(sorted((a, b)))] += 1
    print(f"\ncross-class near-duplicates: {int(cross_mask.sum())} images "
          f"({cross_mask.mean():.1%}) — excluded from val/test in Cell 6")
    print("which source:class pairs collide:")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]:
        flag = ""
        if "indo3" in a and "indo3" in b:
            flag = "   <- both from indo3: within-source label disagreement"
        elif ("indo3" in a) != ("indo3" in b):
            flag = "   <- involves indo3 (check the source_alias mapping)"
        print(f"   {n:4d} {a}  <->  {b}{flag}")
man.to_csv(OUT / "manifest_grouped.csv", index=False)










# %%
# CELL 6 — grouped stratified split
# `grouped_split`, `leak_scan` and the brute-force dedupe helpers are defined in CELL 5, so
# this cell uses the exact same implementations. Do not re-define them here: a second copy is
# how the two halves drift apart.

sp = CFG["split"]
man, got = grouped_split(man, sp, SEED)          # defined in Cell 5, shared with this cell

# Cross-class near-duplicates carry unreliable labels (at least one of the pair is mislabelled),
# so they must not be scored. Keep them in TRAIN (the model averages the noise) but move them
# out of val/test. This is the audit's "cross-class dropped from val/test".
if cross_mask is not None:
    moved = cross_mask & (man["split"] != "train")
    n_moved = int(moved.sum())
    man.loc[moved, "split"] = "train"
    got = {k: int((man["split"] == k).sum()) for k in sp}
    print(f"cross-class near-duplicates: {int(cross_mask.sum())} images; "
          f"{n_moved} moved from val/test to train (unreliable labels)")

print("split sizes: " + ", ".join(f"{k}={int(got[k])}" for k in sp)
      + f"  (target {int(sp['train']*len(man))}/{int(sp['val']*len(man))}/{int(sp['test']*len(man))})")
tab = man.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=list(sp))
print("\nper-class x per-split counts:")
print(tab.to_string())

# The check that would have caught the un-stratified split: every class must land near its
# own target ratio, not merely sum to the right total.
print("\nper-class deviation from target ratio:")
worst = 0.0
for c in tab.index:
    n = int(tab.loc[c].sum())
    dev = max(abs(int(tab.loc[c, s]) - n * sp[s]) / n for s in sp)
    worst = max(worst, dev)
    flag = ""
    if dev > 0.05:
        flag = "   <- OFF TARGET"
    dead = [s for s in sp if int(tab.loc[c, s]) == 0]
    if dead:
        flag += f"   <- NO SAMPLES in {dead}: its metric is undefined"
    thin = [s for s in sp if 0 < int(tab.loc[c, s]) < 50]
    if thin:
        flag += f"   <- thin: {thin}"
    print(f"  {c:24s} n={n:5d}  max dev {dev:5.1%}{flag}")
print(f"  worst per-class deviation: {worst:.1%}")
if worst > 0.05:
    print("  WARNING: per-class ratios are off. Do not trust val/test metrics - they are")
    print("           measuring the split, not the model.")
if any(int(tab.loc[c, s]) == 0 for c in tab.index for s in sp):
    print("  WARNING: a class has no val or no test samples. Grouping won over stratification,")
    print("           which is correct (splitting the cluster would leak), but that class's")
    print("           macro-F1 contribution is undefined and the run must be redone with a")
    print("           different seed.")

sets = {k: set(man.loc[man["split"] == k, "cluster"]) for k in sp}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = sets[a] & sets[b]
    print(f"  cluster overlap {a}/{b}: {len(overlap)}")
    assert not overlap, f"LEAK: {len(overlap)} clusters in both {a} and {b}"
if man["cluster"].nunique() == len(man):
    print("  NOTE: every cluster is a singleton - grouping was inert, the leak assert is weak")
print("  NOTE: the overlap check above is STRUCTURAL. Split is assigned per cluster, so")
print("        overlap is impossible by construction - that assert cannot fail and proves")
print("        nothing on its own. The scan below is the one that can actually fail.")

# ---- the leakage check that can actually fail ----
# Cell 5 merged same-class near-duplicates at dedupe.near_dist on the D4-min pHash. Images
# that are visually near-identical but pHash-different (re-cropped, re-encoded, rotated
# off-grid) never got merged, so they CAN straddle train and val/test. Scan for exactly that
# at a deliberately TIGHTER threshold. Reuses the brute-force D4 distance matrix, so there is
# no re-hashing cost.
TIGHT = 7
_csize = man.groupby("cluster").size()
if CFG["dedupe"]["enable"] and D is not None and D.shape[0] == len(man):
    leaks = leak_scan(D, man, tight=TIGHT)
    n_pairs = sum(len(v) for v in leaks.values())
    print(f"\ncross-split near-duplicate scan (hamming <= {TIGHT}, tighter than the merge "
          f"threshold of {CFG['dedupe']['near_dist']}):")
    if n_pairs:
        n_img = len({i for ps in leaks.values() for i, _ in ps})
        print(f"  {n_pairs} near-duplicate pair(s) straddle splits, touching {n_img} images "
              f"({n_img/len(man):.1%} of the manifest):")
        for pair, ps in sorted(leaks.items(), key=lambda kv: -len(kv[1])):
            i, j = ps[0]
            # .loc, NOT .iloc: this Series is indexed by CLUSTER LABEL, and after union-find
            # merging those labels are sparse (0, 1, 5, 17, ...) — passing them positionally
            # raises IndexError as soon as a label exceeds the group count. Synthetic tests with
            # contiguous ids hid this completely; real cluster ids do not.
            cs = _csize.loc[[man["cluster"].iat[i], man["cluster"].iat[j]]].tolist()
            print(f"    {len(ps):5d}  {pair[0]}/{pair[1]}   e.g. "
                  f"{man['class'].iat[i]} vs {man['class'].iat[j]}  (cluster sizes {cs})")
    else:
        print("  0 - no train image is a near-duplicate of any val/test image  OK")
else:
    print("\ncross-split near-duplicate scan SKIPPED (no canonical keys available)")










# %%
# CELL 7 — enforce min_class on TRAIN, re-splitting if a class is starved
# Must run AFTER the split: a class can clear the raw-count bar and still vanish from a split.
def starved_classes(m, classes, min_class):
    tr = m[m["split"] == "train"]["class"].value_counts()
    bad = []
    for c in classes:
        n = int(tr.get(c, 0))
        missing = [k for k in sp if c not in set(m.loc[m["split"] == k, "class"])]
        if n < min_class or missing:
            bad.append((c, n, missing))
    return bad

CLASSES = list(CFG["classes"])
for attempt in range(4):
    keep = [c for c in CLASSES if c in set(man["class"])]
    gone = [c for c in CLASSES if c not in keep]
    if gone: print(f"dropped (no images at all): {gone}")
    man = man[man["class"].isin(keep)]
    CLASSES, CIDX, NC = keep, {c: i for i, c in enumerate(CLASSES)}, len(keep)
    man, _ = grouped_split(man, sp, SEED)
    bad = starved_classes(man, CLASSES, CFG["min_class"])
    if not bad:
        print(f"\nmin_class OK: every class has >= {CFG['min_class']} TRAIN images "
              f"and appears in all {len(sp)} splits")
        break
    print(f"\nattempt {attempt+1}: starving -> " + ", ".join(
        f"{c} (train={n}{', missing from ' + ','.join(ms) if ms else ''})" for c, n, ms in bad))
    drop = {c for c, _, _ in bad}
    CLASSES = [c for c in CLASSES if c not in drop]
else:
    raise SystemExit("could not satisfy min_class — lower it or add a data source")

tr, va, te = (man[man["split"] == k].reset_index(drop=True) for k in ("train", "val", "test"))
print(f"\nfinal: train {len(tr)} | val {len(va)} | test {len(te)}")
print(tr.groupby("class").size().reindex(CLASSES).to_frame("train_n").to_string())
man.to_csv(OUT / "manifest_split.csv", index=False)










# %%
# CELL 8 — class weights from the manifest
# NB: never touch a prefetched dataset for this — that is what raised
# `'_PrefetchDataset' object has no attribute 'class_names'`.
cnt = tr["class"].value_counts().reindex(CLASSES).to_numpy(float)
N = cnt.sum()
mode = CFG["imbalance"]["mode"]
assert mode == "class_weight", f"only class_weight is supported (got {mode!r})"

w = N / (NC * cnt)
w = w / w.mean()
CLS_W = tf.constant(w, dtype=tf.float32)
print(f"imbalance mode: {mode}  |  majority/minority {cnt.max()/max(cnt[cnt>0].min(),1):.1f}x")
print(pd.DataFrame({"class": CLASSES, "train_n": cnt.astype(int), "weight": w.round(3)}).to_string())

LOSS = lambda: tf.keras.losses.CategoricalCrossentropy()










# %%
# CELL 9 — preload to a RAM array (the speed core)
# 8 GB of high-res JPEG -> ~650 MB of 256px uint8. Decoded once, never again.
def preload(df, pre=PRE, workers=None):
    workers = workers or CFG["workers"]
    paths = df["path"].tolist()
    arr = np.zeros((len(paths), pre, pre, 3), np.uint8)
    ok = np.ones(len(paths), bool)
    def work(i):
        try:
            with Image.open(paths[i]) as im:
                im.draft("RGB", (pre, pre))      # scaled JPEG decode: 5-10x faster
                arr[i] = np.asarray(im.convert("RGB").resize((pre, pre), Image.BILINEAR),
                                    np.uint8)
        except Exception:
            ok[i] = False; arr[i] = 0
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(work, range(len(paths))): pass
    arr, df = arr[ok], df[ok].reset_index(drop=True)
    print(f"  {len(arr)} imgs -> {arr.nbytes/1e9:.2f} GB in {time.time()-t:.0f}s "
          f"({arr.nbytes/max(len(arr),1)/1e6:.0f} KB/img)"
          + (f"  ({int((~ok).sum())} unreadable dropped)" if not ok.all() else ""))
    return arr, df

# budget covers train + val + test together
per_mb = PRE * PRE * 3 / 1e6
need_gb = len(man) * per_mb / 1000
budget = CFG["ram_frac"] * RAM_AVAIL_GB
USE_RAM = CFG["preload_ram"] and need_gb <= budget
print(f"{len(man)} imgs x {per_mb*1000:.0f} KB = {need_gb:.2f} GB vs budget {budget:.2f} GB "
      f"-> {'RAM preload' if USE_RAM else 'disk stream'}")

if USE_RAM:
    Xtr, tr = preload(tr)
    Xva, va = preload(va)
    Xte, te = preload(te)
else:
    Xtr = Xva = Xte = None
ytr = np.array([CIDX[c] for c in tr["class"]], np.int32)
yva = np.array([CIDX[c] for c in va["class"]], np.int32)
yte = np.array([CIDX[c] for c in te["class"]], np.int32)

# (oversample was removed on purpose — class_weight only, see Cell 8)

# Once the pixels are in RAM, the on-disk copies are dead weight — ~10 GB of it. /kaggle/working
# is ~20 GB and the Keras/TFLite exports plus checkpoints need that space later. Only safe when
# USE_RAM, because the disk-stream path still reads from these paths in Cell 13.
if USE_RAM:
    by_name = {r["name"]: str(r["path"]) for r in ROOTS}
    freed = 0.0
    for s in CFG["sources"]:
        if "/kaggle/input" in by_name.get(s["name"], ""):
            continue                                    # a mount, not ours to delete
        dl = WORK / ("_dl_" + s["slug"].split("/")[-1]) if s.get("slug") else None
        if dl and dl.is_dir():
            freed += _dir_gb(dl)
            shutil.rmtree(dl, ignore_errors=True)
    if freed > 0.1:
        print(f"freed {freed:.1f} GB of CLI downloads after preload -> {_free_gb():.1f} GB free")










# %%
# CELL 10 — input pipelines
# Two things this deliberately avoids:
#   * dataset-level .shuffle() on images: a 20k buffer of 196 KB images is ~3.9 GB. We
#     permute INDICES in Python instead, so the shuffle buffer holds ints.
#   * from_tensor_slices(X): that embeds the array as a graph constant and trips the 2 GB
#     protobuf limit. from_generator keeps X in host memory, outside the graph.
def _rrc(im):
    """Random resized crop via sample_distorted_bounding_box (tf.image has no
    random_resized_crop). Returns float32 0..255 at SIZE."""


    a = CFG["aug"]
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(im), bounding_boxes=tf.zeros([1, 0, 4], tf.float32),
        area_range=a["rrc_scale"], aspect_ratio_range=(3/4., 4/3.),
        max_attempts=10, use_image_if_no_bounding_boxes=True)
    return tf.image.resize(tf.slice(im, begin, size), (SIZE, SIZE))

def _aug(im, y):
    """All colour ops run in 0..1, which is what adjust_hue/adjust_saturation assume."""
    a = CFG["aug"]
    h = _rrc(im) / 255.0
    if a["flip_h"]: h = tf.image.random_flip_left_right(h)
    if a["flip_v"]: h = tf.image.random_flip_up_down(h)
    h = tf.image.random_brightness(h, a["brightness"])
    h = tf.image.random_contrast(h, 1 - a["contrast"], 1 + a["contrast"])
    h = tf.image.random_saturation(h, 1 - a["sat"], 1 + a["sat"])
    h = tf.image.random_hue(h, a["hue"])
    h = tf.clip_by_value(h, 0., 1.) * 255.
    h.set_shape([SIZE, SIZE, 3])             # dynamic crop size must not leak unknown H/W
    return h, tf.one_hot(y, NC)

def _eval_t(im, y):
    # PRE (256) -> SIZE (224): the model input is fixed at SIZE, so eval MUST resize.
    # Also cast: resize on a uint8 tensor returns uint8, but the model wants float32.
    h = tf.cast(tf.image.resize(im, (SIZE, SIZE)), tf.float32)
    h.set_shape([SIZE, SIZE, 3])
    return h, tf.one_hot(y, NC)

rng_ds = np.random.RandomState(SEED + 1)
if USE_RAM:
    n_tr = len(ytr)
    def gen_train():
        while True:
            for i in rng_ds.permutation(n_tr):
                yield Xtr[i], ytr[i]
    def gen_eval(X, y):
        for i in range(len(y)):
            yield X[i], y[i]
    train_ds = tf.data.Dataset.from_generator(
        gen_train, output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                     tf.TensorSpec([], tf.int32)))
    train_ds = train_ds.map(_aug, num_parallel_calls=AUTOTUNE).batch(BS).prefetch(AUTOTUNE)
    def mk_eval(X, y):
        return (tf.data.Dataset.from_generator(
                    lambda X=X, y=y: gen_eval(X, y),
                    output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                      tf.TensorSpec([], tf.int32)))
                .map(_eval_t, num_parallel_calls=AUTOTUNE).batch(BS*2).prefetch(AUTOTUNE))
    val_ds, test_ds = mk_eval(Xva, yva), mk_eval(Xte, yte)
else:
    def dec(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.cast(tf.image.resize(img, (PRE, PRE), method="bilinear",
                                       antialias=True), tf.uint8)
    n_tr = len(ytr)
    def gen_train():
        paths = tr["path"].tolist()
        while True:
            for i in rng_ds.permutation(n_tr):
                yield dec(paths[i]), ytr[i]
    train_ds = (tf.data.Dataset.from_generator(
                    gen_train, output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                                 tf.TensorSpec([], tf.int32)))
                .map(_aug, num_parallel_calls=AUTOTUNE).batch(BS).prefetch(AUTOTUNE))
    def mk_eval(df, y):
        paths = df["path"].tolist()
        return (tf.data.Dataset.from_generator(
                    lambda p=paths, yy=y: ((dec(q), yy[i]) for i, q in enumerate(p)),
                    output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                      tf.TensorSpec([], tf.int32)))
                .map(_eval_t, num_parallel_calls=AUTOTUNE).batch(BS*2).prefetch(AUTOTUNE))
    val_ds, test_ds = mk_eval(va, yva), mk_eval(te, yte)

STEPS = int(np.ceil(n_tr / BS))
print(f"pipeline {'RAM-preloaded' if USE_RAM else 'disk stream'} | {STEPS} steps/epoch | "
      f"train {n_tr} | val {len(yva)} | test {len(yte)} | one-hot targets (NC={NC})")










# %%
# CELL 11 — graph-safe macro-F1
# result() is traced into the compiled graph, so it must be pure TF ops — numpy here fails.
# get_config is required for the metric to survive model.save / load_model.
@tf.keras.utils.register_keras_serializable(package="agrisense")
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, k, name="macro_f1", dtype=None, **kw):
        super().__init__(name=name, dtype=dtype, **kw)
        self.k = int(k)
        self.cm = self.add_weight(shape=(int(k), int(k)), initializer="zeros",
                                  dtype=tf.float32, name="confusion")
    def update_state(self, y_true, y_pred, sample_weight=None):
        m = tf.math.confusion_matrix(tf.argmax(y_true, -1), tf.argmax(y_pred, -1),
                                     num_classes=self.k, dtype=tf.float32)
        self.cm.assign_add(m)
    def result(self):
        cm = tf.cast(self.cm, tf.float32)
        tp = tf.linalg.diag_part(cm)
        prec = tf.math.divide_no_nan(tp, tf.reduce_sum(cm, axis=0))
        rec  = tf.math.divide_no_nan(tp, tf.reduce_sum(cm, axis=1))
        return tf.reduce_mean(tf.math.divide_no_nan(2 * prec * rec, prec + rec))
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))
    def get_config(self):
        c = super().get_config(); c["k"] = self.k; return c

def macro_f1(): return MacroF1(NC)










# %%
# CELL 12 — model + trainability
# training=False on the backbone call is deliberate and does NOT block fine-tuning:
# the `training` arg only switches Dropout and BatchNorm to inference mode. Conv/Dense
# kernels still receive gradients whenever layer.trainable is True. It also guarantees
# BatchNorm never updates its running statistics, which is what you want on 8k images.
def build_model(k):
    inp = tf.keras.Input((SIZE, SIZE, 3), dtype=tf.float32)
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_shape=(SIZE, SIZE, 3), pooling="avg")
    x = base(inp, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    out = tf.keras.layers.Dense(k, activation="softmax", dtype="float32", name="pred")(x)
    return tf.keras.Model(inp, out, name="agrisense_b0")

def base_of(m):
    return next(l for l in m.layers if isinstance(l, tf.keras.Model))

def set_trainable(m, n, verbose=True):
    """Unfreeze n layers of the BACKBONE. The head always trains.
    Walking base.layers is essential: Keras names EfficientNet internals stem_conv /
    block1a_dwconv, so any name-based match on 'efficientnet' finds only the wrapper."""
    base = base_of(m)
    m.trainable = True                                    # head trainable in every stage
    wl = [l for l in base.layers if l.get_weights()]
    keep = [] if n == 0 else (wl[:n] if n > 0 else wl[n:])
    ks = {id(l) for l in keep}
    for l in base.layers:
        l.trainable = id(l) in ks
    n_bb = sum(int(l.trainable) for l in wl)
    head_n = len([l for l in m.layers if l is not base and l.get_weights() and l.trainable])
    if verbose:
        print(f"  backbone {n_bb}/{len(wl)} layers trainable | "
              f"head {head_n} layers trainable | {len(m.trainable_weights)} trainable tensors")
    return n_bb

if STRATEGY is not None:
    with STRATEGY.scope():
        model = build_model(NC)
else:
    model = build_model(NC)
assert model.output_shape[-1] == NC, f"CLASS/OUTPUT MISMATCH {NC} vs {model.output_shape[-1]}"
set_trainable(model, 0, verbose=False)
model.summary()
_bb = len([l for l in base_of(model).layers if l.get_weights()])
print(f"backbone has {_bb} weighted layers; stages B/C unfreeze the last 18 and 70")










# %%
# CELL 13 — callbacks
# ONE global best, written to ONE file. Per-stage counters reset to -1 meant Stage C
# always saved its first epoch regardless of F1.
BEST_PATH = str(OUT / "best.keras")

class SaveBestF1(KC.Callback):
    best = -1.0
    def on_epoch_end(self, epoch, logs=None):
        f1 = logs.get("val_macro_f1", -1.0)
        if f1 > SaveBestF1.best:
            SaveBestF1.best = f1
            self.model.save(BEST_PATH)
            print(f"  ** saved best.keras (val_macro_f1={f1:.4f})")

class BudgetStop(KC.Callback):
    """Caps TRAINING only. T_TRAIN0 is set here, not at notebook start, so downloads,
    grouping and preloading do not eat the budget."""
    def on_epoch_begin(self, epoch, logs=None):
        if (time.time() - T_TRAIN0) / 60.0 > CFG["train_budget_min"]:
            self.model.stop_training = True
            print(f"!! training budget {CFG['train_budget_min']} min hit -> stopping")

def make_cbs(tag):
    return [SaveBestF1(),
            KC.EarlyStopping(monitor="val_macro_f1", mode="max", patience=3, verbose=1),
            KC.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5,
                                 patience=1, min_lr=1e-6, verbose=1),
            KC.CSVLogger(str(OUT / f"log_{tag}.csv"), append=False),
            BudgetStop()]










# %%
# CELL 14 — train
import contextlib
T_TRAIN0 = time.time()
cw = {i: float(w[i]) for i in range(NC)}   # class_weight only (oversample/none removed)

print(f"PIPELINE_VERSION {PIPELINE_VERSION} | training {NC} classes | "
      f"train {len(ytr)} | val {len(yva)} | test {len(yte)}")
for st in CFG["stages"]:
    spent = (time.time() - T_TRAIN0) / 60.0
    if spent > CFG["train_budget_min"]:
        print(f"budget spent ({spent:.1f}m) before {st['name']} -> skipping remaining stages")
        break
    epochs = 1 if CFG["smoke"] else st["epochs"]
    print(f"\n=== {st['name']} | unfreeze last {st['unfreeze']} | {epochs} epochs | "
          f"lr {st['lr']} | spent {spent:.1f}m ===")
    nbb = set_trainable(model, st["unfreeze"])
    expect = 0 if st["unfreeze"] == 0 else abs(st["unfreeze"])
    if st["unfreeze"] == 0:
        assert nbb == 0, "stage A should have a frozen backbone"
    else:
        assert nbb == expect, f"expected {expect} backbone layers trainable, got {nbb}"
    assert len(model.trainable_weights) > len(base_of(model).trainable_weights), \
        "nothing is trainable — the head itself is frozen"

    kw = {"jit_compile": True} if JIT else {}
    # compile INSIDE the strategy scope, or optimizer/metric variables land outside it
    with (STRATEGY.scope() if STRATEGY is not None else contextlib.nullcontext()):
        model.compile(optimizer=tf.keras.optimizers.Adam(st["lr"]),
                      loss=LOSS(), metrics=["accuracy", macro_f1()], **kw)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              steps_per_epoch=STEPS, class_weight=cw,
              callbacks=make_cbs(st["name"]), verbose=2)

print(f"\ntraining wall clock: {(time.time()-T_TRAIN0)/60:.1f} min "
      f"(budget {CFG['train_budget_min']}) | best val macro_f1 seen {SaveBestF1.best:.4f}")

# compile=False: we only need inference, and this avoids rebuilding the custom metric
model = tf.keras.models.load_model(BEST_PATH, compile=False)
assert model.output_shape[-1] == NC, f"CLASS/OUTPUT MISMATCH {NC} vs {model.output_shape[-1]}"
print(f"loaded best.keras | outputs {model.output_shape[-1]} == declared {NC}  OK")










# %%
# CELL 15 — validation report
def predict_idx(ds, n):
    # One predict() over the dataset, not one per batch: Keras 3 takes a tf.data.Dataset
    # directly, and the length assert turns a silently short/long result into a hard error.
    p = np.asarray(model.predict(ds, verbose=0)).argmax(1)
    assert len(p) == n, (len(p), n)
    return p

def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred): cm[int(t), int(p)] += 1
    return cm

def report(cm, title, classes=None):
    classes = classes or CLASSES
    k = len(classes)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec  = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(tp), where=(prec+rec) > 0)
    df = pd.DataFrame({"class": classes, "support": cm.sum(1), "precision": prec.round(3),
                       "recall": rec.round(3), "f1": f1.round(3)}).sort_values("recall")
    print(f"\n=== {title} ===\n{df.to_string(index=False)}")
    print(f"MACRO-F1 {np.nanmean(f1):.4f} | ACC {np.trace(cm)/max(cm.sum(),1):.4f} | "
          f"WORST-CLASS RECALL {rec.min():.4f}")
    fig, ax = plt.subplots(figsize=(0.62*k+3, 0.62*k+2.5))
    im = ax.imshow(cm/np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(k), classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} (row-normalised)")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i,j]/max(cm.sum(1)[i],1) > .5 else "black")
    plt.colorbar(im); plt.tight_layout(); plt.show()
    return f1, rec

yva_pred = predict_idx(val_ds, len(yva))
f1_va, rec_va = report(confusion(yva, yva_pred, NC), "VAL")
pd.DataFrame({"class": CLASSES, "support": np.bincount(yva, minlength=NC),
              "recall": rec_va.round(3)}).to_csv(OUT / "val_report.csv", index=False)










# %%
# CELL 16 — test set, touched once
yte_pred = predict_idx(test_ds, len(yte))
f1_te, rec_te = report(confusion(yte, yte_pred, NC), "TEST")
gap = np.nanmean(f1_va) - np.nanmean(f1_te)
print(f"\nGAP (val - test macro-F1) = {gap:+.4f}")
print("  > +0.05 : split still leaky or too small to trust")
print("  <  0.00 : val was pessimistic, test is the better number")
pd.DataFrame(confusion(yte, yte_pred, NC), index=CLASSES,
             columns=CLASSES).to_csv(OUT / "test_confusion.csv")










# %%
# CELL 16.5 — source-held-out eval (HEADLINE METRIC)
# The number that matters: train on anshul6+indo3, test on dedeikh across the 5 shared classes
# (Sheath_Blight is single-source in anshul6, so it stays in training but is excluded here).
# In-source val/test (Cells 15-16) is secondary and expected to be much higher.
if man_held is not None and len(man_held):
    shared = sorted(set(CLASSES) & set(man_held["class"].unique()))
    print(f"\n=== SOURCE-HELD-OUT EVAL ===  held-out source: {CFG['held_out_source']}")
    print(f"shared classes: {shared} ({len(shared)} of {len(CLASSES)})")
    if len(shared) < 2:
        print("  <2 shared classes — held-out eval undefined, skipping")
    else:
        # Preprocess exactly like the field stress test (Cell 17): 0-255 float, resize to SIZE.
        # The parity check (Cell 19) proves the deployed graph agrees with `model` at these sizes.
        keep = np.array([c in shared for c in man_held["class"]])
        mh = man_held[keep].reset_index(drop=True)
        t = time.time()
        Xh = np.stack([np.asarray(Image.open(p).convert("RGB")
                                   .resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
                       for p in mh["path"]])
        print(f"preprocessed {len(Xh)} held-out images in {time.time()-t:.0f}s")
        Ph = model.predict(Xh, verbose=0)                       # (n, NC)
        yh = np.array([CLASSES.index(c) for c in mh["class"]], np.int32)
        shared_idx = {c: i for i, c in enumerate(shared)}
        yh_c = np.array([shared_idx[CLASSES[y]] for y in yh], np.int32)
        Ph_c = Ph[:, [CLASSES.index(c) for c in shared]]
        pred_c = Ph_c.argmax(1)
        cm = confusion(yh_c, pred_c, len(shared))
        f1_ho, rec_ho = report(cm, f"SOURCE-HELD-OUT ({CFG['held_out_source']}, "
                               f"{len(shared)} classes)", classes=shared)

        # bootstrap 95% CI on macro-F1 (resample images with replacement)
        def boot_macro_f1(y, p, k, iters=CFG["bootstrap_iters"], seed=0):
            rng = np.random.RandomState(seed)
            n = len(y); scores = np.empty(iters)
            for it in range(iters):
                idx = rng.randint(0, n, n)
                cmb = confusion(y[idx], p[idx], k)
                tp = np.diag(cmb).astype(float)
                prec = np.divide(tp, cmb.sum(0), out=np.zeros_like(tp), where=cmb.sum(0) > 0)
                recb = np.divide(tp, cmb.sum(1), out=np.zeros_like(tp), where=cmb.sum(1) > 0)
                f1b = np.divide(2*prec*recb, prec+recb, out=np.zeros_like(tp), where=(prec+recb) > 0)
                scores[it] = np.nanmean(f1b)
            return np.percentile(scores, [2.5, 97.5])
        lo, hi = boot_macro_f1(yh_c, pred_c, len(shared))
        print(f"macro-F1 bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")

        # abstain rule: skip predictions below the confidence threshold
        thr = CFG["abstain_threshold"]
        conf = Ph_c.max(1)
        abstain = conf < thr
        cov = (~abstain).mean()
        if cov > 0:
            cm_a = confusion(yh_c[~abstain], pred_c[~abstain], len(shared))
            f1a, _ = report(cm_a, f"HELD-OUT with abstain@{thr}", classes=shared)
            acc_a = np.trace(cm_a) / max(cm_a.sum(), 1)
        else:
            f1a, acc_a = np.nan, 0.0
        print(f"abstain@{thr}: coverage {cov:.1%} | acc on covered {acc_a:.4f} | "
              f"macro-F1 on covered {np.nanmean(f1a):.4f}")

        # per-source macro-F1 (general: works if the held-out set spans several sources)
        print("\nper-source macro-F1 (held-out set):")
        for src, grp in mh.groupby("source"):
            idx = np.nonzero(mh["source"].to_numpy() == src)[0]
            cm_s = confusion(yh_c[idx], pred_c[idx], len(shared))
            tp = np.diag(cm_s).astype(float)
            prec = np.divide(tp, cm_s.sum(0), out=np.zeros_like(tp), where=cm_s.sum(0) > 0)
            recb = np.divide(tp, cm_s.sum(1), out=np.zeros_like(tp), where=cm_s.sum(1) > 0)
            f1s = np.divide(2*prec*recb, prec+recb, out=np.zeros_like(tp), where=(prec+recb) > 0)
            print(f"  {src:12s} n={len(idx):5d}  macro-F1 {np.nanmean(f1s):.4f}")

        # in-source test per-source macro-F1 (uses yte_pred/yte from Cell 16)
        print("\nper-source macro-F1 (in-source TEST):")
        for src, grp in te.groupby("source"):
            idx = np.nonzero(te["source"].to_numpy() == src)[0]
            cm_s = confusion(yte[idx], yte_pred[idx], NC)
            tp = np.diag(cm_s).astype(float)
            prec = np.divide(tp, cm_s.sum(0), out=np.zeros_like(tp), where=cm_s.sum(0) > 0)
            recb = np.divide(tp, cm_s.sum(1), out=np.zeros_like(tp), where=cm_s.sum(1) > 0)
            f1s = np.divide(2*prec*recb, prec+recb, out=np.zeros_like(tp), where=(prec+recb) > 0)
            print(f"  {src:12s} n={len(idx):5d}  macro-F1 {np.nanmean(f1s):.4f}")

        pd.DataFrame({"class": shared, "f1": f1_ho.round(4),
                      "recall": rec_ho.round(4)}).to_csv(OUT / "source_held_out.csv", index=False)
else:
    print("source-held-out eval SKIPPED (CFG['held_out_source'] is None or empty)")










# %%
# CELL 17 — crawler, quarantined
# NOT bulk-injected into training: web results are 20-40% mislabelled. Default role is a
# held-out field stress test. Wikimedia is tried first because it does not block datacentre
# IPs; DuckDuckGo usually does.
import requests

CR = WORK / "field_crawl"; CR.mkdir(parents=True, exist_ok=True)
prov = []
SESS = requests.Session()
# Wikimedia's UA policy asks for a descriptive agent with contact info.
SESS.headers.update({"User-Agent": "AgrisenseResearch/1.0 (academic rice-disease model; "
                                   "contact: gau.mah077@gmail.com)"})

def _fetch_bytes(url, max_bytes):
    # Stream the body and stop at max_bytes: a 20 MB "photo" is never useful, and reading it
    # whole wastes RAM and wall time on a crawl that is already time-boxed.
    with SESS.get(url, timeout=15, stream=True) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(1 << 16):
            buf.write(chunk)
            if buf.tell() > max_bytes:
                raise ValueError(f"body exceeds {max_bytes} bytes")
        return buf.getvalue()

def wiki_urls(q, n):
    r = SESS.get("https://commons.wikimedia.org/w/api.php", timeout=20, params={
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {q}", "gsrnamespace": "6", "gsrlimit": str(n*2),
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "640"})
    r.raise_for_status()
    for pg in (r.json().get("query", {}).get("pages", {}) or {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        if ii.get("thumburl"):
            lic = (ii.get("extmetadata") or {}).get("LicenseShortName", {}).get("value", "?")
            yield ii["thumburl"], lic

def ddg_urls(q, n):
    try:
        from ddgs import DDGS
    except ImportError:
        try: from duckduckgo_search import DDGS
        except ImportError: return
    try:
        with DDGS() as d:
            for r in d.images(q, max_results=n*2):
                yield r["image"], "web-search-license-unknown"
    except Exception as e:
        print(f"   ddg blocked (datacentre IP): {str(e)[:60]}")

def crawl():
    t0 = time.time()
    for cls, queries in CFG["crawl"]["queries"].items():
        d = CR / cls; d.mkdir(exist_ok=True); got = 0
        stop = False
        for q in queries:
            for pv in CFG["crawl"]["providers"]:
                if got >= CFG["crawl"]["per_class"] or time.time()-t0 > CFG["crawl"]["max_seconds"]:
                    stop = True; break
                gen = (wiki_urls(q, CFG["crawl"]["per_class"]) if pv == "wikimedia"
                       else ddg_urls(q, CFG["crawl"]["per_class"]))
                try:
                    for url, lic in gen:
                        if got >= CFG["crawl"]["per_class"]: break
                        try:
                            with Image.open(io.BytesIO(_fetch_bytes(url, CFG["crawl"]["max_bytes"]))) as im:
                                if min(im.size) < CFG["crawl"]["min_side"] or \
                                   im.format not in ("JPEG", "PNG"): continue
                                p = d / f"{got:03d}.jpg"
                                im.convert("RGB").save(p, quality=92)
                        except Exception:
                            continue
                        prov.append({"class": cls, "file": str(p), "url": url, "query": q,
                                     "license": lic, "role": CFG["crawl"]["role"]})
                        got += 1; time.sleep(CFG["crawl"]["sleep"])
                except Exception as e:
                    print(f"   {pv} '{q[:26]}': {str(e)[:60]}")
            if stop: break
        print(f"  {cls}: {got} images")
    pd.DataFrame(prov).to_csv(CR / "provenance.csv", index=False)
    print(f"crawl done in {time.time()-t0:.0f}s -> {CR}")

assert CFG["crawl"]["role"] == "stress_test", "finetune role was removed on purpose"
if CFG["crawl"]["enable"] and not CFG["smoke"]:
    crawl()
    rows = []
    for p in sorted(CR.rglob("*.jpg")):
        with Image.open(p) as im:
            # 0-255 floats: EfficientNet preprocesses internally, so no /255 here. PIL resize
            # takes a Resampling enum (np.float32 raises "Unknown resampling filter"), and a
            # PIL Image is not subscriptable — np.asarray must come before [None].
            x = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR),
                           dtype=np.float32)[None]
        pr = model.predict(x, verbose=0)[0]
        s = np.sort(pr)
        rows.append({"true": p.parent.name, "pred": CLASSES[int(pr.argmax())],
                     "conf": round(float(pr.max()), 3),
                     "margin": round(float(s[-1]-s[-2]), 3)})
    if rows:
        df = pd.DataFrame(rows)
        print("\n=== FIELD STRESS TEST ===\n" + df.to_string(index=False))
        print(f"top-1 {float((df['true']==df['pred']).mean()):.1%} | "
              f"mean margin {df['margin'].mean():.3f}")
        # abstain rule: same threshold as the held-out eval (Cell 16.5)
        thr = CFG["abstain_threshold"]
        cov = float((df["conf"] >= thr).mean())
        acc_cov = (float((df.loc[df["conf"] >= thr, "true"] ==
                          df.loc[df["conf"] >= thr, "pred"]).mean()) if cov > 0 else float("nan"))
        print(f"abstain@{thr}: coverage {cov:.1%} | top-1 on covered {acc_cov:.1%}")
        df.to_csv(OUT / "field_stress_test.csv", index=False)
        print("Crawled labels are themselves noisy. Read the pattern, not the number.")










# %%
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
print(f"PIPELINE_VERSION {PIPELINE_VERSION} | export")
if not CFG["smoke"]:
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
else:
    print("SMOKE: export skipped")










# %%
# CELL 19 — TFLite + parity check + bundle
print(f"PIPELINE_VERSION {PIPELINE_VERSION} | bundle")
if not CFG["smoke"]:
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
        # Read the input shape FROM THE INTERPRETER instead of assuming it. A dynamic HxW input
        # reports an allocated shape of [1,1,1,3]; the old fallback then resized every probe to
        # 1x1, so the parity check compared nothing. The export input is fixed now, so anything
        # other than PRE means something is wrong and the number below would be meaningless.
        shp = inp_d["shape"]
        h, w = int(shp[1]), int(shp[2])
        assert (h, w) == (PRE, PRE), f"unexpected TFLite input shape {list(shp)} — parity invalid"
        rng_p = np.random.RandomState(0)
        agree, maxdiff, nprobe = 0, 0.0, min(16, len(yva))
        for i in rng_p.choice(len(yva), size=nprobe, replace=False):
            probe = (Xva[i][None] if Xva is not None
                     else np.asarray(Image.open(va["path"].iloc[i]).convert("RGB")
                                     .resize((PRE, PRE), Image.BILINEAR))[None])
            probe = probe.astype(np.float32)
            k_out = clean.predict(probe, verbose=0)[0]   # `clean`, NOT `model`: model is fixed at
            # SIZE (224) and raises on a 256px probe; clean takes (1, PRE, PRE, 3) like tflite.
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
else:
    print("SMOKE: bundle skipped")










# %%
# CELL 20 — cleanup
del train_ds, val_ds, test_ds
if USE_RAM:
    del Xtr, Xva, Xte
import gc; gc.collect()
print(f"freed input arrays; RAM available now {_ram_gb()[1]:.1f} GB")
