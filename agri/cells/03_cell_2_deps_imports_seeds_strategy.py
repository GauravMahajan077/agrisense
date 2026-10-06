# CELL 2 — deps, imports, seeds, strategy
# imagehash and ddgs are NOT Kaggle preinstalls.
import importlib.util, subprocess as _sp

def _pip(*pkgs, required=True):
    # Probe by IMPORT name, install by PyPI name. "PIL" is the import name of "Pillow"; pip has
    # no package called "PIL", so installing the probe name fails on a clean environment.
    _PYPI = {"PIL": "Pillow"}
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if not missing:
        print(f"present: {' '.join(pkgs)}"); return
    names = [_PYPI.get(p, p) for p in missing]
    print(f"installing: {' '.join(names)} (import name(s): {' '.join(missing)})")
    try:
        _sp.run([sys.executable, "-m", "pip", "install", "-q", *names], check=True)
        # pip returning 0 is not proof: the install can land yet the module still not import
        # (stale sys.path cache, or a shadowing local file). Re-probe before claiming OK.
        importlib.invalidate_caches()
        for p in missing:
            if importlib.util.find_spec(p) is None:
                raise ImportError(f"{p} still not importable after installing {' '.join(names)}")
        print("  installed and importable")
    except Exception as e:
        if required: raise
        print(f"  optional install failed ({type(e).__name__}) — continuing without it")

# NOTE: import name "PIL" is the PyPI package "Pillow" — _pip maps between them.
_pip("PIL", "imagehash")
try:
    _pip("ddgs", required=False)       # DuckDuckGo image search, crawler only
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
