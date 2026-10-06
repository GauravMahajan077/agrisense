# %% [markdown]
# # Agrisense — paddy disease classification, Kaggle notebook
#
# **Read `README.md` first.** This notebook is a thin 3-cell wrapper around the
# `agrisense.py` module. Cell 1 (`CFG`) is the only cell you edit.
#
# - **Cell 1 — CONFIG**: the `CFG` dict. It is the editable copy of `DEFAULT_CFG`
#   in `agrisense.py`; `verify.py` AST-compares the two so they cannot drift.
# - **Cell 2 — MODULE**: generated from `agrisense.py` by `split_cells.py`
#   (the `# %% include:agrisense.py` directive). Do not hand-edit.
# - **Cell 3 — RUN**: calls `run(CFG)`, which prints `PIPELINE_VERSION` and runs
#   the whole pipeline.
#
# The full "what changed" history lives in `README.md` and in the module docstring.

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
    "jit": False,                  # XLA off by default: MacroF1 uses tf.math.confusion_matrix,
                                   # a possible XLA compile failure, and at ~35 steps/epoch XLA
                                   # saves almost nothing. True to try it on a single GPU.
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
        "contact": "gau.mah077@gmail.com",   # crawler User-Agent contact (Wikimedia UA policy)
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
# CELL 2 — MODULE. Generated from agrisense.py by split_cells.py. Do not hand-edit.
# %% include:agrisense.py

# %%
# CELL 3 — RUN. Prints PIPELINE_VERSION and runs the whole pipeline.
run(CFG)

# Optional, after a full (non-smoke) run: evaluate the exported bundle on your own photos
# with agri/new/field_test.py (see README section 7.5).