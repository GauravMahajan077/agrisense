# %% [markdown]
# # AgriSense — XGBoost risk model, Kaggle notebook
#
# **Read `README.md` first.** This notebook is a thin 3-cell wrapper around the
# `risk_xgb.py` module. Cell 1 (`CFG`) is the only cell you edit.
#
# - **Cell 1 — CONFIG**: the `CFG` dict. It is the editable copy of `DEFAULT_CFG`
#   in `risk_xgb.py`; keep the two in sync (the module is the source of truth).
# - **Cell 2 — MODULE**: generated from `risk_xgb.py` by `split_cells.py`
#   (the `# %% include:risk_xgb.py` directive). Do not hand-edit.
# - **Cell 3 — RUN**: calls `Pipeline(CFG).run()`, which prints `PIPELINE_VERSION`
#   and runs the whole pipeline (load → clean → features → split → train →
#   evaluate → save).
#
# The full "what changed" history lives in `README.md` and in the module docstring.

# %%
# CELL 1 — CONFIG. The only cell you edit.
CFG = {
    "seed": 1337,

    # smoke = 500 rows, 50 trees, no calibration bins. Run it before any full run.
    "smoke": False,

    # ---------- data ----------
    # The friend's xlsheet is CSV content with a .xls extension. Default path is the
    # project root (konkan_rice_disease_risk_10k.csv.xls). On Kaggle, upload the file
    # to the notebook (Add Input / Upload) and point this at it, e.g.
    #   "/kaggle/input/konkan-rice-disease-risk/konkan_rice_disease_risk_10k.csv.xls"
    "data_path": "konkan_rice_disease_risk_10k.csv.xls",

    # Where the artifact + model are written. On Kaggle use /kaggle/working, then
    # download both files and drop them into ml/models/ locally.
    "out_dir": "ml/models",

    # ---------- split ----------
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},

    # ---------- XGBoost ----------
    # Regularized on purpose: the dataset is synthetic/deterministic, so a deep model
    # would memorize the generator. These params keep it smooth for real data.
    "xgb": {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "early_stopping_rounds": 50,
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
    },

    # 90% conformal interval width (percentile of |val error|).
    "conformal_q": 0.90,
}

# %%
# CELL 2 — MODULE. Generated from risk_xgb.py by split_cells.py. Do not hand-edit.
# %% include:risk_xgb.py

# %%
# CELL 3 — RUN. Prints PIPELINE_VERSION and runs the whole pipeline.
Pipeline(CFG).run()

# After a full run, download ml/models/risk_xgb_artifact.json + risk_xgb_model.json
# and drop them into the local ml/models/ folder. The /ml/risk endpoint then uses
# risk_xgb.predict() automatically (see ml/risk_xgb/README.md).