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
