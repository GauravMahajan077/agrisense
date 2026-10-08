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
