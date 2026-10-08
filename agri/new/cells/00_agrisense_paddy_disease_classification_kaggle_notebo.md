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
