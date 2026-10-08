# AgriSense — XGBoost risk model (`ml/risk_xgb/`)

Predicts the **overall risk score (0–100)** from raw field features with XGBoost,
then calibrates it to a 0–1 probability for the `/ml/risk` endpoint. This replaces
the old linear `fit_risk.py` placeholder.

**PIPELINE_VERSION: `risk-xgb-v1`** — printed by every entry point.

---

## 1. How to add the xlsheet data (the friend's dataset)

The file is `konkan_rice_disease_risk_10k.csv.xls` — **CSV content with a `.xls`
extension** (Excel opens it fine; pandas reads it as CSV). It is **not** pushed to
GitHub (`.gitignore`), so you must place it yourself.

**Local run:**
1. Put `konkan_rice_disease_risk_10k.csv.xls` in the **project root**
   (`AgriSense/`). That is the default `CFG["data_path"]`.
2. Or put it anywhere and set the path in Cell 1:
   ```python
   CFG["data_path"] = "C:/path/to/konkan_rice_disease_risk_10k.csv.xls"
   ```

**Kaggle run:**
1. Open the notebook (cells/ pasted top-to-bottom, or `risk_xgb_notebook.py`).
2. **Add Input → Upload** the xlsheet (or attach it as a dataset).
3. Set the path in Cell 1, e.g.:
   ```python
   CFG["data_path"] = "/kaggle/input/konkan-rice-disease-risk/konkan_rice_disease_risk_10k.csv.xls"
   CFG["out_dir"]   = "/kaggle/working"
   ```
4. Run all cells. Download `ml/models/risk_xgb_artifact.json` and
   `ml/models/risk_xgb_model.json` from `/kaggle/working` and drop them into the
   local `ml/models/` folder.

**Required columns** (the pipeline fails loudly if any are missing):
`growth_stage, rice_variety_type, temperature_min, temperature_max,
relative_humidity, rainfall_7d_forecast, consecutive_rainy_days,
field_water_level_cm, soil_ph, nitrogen_applied_level, primary_disease_risk,
overall_risk_score`.

`risk_level` is **dropped on purpose** — it is 1:1 with `primary_disease_risk =
None (Healthy)`, so a model that sees it would cheat instead of learn.

---

## 2. How to run

```bash
# smoke first (500 rows, 50 trees, ~3 s) — always run this before a full run
python -c "from risk_xgb import Pipeline; Pipeline({'smoke': True}).run()"

# full run (10k rows, 1000 trees + early stopping, ~3 s on CPU)
python -c "from risk_xgb import Pipeline; Pipeline({}).run()"
```

Run from inside `ml/risk_xgb/`. The full run writes:

| File | Contents |
|---|---|
| `ml/models/risk_xgb_artifact.json` | feature maps, calibration bins, conformal q, metrics, importance |
| `ml/models/risk_xgb_model.json` | the XGBoost booster (JSON format) |

**Current full-run metrics (10k rows, 70/15/15):** test RMSE **4.64**, MAE **3.61**,
R² **0.926**, 90% conformal interval **±7.5 pts**.

---

## 3. Architecture

The notebook is a thin 3-cell wrapper (CONFIG → MODULE → RUN) around `risk_xgb.py`.
`split_cells.py` generates `cells/` (one file per cell) from the notebook source —
the module is the single source of truth.

Pipeline stages (`Pipeline(CFG).run()`):

1. **load_data** — read the xlsheet (CSV content).
2. **clean** — drop `risk_level` leakage, coerce numerics, drop NaN rows, sanity-check
   `temperature_min < temperature_max`.
3. **features** — one-hot encode the 4 categoricals (growth_stage, rice_variety_type,
   nitrogen_applied_level, primary_disease_risk) + keep 7 numerics = **24 features**.
4. **split** — stratified 70/15/15 by risk band (Low/Moderate/High/Critical).
5. **train** — `XGBRegressor`-style `xgb.train`, early stopping 50, deliberately
   regularized (`max_depth=5, lr=0.05, subsample=0.8, colsample=0.8, λ=1.0, α=0.1`).
6. **evaluate** — RMSE/MAE/R² on train/val/test + isotonic calibration (val) +
   conformal interval (90th percentile of |val error|).
7. **save** — write the two files above.

---

## 4. Inference — `predict()` (never fails on real data)

```python
from risk_xgb import predict

out = predict({
    "growth_stage": "Panicle Initiation",
    "rice_variety_type": "Short Duration (Karjat-3)",
    "nitrogen_applied_level": "Excessive",
    "primary_disease_risk": "Leaf Blast",
    "temperature_min": 22.0, "temperature_max": 31.0,
    "relative_humidity": 85.0, "rainfall_7d_forecast": 45.0,
    "consecutive_rainy_days": 4, "field_water_level_cm": 8.0, "soil_ph": 6.2,
})
# -> {"score": 0.686, "interval": [0.577, 0.794], "priority": "medium",
#     "drivers": [...top-3 features...], "model_version": "risk-xgb-v1",
#     "needs_expert": False}
```

Robustness guarantees (this is what "no fail on real data" means):

- **Never raises.** Every failure path returns the linear-prior fallback
  (`model_version: "risk-xgb-fallback-priors"`, `needs_expert: True`).
- **Missing values** → NaN, which XGBoost handles natively.
- **Unseen categories** (e.g. a variety not in training) → all-zero one-hot, no crash.
- **Out-of-range numerics** → clipped to the training min/max (saved in the artifact).
- **Degenerate input** (empty dict / all-NaN) → prior fallback, not a guess.
- **No artifact/model present** → prior fallback (the old linear engine still works).

Priority bands match `ml/src/risk_engine.py`: **high ≥ 0.70, medium ≥ 0.45, low < 0.45**.

---

## 5. `/ml/risk` contract extension

The old endpoint took 5 engineered `RiskInput` fields (`disease_prob`, `conf`,
`rainfall_anomaly`, `stage_risk`, `yield_stress`). The XGBoost model takes **raw
field features** instead — the backend maps `/ml/disease` output to
`primary_disease_risk` (most likely class) and passes the field context through:

```json
POST /ml/risk
{
  "growth_stage": "Flowering",
  "rice_variety_type": "Short Duration (Karjat-3)",
  "nitrogen_applied_level": "Excessive",
  "primary_disease_risk": "Leaf Blast",
  "temperature_min": 20.0, "temperature_max": 28.0,
  "relative_humidity": 92.0, "rainfall_7d_forecast": 80.0,
  "consecutive_rainy_days": 6, "field_water_level_cm": 12.0, "soil_ph": 5.8
}
```

Response shape is unchanged from the current contract (`score`, `interval`,
`priority`, `drivers`, `needs_expert`), so the UI does not change.

---

## 6. Honest caveats (read before trusting the numbers)

- The 10k rows are **synthetic**: `overall_risk_score` is a deterministic function
  of the features (0 feature-key collisions in 10k rows). The near-perfect fit
  reflects the **data generator**, not real-world generalization.
- The model is deliberately regularized so it learns the *shape* of the generator,
  not its exact formula. R² 0.926 is the honest number; chasing 0.99 would mean
  memorizing the generator and would generalize worse on real fields.
- The conformal interval (±7.5 pts at 90%) is computed on the synthetic val set.
  On real data the interval will be wider — treat it as a lower bound.
- `primary_disease_risk` is a required feature. If the disease model is unsure,
  pass the most likely class anyway; the score will be smoothed by calibration.