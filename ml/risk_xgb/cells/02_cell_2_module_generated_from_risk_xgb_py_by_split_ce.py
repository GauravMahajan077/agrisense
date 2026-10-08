# CELL 2 — MODULE. Generated from risk_xgb.py by split_cells.py. Do not hand-edit.
"""XGBoost risk model for AgriSense — predicts overall_risk_score (0-100).

The Kaggle notebook is a thin 3-cell wrapper: CONFIG -> MODULE -> RUN. This module
holds all the logic. PIPELINE_VERSION is printed by every entry point so a stale
paste is visible in the log instead of silently shipping.

Pipeline stages (run()):
  1. load_data  — read the friend's xlsheet (CSV content, .xls extension)
  2. clean      — drop the risk_level leakage column, coerce types, validate ranges
  3. features   — one-hot encode categoricals, build the feature matrix + maps
  4. split      — stratified train/val/test split by risk band
  5. train      — XGBoost regressor, early stopping, regularization
  6. evaluate   — RMSE/MAE/R2 + isotonic calibration + conformal interval
  7. save       — write ml/models/risk_xgb_artifact.json + risk_xgb_model.json

Inference (predict()): maps raw field features -> model features, never raises,
falls back to the linear risk_engine priors if the artifact is missing.

Known caveat (honest): the friend's 10k rows are SYNTHETIC and the target is a
deterministic function of the features (0 feature-key collisions in 10k rows).
So near-perfect in-sample metrics reflect the data generator, not real-world
generalization. The robustness layer (clipping, unseen-category fallback, NaN
handling, conformal interval, prior fallback) is what makes it "no fail on real
data" — not the R2.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover — xgboost absent -> predict falls back to priors
    xgb = None
    _HAS_XGB = False

PIPELINE_VERSION = "risk-xgb-v1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_MODEL_DIR = ROOT / "ml" / "models"

CATEGORICALS = ["growth_stage", "rice_variety_type", "nitrogen_applied_level",
                "primary_disease_risk"]
NUMERICS = ["temperature_min", "temperature_max", "relative_humidity",
            "rainfall_7d_forecast", "consecutive_rainy_days", "field_water_level_cm",
            "soil_ph"]
TARGET = "overall_risk_score"
LEAK_COLS = ["risk_level"]  # 1:1 with 'None (Healthy)' -> model could cheat

# Same priority bands as ml/src/risk_engine.py so the two share a contract.
PRIORITY_HIGH = 0.70
PRIORITY_MED = 0.45


def _risk_band(score: float) -> str:
    if score >= PRIORITY_HIGH:
        return "high"
    if score >= PRIORITY_MED:
        return "medium"
    return "low"


DEFAULT_CFG = {
    "seed": 1337,
    "smoke": False,  # smoke = 500 rows, 50 trees, no calibration bins. Run it before any full run.
    "data_path": str(ROOT / "konkan_rice_disease_risk_10k.csv.xls"),
    "out_dir": str(DEFAULT_MODEL_DIR),
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
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
    "conformal_q": 0.90,  # 90% interval
}


# ─────────────────────────── feature encoding ───────────────────────────

def build_feature_matrix(df: pd.DataFrame, cat_maps: dict | None = None,
                         num_ranges: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """One-hot encode categoricals + keep numerics. Returns (X, cat_maps, num_ranges).

    cat_maps: {col: [categories in order]} — learned on train, reused at inference.
    num_ranges: {col: [min, max]} — learned on train, used to clip inference inputs.
    """
    X = pd.DataFrame(index=df.index)
    learned_maps = cat_maps is None
    cat_maps = cat_maps or {}
    num_ranges = num_ranges or {}

    for col in CATEGORICALS:
        cats = cat_maps.get(col)
        if cats is None:
            cats = sorted(df[col].dropna().unique().tolist())
            cat_maps[col] = cats
        for c in cats:
            X[f"{col}::{c}"] = (df[col] == c).astype(np.float32)

    for col in NUMERICS:
        vals = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        if learned_maps:
            num_ranges[col] = [float(vals.min()), float(vals.max())]
        lo, hi = num_ranges.get(col, [float(vals.min()), float(vals.max())])
        X[col] = vals.clip(lo, hi)

    return X, cat_maps, num_ranges


def encode_row(row: dict, cat_maps: dict, num_ranges: dict) -> dict:
    """Map one raw field-context dict to the model feature vector (never raises)."""
    feats: dict[str, float] = {}
    for col in CATEGORICALS:
        val = str(row.get(col) or "")
        for c in cat_maps.get(col, []):
            feats[f"{col}::{c}"] = 1.0 if val == c else 0.0
    for col in NUMERICS:
        try:
            v = float(row.get(col))
        except (TypeError, ValueError):
            v = float("nan")
        lo, hi = num_ranges.get(col, [float("-inf"), float("inf")])
        feats[col] = float(np.clip(v, lo, hi))
    return feats


# ─────────────────────────── calibration helpers ───────────────────────────

def calibrate(score: float, calibration: dict) -> float:
    """Piecewise-linear isotonic calibration (same contract as risk_engine.calibrate)."""
    bins = calibration.get("bins") or []
    if calibration.get("type") != "isotonic" or len(bins) < 2:
        return score
    xs = [b[0] for b in bins]
    ys = [b[1] for b in bins]
    for i in range(1, len(xs)):
        if score <= xs[i]:
            t = (score - xs[i - 1]) / (xs[i] - xs[i - 1] + 1e-9)
            return max(0.0, min(1.0, ys[i - 1] + t * (ys[i] - ys[i - 1])))
    return ys[-1]


# ─────────────────────────── Pipeline ───────────────────────────

class Pipeline:
    """Train + evaluate + save the XGBoost risk model. run() drives all stages."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg["xgb"] = {**DEFAULT_CFG["xgb"], **self.cfg.get("xgb", {})}
        self.seed = int(self.cfg["seed"])
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: np.ndarray | None = None
        self.cat_maps: dict = {}
        self.num_ranges: dict = {}
        self.model = None
        self.artifact: dict = {}

    # ---- 1. load ----
    def load_data(self) -> pd.DataFrame:
        path = Path(self.cfg["data_path"])
        if not path.exists():
            raise SystemExit(f"data not found: {path}\n"
                             "Place konkan_rice_disease_risk_10k.csv.xls at the project root "
                             "(or set CFG['data_path']). See ml/risk_xgb/README.md.")
        # The file is CSV content with a .xls extension.
        df = pd.read_csv(path)
        missing = [c for c in CATEGORICALS + NUMERICS + [TARGET] if c not in df.columns]
        if missing:
            raise SystemExit(f"data missing required columns: {missing}")
        self.df = df
        return df

    # ---- 2. clean ----
    def clean(self) -> pd.DataFrame:
        df = self.df
        if df is None:
            raise RuntimeError("load_data() first")
        df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])
        for col in NUMERICS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=NUMERICS + [TARGET]).reset_index(drop=True)
        # sanity: tmin < tmax
        bad = int((df["temperature_min"] >= df["temperature_max"]).sum())
        if bad:
            raise SystemExit(f"clean: {bad} rows with temperature_min >= temperature_max")
        self.df = df
        return df

    # ---- 3. features ----
    def features(self) -> pd.DataFrame:
        X, cat_maps, num_ranges = build_feature_matrix(self.df)
        self.X = X
        self.cat_maps = cat_maps
        self.num_ranges = num_ranges
        self.y = self.df[TARGET].to_numpy(dtype=np.float32)
        return X

    # ---- 4. split (stratified by risk band) ----
    def split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split
        bands = pd.Series(pd.cut(self.y, bins=[-1, 30, 60, 80, 101], labels=[0, 1, 2, 3]))
        s = self.cfg["split"]
        idx = np.arange(len(self.y))
        tr_idx, rest = train_test_split(idx, test_size=s["val"] + s["test"],
                                        stratify=bands, random_state=self.seed)
        rest_bands = bands.iloc[rest]
        val_idx, te_idx = train_test_split(
            rest, test_size=s["test"] / (s["val"] + s["test"]),
            stratify=rest_bands, random_state=self.seed)
        self.tr_idx, self.val_idx, self.te_idx = tr_idx, val_idx, te_idx
        return tr_idx, val_idx, te_idx

    # ---- 5. train ----
    def train(self) -> object:
        if not _HAS_XGB:
            raise SystemExit("xgboost not installed — run on Kaggle (pip install xgboost) "
                             "or install locally for the full run")
        smoke = bool(self.cfg["smoke"])
        X, y = self.X, self.y
        if smoke:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(len(y), size=min(500, len(y)), replace=False)
            X, y = X.iloc[keep], y[keep]
            self.tr_idx = np.arange(len(y))
            self.val_idx = np.arange(len(y))  # early stopping needs a val set
            self.te_idx = np.arange(len(y))
        p = self.cfg["xgb"]
        params = {k: v for k, v in p.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}
        params["random_state"] = self.seed
        params["n_jobs"] = max(1, (os_cpu_count() or 4) - 1)
        dtr = xgb.DMatrix(X.iloc[self.tr_idx], label=y[self.tr_idx])
        dva = xgb.DMatrix(X.iloc[self.val_idx], label=y[self.val_idx])
        self.model = xgb.train(
            params, dtr, num_boost_round=p["n_estimators"],
            evals=[(dva, "val")], early_stopping_rounds=p["early_stopping_rounds"],
            verbose_eval=False)
        return self.model

    # ---- 6. evaluate ----
    def evaluate(self) -> dict:
        from sklearn.isotonic import IsotonicRegression
        X, y = self.X, self.y
        pred_tr = self.model.predict(xgb.DMatrix(X.iloc[self.tr_idx]))
        pred_va = self.model.predict(xgb.DMatrix(X.iloc[self.val_idx]))
        pred_te = self.model.predict(xgb.DMatrix(X.iloc[self.te_idx]))

        def metrics(p, t):
            err = p - t
            return {"rmse": float(np.sqrt(np.mean(err ** 2))),
                    "mae": float(np.mean(np.abs(err))),
                    "r2": float(1 - np.sum(err ** 2) / np.sum((t - t.mean()) ** 2))}

        m = {"train": metrics(pred_tr, y[self.tr_idx]),
             "val": metrics(pred_va, y[self.val_idx]),
             "test": metrics(pred_te, y[self.te_idx])}

        # calibration: isotonic on val (raw 0-100 -> 0-1)
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.clip(pred_va, 0, 100) / 100.0, np.clip(y[self.val_idx], 0, 100) / 100.0)
        xs = np.linspace(0, 1, 21)
        bins = [[float(x), float(iso.predict([x])[0])] for x in xs]

        # conformal: 90th percentile |error| on val, in 0-1 space
        cal_err = np.abs(np.clip(pred_va, 0, 100) / 100.0 - np.clip(y[self.val_idx], 0, 100) / 100.0)
        q = float(np.quantile(cal_err, self.cfg["conformal_q"]))

        self.artifact = {
            "model_version": PIPELINE_VERSION,
            "model_type": "xgboost",
            "model_file": "risk_xgb_model.json",
            "feature_order": list(X.columns),
            "cat_maps": self.cat_maps,
            "num_ranges": self.num_ranges,
            "calibration": {"type": "isotonic", "bins": bins},
            "conformal": {"q": round(q, 4), "n_calib": int(len(self.val_idx))},
            "metrics": m,
            "n_train": int(len(self.tr_idx)), "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.te_idx)),
            "target_min": float(y.min()), "target_max": float(y.max()),
            "importance": {k: float(v) for k, v in
                           self.model.get_score(importance_type="gain").items()},
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return m

    # ---- 7. save ----
    def save(self) -> list[Path]:
        out = Path(self.cfg["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        art_path = out / "risk_xgb_artifact.json"
        model_path = out / "risk_xgb_model.json"
        art_path.write_text(json.dumps(self.artifact, indent=2), encoding="utf-8")
        self.model.save_model(str(model_path))
        return [art_path, model_path]

    # ---- run ----
    def run(self) -> dict:
        print(f"PIPELINE_VERSION: {PIPELINE_VERSION}  (smoke={bool(self.cfg['smoke'])})")
        t0 = time.time()
        self.load_data()
        self.clean()
        self.features()
        self.split()
        self.train()
        m = self.evaluate()
        paths = self.save()
        print(f"  rows={len(self.df)}  features={self.X.shape[1]}  "
              f"train/val/test={len(self.tr_idx)}/{len(self.val_idx)}/{len(self.te_idx)}")
        print(f"  test  RMSE={m['test']['rmse']:.2f}  MAE={m['test']['mae']:.2f}  "
              f"R2={m['test']['r2']:.4f}")
        print(f"  val   RMSE={m['val']['rmse']:.2f}  MAE={m['val']['mae']:.2f}  "
              f"R2={m['val']['r2']:.4f}")
        print(f"  conformal q={self.artifact['conformal']['q']:.3f} "
              f"({self.cfg['conformal_q'] * 100:.0f}% interval)")
        print(f"  artifact -> {paths[0]}")
        print(f"  model    -> {paths[1]}")
        print(f"  elapsed  {time.time() - t0:.1f}s")
        return self.artifact


# ─────────────────────────── inference ───────────────────────────

_ARTIFACT_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _load_artifact(model_dir: str | Path | None = None) -> dict | None:
    d = str(model_dir or DEFAULT_MODEL_DIR)
    if d in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[d]
    art_path = Path(d) / "risk_xgb_artifact.json"
    if not art_path.exists():
        return None
    try:
        art = json.loads(art_path.read_text(encoding="utf-8"))
        _ARTIFACT_CACHE[d] = art
        return art
    except Exception:  # pragma: no cover
        return None


def _load_model(model_dir: str | Path | None = None) -> object | None:
    d = str(model_dir or DEFAULT_MODEL_DIR)
    if d in _MODEL_CACHE:
        return _MODEL_CACHE[d]
    if not _HAS_XGB:
        return None
    model_path = Path(d) / "risk_xgb_model.json"
    if not model_path.exists():
        return None
    try:
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        _MODEL_CACHE[d] = bst
        return bst
    except Exception:  # pragma: no cover
        return None


def predict(features: dict, model_dir: str | Path | None = None) -> dict:
    """Risk score from raw field features. NEVER raises — falls back to priors.

    features keys: growth_stage, rice_variety_type, nitrogen_applied_level,
    primary_disease_risk, temperature_min, temperature_max, relative_humidity,
    rainfall_7d_forecast, consecutive_rainy_days, field_water_level_cm, soil_ph.
    """
    art = _load_artifact(model_dir)
    model = _load_model(model_dir)
    if art is None or model is None:
        return _fallback_prior(features)

    try:
        row = encode_row(features, art["cat_maps"], art["num_ranges"])
        # Degenerate input (empty dict / all NaN / no category present): the model has
        # nothing to work with — fall back to priors instead of guessing.
        usable = [v for v in row.values() if v == v and v != 0.0]
        if not usable:
            return _fallback_prior(features)
        order = art["feature_order"]
        vec = np.array([[row.get(c, float("nan")) for c in order]], dtype=np.float32)
        raw = float(model.predict(xgb.DMatrix(vec, feature_names=order))[0])
        tmin, tmax = art["target_min"], art["target_max"]
        score01 = float(np.clip((raw - tmin) / (tmax - tmin), 0.0, 1.0))
        score = calibrate(score01, art["calibration"])
        q = art["conformal"].get("q", 0.10)
        drivers = _drivers(row, art)
        return {
            "score": round(score, 3),
            "interval": [round(max(0.0, score - q), 3), round(min(1.0, score + q), 3)],
            "priority": _risk_band(score),
            "drivers": drivers,
            "model_version": art.get("model_version", PIPELINE_VERSION),
            "needs_expert": False,
        }
    except Exception:  # pragma: no cover — never fail on real data
        return _fallback_prior(features)


def _drivers(row: dict, art: dict) -> list[dict]:
    """Top-3 features by |deviation from training mean| (interpretable, deterministic)."""
    means = {c: float(np.mean([lo, hi])) for c, (lo, hi) in art["num_ranges"].items()}
    spans = {c: max(1e-6, hi - lo) for c, (lo, hi) in art["num_ranges"].items()}
    scored: list[tuple[float, str, float]] = []
    for c in NUMERICS:
        v = row.get(c)
        if v is None or v != v:  # NaN
            continue
        dev = abs(v - means[c]) / spans[c]
        scored.append((dev, c, v))
    for col in CATEGORICALS:
        for c in art["cat_maps"].get(col, []):
            if row.get(f"{col}::{c}") == 1.0:
                scored.append((1.0, f"{col}::{c}", c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{"feature": f, "value": round(v, 2) if isinstance(v, float) else v}
            for _, f, v in scored[:3]]


def _fallback_prior(features: dict) -> dict:
    """Linear-prior fallback (same shape as ml/src/risk_engine.py) when XGBoost is absent."""
    try:
        from risk_engine import RiskInput, risk
        dp = 0.5 if str(features.get("primary_disease_risk", "")).lower() not in ("", "none (healthy)") else 0.1
        out = risk(RiskInput(dp, 0.5, 0.0, 0.5, 0.5))
        out["model_version"] = "risk-xgb-fallback-priors"
        out["needs_expert"] = True
        return out
    except Exception:  # pragma: no cover
        return {"score": 0.5, "interval": [0.4, 0.6], "priority": "medium",
                "drivers": [], "model_version": "risk-xgb-fallback-priors",
                "needs_expert": True}


def os_cpu_count() -> int:
    try:
        import os
        return os.cpu_count() or 4
    except Exception:  # pragma: no cover
        return 4


if __name__ == "__main__":
    print(f"PIPELINE_VERSION: {PIPELINE_VERSION}  xgboost={'yes' if _HAS_XGB else 'NO'}")
    print("Run the full pipeline via the notebook (CONFIG -> MODULE -> RUN) or:")
    print("  python -c \"from risk_xgb import Pipeline; Pipeline({'smoke': True}).run()\"")
