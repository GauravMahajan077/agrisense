"""Risk engine — transparent linear score + isotonic-style calibration + conformal interval.

Fitting runs on Kaggle (ml/kaggle/fit_risk.py); inference here is pure arithmetic
using a shipped JSON artifact — zero model runtime, zero local training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "risk_artifact.json"

FEATURE_ORDER = ["disease_prob", "conf", "rainfall_anomaly", "stage_risk", "yield_stress"]

# Fallback priors when artifact not yet fitted (rules-of-thumb, documented as v0)
DEFAULT_WEIGHTS = {
    "disease_prob": 1.6, "conf": -0.4, "rainfall_anomaly": 0.9,
    "stage_risk": 0.7, "yield_stress": 0.6, "intercept": -1.9,
}
DEFAULT_CALIBRATION = {"type": "none", "bins": []}
DEFAULT_CONFORMAL = {"q": 0.10, "n_calib": 0}


@dataclass(frozen=True)
class RiskInput:
    disease_prob: float  # top-class probability from /ml/disease
    conf: float  # model confidence (margin-aware)
    rainfall_anomaly: float  # (actual-normal)/normal clipped to [-1, 1]
    stage_risk: float  # 0..1 phenology risk (panicle/booting higher)
    yield_stress: float  # 0..1 district DWE yield stress proxy


def load_artifact(path: Path | None = None) -> dict:
    target = path or ARTIFACT_PATH
    if target.exists():
        with open(target, encoding="utf-8") as f:
            return json.load(f)
    return {"weights": DEFAULT_WEIGHTS, "calibration": DEFAULT_CALIBRATION,
            "conformal": DEFAULT_CONFORMAL, "model_version": "risk-v0-priors"}


def features(x: RiskInput) -> dict:
    return {"disease_prob": float(x.disease_prob), "conf": float(x.conf),
            "rainfall_anomaly": max(-1.0, min(1.0, float(x.rainfall_anomaly))),
            "stage_risk": float(x.stage_risk), "yield_stress": float(x.yield_stress)}


def raw_score(f: dict, weights: dict) -> float:
    z = weights.get("intercept", 0.0) + sum(weights[k] * f[k] for k in FEATURE_ORDER)
    return 1.0 / (1.0 + pow(2.718281828, -z))  # sigmoid, no numpy needed


def calibrate(score: float, calibration: dict) -> float:
    """Piecewise/isotonic calibration applied as linear interp over stored bins."""
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


def priority_band(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def risk(x: RiskInput, artifact: dict | None = None) -> dict:
    """Contract payload for POST /ml/risk: score + conformal interval + drivers."""
    art = artifact or load_artifact()
    f = features(x)
    s = calibrate(raw_score(f, art["weights"]), art["calibration"])
    q = art["conformal"].get("q", 0.10)
    drivers = sorted(
        ({"feature": k, "value": round(f[k], 3), "contribution": round(art["weights"].get(k, 0) * f[k], 3)}
         for k in FEATURE_ORDER), key=lambda d: abs(d["contribution"]), reverse=True)
    return {
        "score": round(s, 3),
        "interval": [round(max(0.0, s - q), 3), round(min(1.0, s + q), 3)],
        "priority": priority_band(s),
        "drivers": drivers[:3],
        "model_version": art.get("model_version", "risk-v0-priors"),
    }