# Fit risk-engine artifact — RUN ON KAGGLE (seconds, CPU ok there)
# Input CSV columns: disease_prob,conf,rainfall_anomaly,stage_risk,yield_stress,target
# target = 0..1 loss/severity label you annotate (or proxy: expert_priority rating)
# Output: ml/models/risk_artifact.json  -> upload back, ml/src/risk_engine.py loads it
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CSV = "/kaggle/input/risk-data/risk_features.csv"  # <- EDIT
OUT = "risk_artifact.json"
SEED = 1337

FEATS = ["disease_prob", "conf", "rainfall_anomaly", "stage_risk", "yield_stress"]


def main():
    df = pd.read_csv(CSV)
    # chronological split: first 80% train, last 20% calibration (no shuffle)
    df = df.sort_values(df.columns[0], kind="stable") if "date" in df.columns else df
    split = int(0.8 * len(df))
    tr, cal = df.iloc[:split], df.iloc[split:]

    lr = LogisticRegression(max_iter=1000)
    lr.fit(tr[FEATS], (tr["target"] > tr["target"].median()).astype(int))
    raw_tr = lr.decision_function(tr[FEATS]) / 3.0 + 0.5  # pseudo-prob for isotonic
    raw_cal = lr.decision_function(cal[FEATS]) / 3.0 + 0.5

    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_tr.clip(0, 1), tr["target"])
    cal_pred = np.clip(iso.predict(raw_cal), 0, 1)
    q = float(np.quantile(np.abs(cal_pred - cal["target"].to_numpy()), 0.90))  # 90% interval

    xs = np.linspace(0, 1, 11)
    bins = [[float(x), float(iso.predict([x])[0])] for x in xs]

    artifact = {
        "weights": {"intercept": float(lr.intercept_[0]),
                    **{f: float(c) for f, c in zip(FEATS, lr.coef_[0])}},
        "calibration": {"type": "isotonic", "bins": bins},
        "conformal": {"q": round(q, 3), "n_calib": int(len(cal))},
        "model_version": "risk-v1",
        "feature_order": FEATS,
        "n_train": int(len(tr)),
    }
    with open(OUT, "w") as f:
        json.dump(artifact, f, indent=2)
    print("artifact ->", OUT, "| conformal q =", artifact["conformal"]["q"])


if __name__ == "__main__":
    main()