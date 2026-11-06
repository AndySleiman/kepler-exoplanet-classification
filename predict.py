"""Score new Kepler Objects of Interest with the trained model.

Usage:
    python predict.py path/to/koi.csv [--model models/xgboost_pipeline.joblib] [--out predictions.csv]

The input CSV must contain the 12 feature columns listed in FEATS (the NASA KOI cumulative table does).
Output: one row per object with the predicted probability of being a real planet.
"""
import argparse
import joblib
import numpy as np
import pandas as pd

FEATS = ["koi_period", "koi_duration", "koi_depth", "koi_impact", "koi_model_snr",
         "koi_prad", "koi_teq", "koi_insol",
         "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"]
LOG_COLS = ["koi_period", "koi_duration", "koi_depth", "koi_model_snr",
            "koi_prad", "koi_insol", "koi_srad", "koi_teq"]


def log_transform(X):
    """Same transform used in training; must exist here so the saved pipeline can be loaded."""
    X = X.copy()
    for c in LOG_COLS:
        if c in X:
            X[c] = np.log10(X[c].clip(lower=1e-3))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="models/xgboost_pipeline.joblib")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, comment="#")
    model = joblib.load(args.model)
    df["planet_probability"] = model.predict_proba(df[FEATS])[:, 1]

    keep = [c for c in ["kepoi_name", "kepid", "koi_disposition"] if c in df] + ["planet_probability"]
    out = df[keep].sort_values("planet_probability", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} objects -> {args.out}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
