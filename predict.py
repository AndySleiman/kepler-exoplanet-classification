"""Score Kepler Objects of Interest with the final model (XGBoost + PyTorch ResMLP ensemble, 17 features).

Usage:
    python predict.py path/to/koi.csv [--out predictions.csv]

The input CSV needs the 12 base columns in FEATS (the NASA KOI cumulative table has them); the 5 engineered
features are computed automatically inside the saved pipelines.
Output: one row per object with the predicted probability of being a real planet, sorted high to low.
"""
import argparse, sys, os
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "final"))   # saved pipelines need common.py / torch_model.py
from common import FEATS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, comment="#")
    xgb = joblib.load(os.path.join(HERE, "models", "final_xgboost.joblib"))
    nn = joblib.load(os.path.join(HERE, "models", "final_resmlp.joblib"))
    df["planet_probability"] = (xgb.predict_proba(df[FEATS])[:, 1] + nn.predict_proba(df[FEATS])[:, 1]) / 2
    keep = [c for c in ["kepoi_name", "kepid", "koi_disposition"] if c in df] + ["planet_probability"]
    out = df[keep].sort_values("planet_probability", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} objects -> {args.out}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
