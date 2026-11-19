"""Shared setup for the final experiments: data, split, folds, metrics, feature engineering.
Uses exactly the same split (seed 42, grouped by star) as the progress-report notebook."""
import time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             brier_score_loss, precision_recall_curve)
warnings.filterwarnings("ignore")
RS = 42
DATA_PATH = "../data/kepler_cumulative.csv"

FEATS = ["koi_period", "koi_duration", "koi_depth", "koi_impact", "koi_model_snr",
         "koi_prad", "koi_teq", "koi_insol", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"]
LOG_COLS = ["koi_period", "koi_duration", "koi_depth", "koi_model_snr",
            "koi_prad", "koi_insol", "koi_srad", "koi_teq"]

# ---- new physically motivated features (computed from the 12 base columns only) ----
NEW_FEATS = ["depth_consistency", "duration_ratio", "implied_noise", "star_density", "rp_over_rs"]
ALL_FEATS = FEATS + NEW_FEATS
R_SUN_IN_R_EARTH = 109.1
KEPLER_BASELINE_DAYS = 1460.0

def add_features(X):
    """depth_consistency : observed depth / depth expected from (Rp/Rs)^2. ~1 for a real planet-sized dip,
                           far from 1 when the 'planet radius' was derived from a contaminated/ eclipsing signal.
       duration_ratio    : observed duration / duration expected for a circular central transit around a star of
                           this density (T ~ P^(1/3) * rho^(-1/3)).  Eclipsing binaries on other stars deviate.
       implied_noise     : per-point noise level implied by depth, SNR and the number of transits (weak-signal indicator).
       star_density      : relative stellar density from log g and radius (rho ~ g / R).
       rp_over_rs        : planet-to-star radius ratio."""
    X = X.copy()
    rp_rs = (X["koi_prad"] / (X["koi_srad"] * R_SUN_IN_R_EARTH)).clip(lower=1e-6)
    X["rp_over_rs"] = np.log10(rp_rs)
    expected_depth_ppm = (rp_rs ** 2) * 1e6
    X["depth_consistency"] = np.log10((X["koi_depth"].clip(lower=1e-3) / expected_depth_ppm).clip(lower=1e-3))
    rho = (10 ** X["koi_slogg"]) / X["koi_srad"].clip(lower=1e-3)          # ~ density up to a constant
    X["star_density"] = np.log10(rho.clip(lower=1e-3))
    expected_dur = (X["koi_period"].clip(lower=1e-3) ** (1 / 3)) * (rho.clip(lower=1e-3) ** (-1 / 3))
    X["duration_ratio"] = np.log10((X["koi_duration"].clip(lower=1e-3) / expected_dur).clip(lower=1e-3))
    n_tr = (KEPLER_BASELINE_DAYS / X["koi_period"].clip(lower=1e-3)).clip(lower=1)
    X["implied_noise"] = np.log10((X["koi_depth"].clip(lower=1e-3) * np.sqrt(n_tr) / X["koi_model_snr"].clip(lower=1e-3)).clip(lower=1e-3))
    return X

def log_transform(X):
    X = X.copy()
    for c in LOG_COLS:
        if c in X:
            X[c] = np.log10(X[c].clip(lower=1e-3))
    return X

def log_and_engineer(X):
    return log_transform(add_features(X))     # engineered features use RAW units, then logs are applied

def preprocessing(scale, engineered=False):
    steps = [("log", FunctionTransformer(log_and_engineer if engineered else log_transform)),
             ("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return steps

def recall_at_precision(y_true, p, target=0.95):
    prec, rec, _ = precision_recall_curve(y_true, p)
    ok = rec[prec >= target]
    return float(ok.max()) if len(ok) else 0.0

def all_metrics(y_true, p, thr=0.5):
    return {"PR-AUC": average_precision_score(y_true, p), "ROC-AUC": roc_auc_score(y_true, p),
            "F1": f1_score(y_true, p >= thr), "Recall@P95": recall_at_precision(y_true, p),
            "Brier": brier_score_loss(y_true, p)}

def load():
    df = pd.read_csv(DATA_PATH, comment="#")
    lab = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    cand = df[df["koi_disposition"] == "CANDIDATE"].copy()
    lab["y"] = (lab["koi_disposition"] == "CONFIRMED").astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
    tr_idx, te_idx = next(gss.split(lab, lab["y"], lab["kepid"]))
    tr, te = lab.iloc[tr_idx].copy(), lab.iloc[te_idx].copy()
    y = tr["y"].values
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(tr, y, tr["kepid"]))
    return df, lab, cand, tr, te, y, splits
