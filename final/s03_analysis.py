"""Step 3: model selection (validation only), failure analysis, failure fixes, SHAP, complexity.
The held-out test set is NOT used anywhere in this script."""
import sys, pickle, time, json, os, io
sys.path.insert(0, ".")
import __main__
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, shap
from sklearn.model_selection import cross_val_predict, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from torch_model import TorchMLP

FIG = "../results/figures"; os.makedirs(FIG, exist_ok=True)
df, lab, cand, tr, te, y, splits = load()
groups = tr["kepid"].values
S1 = pickle.load(open("cache/s01_classical.pkl", "rb")); S2 = pickle.load(open("cache/s02_torch.pkl", "rb"))
R = {**S1, **S2}                                    # name -> dict(metrics, oof, model, params)
R["Ensemble XGB + ResMLP +new features"] = dict(name="Ensemble XGB + ResMLP +new features",
    oof=(R["XGBoost +new features"]["oof"] + R["PyTorch ResMLP +new features"]["oof"]) / 2,
    metrics=None, model=None, params=None)
R["Ensemble XGB + ResMLP +new features"]["metrics"] = all_metrics(y, R["Ensemble XGB + ResMLP +new features"]["oof"])

# ------------------------------------------------------------------ 1. validation table
rows = {k: {**{m: v for m, v in r["metrics"].items()}} for k, r in R.items()}
val = pd.DataFrame(rows).T
val.to_csv("../results/validation_results_final.csv")
print(val[["PR-AUC", "ROC-AUC", "F1", "Recall@P95", "Brier"]].round(4).to_string())

# frozen primary model: highest validation PR-AUC among the candidates
cands = ["XGBoost +new features", "PyTorch ResMLP +new features", "Ensemble XGB + ResMLP +new features"]
primary = max(cands, key=lambda k: R[k]["metrics"]["PR-AUC"])
print("\nPRIMARY MODEL (by validation PR-AUC):", primary)
p0 = R[primary]["oof"]

# ------------------------------------------------------------------ 2. failure analysis (primary, threshold 0.5)
X12 = tr[FEATS]
pred = (p0 >= 0.5).astype(int)
fa = X12.copy(); fa["y"], fa["p"] = y, p0
fa["err"] = (pred != y).astype(int); fa["FP"] = ((y == 0) & (pred == 1)).astype(int); fa["FN"] = ((y == 1) & (pred == 0)).astype(int)
fa["SNR"] = pd.cut(fa.koi_model_snr, [0, 10, 20, 50, 1e9], labels=["<10", "10-20", "20-50", ">50"])
fa["Planet radius (R_earth)"] = pd.cut(fa.koi_prad, [0, 1.25, 2, 4, 15, 1e9], labels=["<1.25", "1.25-2", "2-4", "4-15", ">15"])
fa["Star Teff"] = pd.cut(fa.koi_steff, [0, 4000, 5200, 6000, 6800, 1e9], labels=["<4000K", "4000-5200", "5200-6000", "6000-6800", ">6800K"])
fa["Impact parameter"] = pd.cut(fa.koi_impact, [-1, 0.5, 0.9, 1.0, 1e9], labels=["<0.5", "0.5-0.9", "0.9-1.0", ">1 (grazing)"])
fa["Kepler mag"] = pd.cut(fa.koi_kepmag, [0, 12, 14, 15, 30], labels=["<12 (bright)", "12-14", "14-15", ">15 (faint)"])
fa["Period (d)"] = pd.cut(fa.koi_period, [0, 3, 10, 50, 1e9], labels=["<3", "3-10", "10-50", ">50"])
fa["Missing value"] = np.where(X12.isna().any(axis=1), "has missing", "complete")
def subgroup_table(col):
    g = fa.groupby(col, observed=True)
    out = pd.DataFrame({"n": g.size(), "planet rate": g.y.mean(), "error rate": g.err.mean(), "FP": g.FP.sum(), "FN": g.FN.sum()})
    out["PR-AUC"] = g.apply(lambda d: average_precision_score(d.y, d.p) if 0 < d.y.sum() < len(d) else np.nan)
    return out.round(3)
tabs = {}
print(f"\nFAILURES: total {len(fa)}, errors {fa.err.sum()} ({fa.err.mean():.3f}), false alarms {fa.FP.sum()}, missed planets {fa.FN.sum()}")
for col in ["SNR", "Planet radius (R_earth)", "Star Teff", "Impact parameter", "Kepler mag", "Period (d)", "Missing value"]:
    tabs[col] = subgroup_table(col); print("\n==", col); print(tabs[col].to_string())
pickle.dump(tabs, open("cache/s03_subgroups.pkl", "wb"))
cols = ["koi_prad", "koi_depth", "koi_model_snr", "koi_impact", "koi_period", "koi_duration", "koi_steff"]
prof = pd.DataFrame({k: fa.loc[m, cols].median() for k, m in {
    "False alarm (FP)": fa.FP == 1, "Missed planet (FN)": fa.FN == 1,
    "Found planet (TP)": (fa.y == 1) & (pred == 1), "Rejected FP (TN)": (fa.y == 0) & (pred == 0)}.items()}).T.round(2)
print("\nMedian profile per outcome:\n", prof.to_string())
conf = fa.assign(kepoi_name=tr["kepoi_name"].values).query("err == 1").assign(conf=lambda d: np.abs(d.p - 0.5)).sort_values("conf", ascending=False)
print("\nMost confident mistakes:\n", conf[["kepoi_name", "y", "p", "koi_model_snr", "koi_prad", "koi_impact", "koi_period"]].head(12).round(3).to_string(index=False))
conf.head(30).to_csv("../results/most_confident_errors.csv", index=False)

# ------------------------------------------------------------------ 3. fixes (all evaluated on validation folds only)
def cross_fit(fn, p):
    """Fit a post-hoc correction on 4 grouped folds of out-of-fold predictions, apply to the 5th (no leakage)."""
    out = np.zeros_like(p)
    for a, b in splits:
        out[b] = fn(p[a], y[a], p[b])
    return out
def platt(pa, ya, pb):
    lr = LogisticRegression(C=1e6).fit(np.log(np.clip(pa, 1e-4, 1 - 1e-4) / (1 - np.clip(pa, 1e-4, 1 - 1e-4)))[:, None], ya)
    return lr.predict_proba(np.log(np.clip(pb, 1e-4, 1 - 1e-4) / (1 - np.clip(pb, 1e-4, 1 - 1e-4)))[:, None])[:, 1]
def isotonic(pa, ya, pb):
    return IsotonicRegression(out_of_bounds="clip").fit(pa, ya).predict(pb)
def ece(yt, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop"); d = pd.DataFrame({"y": yt, "p": p, "q": q}).groupby("q", observed=True)
    return float(sum(len(g) * abs(g.y.mean() - g.p.mean()) for _, g in d) / len(p))
def best_f1_threshold(pa, ya):
    prec, rec, thr = precision_recall_curve(ya, pa); f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return thr[np.argmax(f1[:-1])]
def thr_pred(pa_all=None):
    out = np.zeros(len(y), dtype=int)
    for a, b in splits:
        out[b] = (p0[b] >= best_f1_threshold(p0[a], y[a])).astype(int)
    return out
def row(label, p, hard=None):
    d = {"fix": label, "PR-AUC": average_precision_score(y, p), "Recall@P95": recall_at_precision(y, p),
         "Brier": brier_score_loss(y, p), "ECE": ece(y, p)}
    h = hard if hard is not None else (p >= 0.5)
    d["F1"] = f1_score(y, h); d["errors"] = int((h != y).sum())
    for nm, mask in [("err SNR 10-20", (fa.SNR == "10-20").values), ("err radius>15", (fa["Planet radius (R_earth)"] == ">15").values),
                     ("err missing", (fa["Missing value"] == "has missing").values)]:
        d[nm] = float((h[mask] != y[mask]).mean())
    return d
fixes = [row("baseline (primary, thr 0.5)", p0)]
fixes.append(row("threshold tuned for F1 (cross-fit)", p0, hard=thr_pred()))
fixes.append(row("Platt calibration (cross-fit)", cross_fit(platt, p0)))
fixes.append(row("isotonic calibration (cross-fit)", cross_fit(isotonic, p0)))
# re-weighting: retrain the XGBoost with sample weights, same hyperparameters, same folds
xgbp = R["XGBoost +new features"]["model"]
def xgb_weighted(w, label):
    pw = cross_val_predict(xgbp, X12, y, cv=splits, method="predict_proba", params={"m__sample_weight": w})[:, 1]
    return pw, row(label, pw)
snr = tr.koi_model_snr.values; rad = tr.koi_prad.values
pw_list = {}
for label, w in {
    "re-weight: weak signals (SNR<20) x3": np.where(snr < 20, 3.0, 1.0),
    "re-weight: weak signals x3 + giant planets x3": np.where(snr < 20, 3.0, 1.0) * np.where(rad > 15, 3.0, 1.0),
    "re-weight: inverse class frequency": np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0),
}.items():
    pw, r = xgb_weighted(w, label); fixes.append(r); pw_list[label] = pw
fixdf = pd.DataFrame(fixes).set_index("fix")
print("\nFIXES (validation, 5-fold grouped, cross-fitted):\n", fixdf.round(4).to_string())
fixdf.to_csv("../results/failure_fixes.csv")
pickle.dump({"pw": pw_list}, open("cache/s03_fixes.pkl", "wb"))

fig, ax = plt.subplots(figsize=(5, 5))
for lbl, pp in [("uncalibrated", p0), ("Platt", cross_fit(platt, p0)), ("isotonic", cross_fit(isotonic, p0))]:
    fr, mp = calibration_curve(y, pp, n_bins=10, strategy="quantile"); ax.plot(mp, fr, marker="o", label=lbl)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction of planets")
ax.set_title("Calibration before / after correction"); ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/fig_calibration_fix.png", dpi=200); plt.close()

# ------------------------------------------------------------------ 4. SHAP on the XGBoost model (+ permutation importance)
xg = R["XGBoost +new features"]["model"]
Xt = xg[:-1].transform(tr[FEATS]); Xt = pd.DataFrame(Xt, columns=ALL_FEATS)
expl = shap.TreeExplainer(xg.named_steps["m"]); sv = expl.shap_values(Xt)
imp = pd.Series(np.abs(sv).mean(0), index=ALL_FEATS).sort_values(ascending=False)
print("\nMean |SHAP| (log-odds):\n", imp.round(3).to_string()); imp.to_csv("../results/shap_importance.csv")
shap.summary_plot(sv, Xt, show=False, max_display=17); plt.tight_layout(); plt.savefig(f"{FIG}/fig_shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(figsize=(6, 5)); imp.iloc[::-1].plot.barh(ax=ax, color=["#2a78d6" if f in NEW_FEATS else "#c4c3b8" for f in imp.index[::-1]])
ax.set_xlabel("mean |SHAP value| (log-odds)"); ax.set_title("Feature importance (blue = new features)"); plt.tight_layout(); plt.savefig(f"{FIG}/fig_shap_bar.png", dpi=200); plt.close()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for a, f in zip(axs, [imp.index[0], imp.index[1], imp.index[2]]):
    shap.dependence_plot(f, sv, Xt, ax=a, show=False, interaction_index=None)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_shap_dependence.png", dpi=200); plt.close()
# permutation importance (PR-AUC drop) on out-of-fold style: use 5 repeats on the full fitted model
from sklearn.inspection import permutation_importance
perm = permutation_importance(xg, tr[FEATS], y, scoring="average_precision", n_repeats=5, random_state=RS, n_jobs=2)
# permuting raw columns permutes the 12 base inputs; engineered columns are derived from them
pimp = pd.Series(perm.importances_mean, index=FEATS).sort_values(ascending=False)
print("\nPermutation importance of the 12 raw inputs (PR-AUC drop):\n", pimp.round(3).to_string()); pimp.to_csv("../results/permutation_importance.csv")
# SHAP of the most confident false alarm
i = np.where((fa.FP.values == 1))[0]; i = i[np.argmax(p0[i])]
print("\nMost confident false alarm:", tr.kepoi_name.values[i], "p=", round(float(p0[i]), 3))
print(pd.Series(sv[i], index=ALL_FEATS).sort_values(key=np.abs, ascending=False).head(6).round(2).to_string())

# ------------------------------------------------------------------ 5. complexity table
def size_kb(m):
    b = io.BytesIO(); joblib.dump(m, b); return len(b.getvalue()) / 1024
def n_par(name, m):
    mm = m.named_steps["m"]
    if "Logistic" in name: return f"{mm.coef_.size + mm.intercept_.size} weights"
    if "Random forest" in name: return f"{len(mm.estimators_)} trees (depth<={mm.max_depth})"
    if "XGBoost" in name: return f"{mm.get_booster().num_boosted_rounds()} trees (depth {mm.max_depth})"
    if "sklearn" in name: return f"{sum(w.size for w in mm.coefs_) + sum(b.size for b in mm.intercepts_)} weights"
    if "PyTorch" in name: return f"{mm.n_parameters()} weights ({len(mm.nets_)} nets)"
interp = {"Logistic regression": "High: one coefficient per feature", "Random forest": "Medium: feature importance only",
          "XGBoost": "Medium-high: TreeSHAP gives exact per-object explanations", "MLP (sklearn)": "Low: black box (post-hoc SHAP only)",
          "PyTorch ResMLP": "Low: black box (post-hoc SHAP only)", "Ensemble": "Low: black box"}
crow = []
for k in ["Logistic regression +new features", "Random forest +new features", "XGBoost +new features", "MLP (sklearn) +new features", "PyTorch ResMLP +new features"]:
    m = R[k]["model"]; Xb = tr[FEATS]
    t0 = time.time()
    for _ in range(5): m.predict_proba(Xb)
    us = (time.time() - t0) / 5 / len(Xb) * 1e6
    crow.append({"Model": k.replace(" +new features", ""), "PR-AUC": R[k]["metrics"]["PR-AUC"], "Size": n_par(k, m), "File size (KB)": size_kb(m),
                 "Tuning time (s)": R[k]["metrics"]["Tuning time (s)"], "Fit time (s)": R[k]["metrics"]["Fit time (s)"],
                 "Inference (us/object)": us, "Interpretability": interp[k.replace(" +new features", "")]})
ens_fit = R["XGBoost +new features"]["metrics"]["Fit time (s)"] + R["PyTorch ResMLP +new features"]["metrics"]["Fit time (s)"]
crow.append({"Model": "Ensemble XGB + ResMLP", "PR-AUC": R["Ensemble XGB + ResMLP +new features"]["metrics"]["PR-AUC"], "Size": "both models",
             "File size (KB)": crow[2]["File size (KB)"] + crow[4]["File size (KB)"], "Tuning time (s)": crow[2]["Tuning time (s)"] + crow[4]["Tuning time (s)"],
             "Fit time (s)": ens_fit, "Inference (us/object)": crow[2]["Inference (us/object)"] + crow[4]["Inference (us/object)"], "Interpretability": interp["Ensemble"]})
cx = pd.DataFrame(crow).set_index("Model"); print("\nCOMPLEXITY:\n", cx.round(2).to_string()); cx.to_csv("../results/complexity.csv")
json.dump({"primary": primary}, open("cache/primary_choice.json", "w"))
print("DONE")
