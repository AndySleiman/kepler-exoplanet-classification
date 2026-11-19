"""Step 4: freeze everything on validation data, evaluate ONCE on the held-out test set, rank the CANDIDATE objects."""
import sys, pickle, json, os, time
sys.path.insert(0, ".")
from common import *
import joblib
from sklearn.base import clone
from scipy.stats import spearmanr

df, lab, cand, tr, te, y, splits = load()
S1 = pickle.load(open("cache/s01_classical.pkl", "rb")); S2 = pickle.load(open("cache/s02_torch.pkl", "rb"))
R = {**S1, **S2}
xgb_n, nn_n = "XGBoost +new features", "PyTorch ResMLP +new features"
p_val = (R[xgb_n]["oof"] + R[nn_n]["oof"]) / 2

# ---------- freeze (validation only) ----------
prec, rec, thr = precision_recall_curve(y, p_val)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
thr_f1 = float(thr[np.argmax(f1[:-1])])
ok = np.where(prec[:-1] >= 0.95)[0]; thr_p95 = float(thr[ok[np.argmax(rec[ok])]])
frozen = {"primary_model": "Equal-weight average of XGBoost and PyTorch ResMLP (17 features)",
          "features": ALL_FEATS, "threshold_default": 0.5, "threshold_F1_optimal_on_validation": thr_f1,
          "threshold_95pct_precision_on_validation": thr_p95, "val_PR_AUC": float(average_precision_score(y, p_val)),
          "xgb_params": R[xgb_n]["params"], "resmlp_params": R[nn_n]["params"], "split_seed": RS,
          "selection_rule": "highest 5-fold grouped-CV PR-AUC on the 80% train/validation part; test set not used"}
os.makedirs("../results", exist_ok=True); json.dump(frozen, open("../results/frozen_config.json", "w"), indent=2, default=str)
print("FROZEN:", json.dumps({k: v for k, v in frozen.items() if k.startswith(("thr", "threshold", "val"))}, default=str))

# ---------- ONE evaluation on the held-out test set ----------
yt = te["y"].values; Xte = te[FEATS]
tp = {k: R[k]["model"].predict_proba(Xte)[:, 1] for k in R if R[k]["model"] is not None}
tp["Ensemble XGB + ResMLP +new features"] = (tp[xgb_n] + tp[nn_n]) / 2
rows = {}
for k, p in tp.items():
    m = all_metrics(yt, p); rows[k] = m
test_df = pd.DataFrame(rows).T; test_df.to_csv("../results/test_results.csv")
print("\nTEST RESULTS (single evaluation):\n", test_df.round(4).to_string())

P = tp["Ensemble XGB + ResMLP +new features"]
# thresholds chosen on validation, applied to test
from sklearn.metrics import precision_score, recall_score, confusion_matrix
tr_rows = []
for nm, t in [("default 0.5", 0.5), ("F1-optimal (val)", thr_f1), ("95%-precision (val)", thr_p95)]:
    h = (P >= t).astype(int); tn, fp, fn, tpp = confusion_matrix(yt, h).ravel()
    tr_rows.append({"threshold": nm, "value": round(t, 3), "precision": precision_score(yt, h), "recall": recall_score(yt, h), "F1": f1_score(yt, h), "TP": tpp, "FP": fp, "FN": fn, "TN": tn})
thr_df = pd.DataFrame(tr_rows).set_index("threshold"); thr_df.to_csv("../results/test_thresholds.csv"); print("\nThresholds on test:\n", thr_df.round(3).to_string())

# bootstrap CI, resampling whole stars (so KOIs of the same star stay together)
rng = np.random.default_rng(RS); stars = te["kepid"].values; ustars = np.unique(stars)
idx_by = {s: np.where(stars == s)[0] for s in ustars}; B = []
for _ in range(1000):
    pick = rng.choice(ustars, len(ustars), replace=True); ii = np.concatenate([idx_by[s] for s in pick])
    if yt[ii].sum() == 0: continue
    B.append([average_precision_score(yt[ii], P[ii]), roc_auc_score(yt[ii], P[ii]), f1_score(yt[ii], P[ii] >= .5), recall_at_precision(yt[ii], P[ii]), brier_score_loss(yt[ii], P[ii])])
B = np.array(B); ci = pd.DataFrame({"2.5%": np.percentile(B, 2.5, 0), "97.5%": np.percentile(B, 97.5, 0)}, index=["PR-AUC", "ROC-AUC", "F1", "Recall@P95", "Brier"])
ci.to_csv("../results/test_bootstrap_ci.csv"); print("\nBootstrap 95% CI (primary):\n", ci.round(4).to_string())

# test-set failure breakdown
fa = Xte.copy(); fa["y"], fa["p"] = yt, P; fa["err"] = ((P >= .5).astype(int) != yt).astype(int)
fa["SNR"] = pd.cut(fa.koi_model_snr, [0, 10, 20, 50, 1e9], labels=["<10", "10-20", "20-50", ">50"])
fa["Radius"] = pd.cut(fa.koi_prad, [0, 1.25, 2, 4, 15, 1e9], labels=["<1.25", "1.25-2", "2-4", "4-15", ">15"])
fa["Missing"] = np.where(Xte.isna().any(axis=1), "has missing", "complete")
tt = {}
for c in ["SNR", "Radius", "Missing"]:
    g = fa.groupby(c, observed=True); tt[c] = pd.DataFrame({"n": g.size(), "planet rate": g.y.mean(), "error rate": g.err.mean()}).round(3); print("\n", tt[c].to_string())
print("test errors:", int(fa.err.sum()), "of", len(fa))
pickle.dump({"test_df": test_df, "thr_df": thr_df, "ci": ci, "tt": tt, "P": P, "yt": yt}, open("cache/s04_test.pkl", "wb"))

# ---------- refit on ALL labelled data and rank the CANDIDATEs ----------
Xall, yall = lab[FEATS], lab["y"].values
xg_f = clone(R[xgb_n]["model"]).fit(Xall, yall)
nn_f = clone(R[nn_n]["model"]).fit(Xall, yall)
joblib.dump(xg_f, "../models/final_xgboost.joblib"); joblib.dump(nn_f, "../models/final_resmlp.joblib")
pc = (xg_f.predict_proba(cand[FEATS])[:, 1] + nn_f.predict_proba(cand[FEATS])[:, 1]) / 2
out = cand[["kepoi_name", "kepid", "koi_period", "koi_prad", "koi_model_snr", "koi_steff"]].copy(); out["planet_probability"] = pc
out = out.sort_values("planet_probability", ascending=False); out["rank"] = np.arange(1, len(out) + 1)
out.to_csv("../results/candidate_ranking.csv", index=False)
print("\nCANDIDATES:", len(out), "| p>=0.5:", int((pc >= .5).sum()), "| p>=thr95:", int((pc >= thr_p95).sum()), "| p<0.1:", int((pc < .1).sum()), "| missing-feature rows:", int(cand[FEATS].isna().any(axis=1).sum()))
print(out.head(10).round(3).to_string(index=False))
# external sanity check: agreement with the NASA Robovetter score (NOT used as a feature)
sc = cand["koi_score"].reindex(out.index) if False else cand.loc[out.index, "koi_score"]
m = sc.notna().values; print("Spearman(our probability, koi_score) on candidates:", round(spearmanr(out.planet_probability.values[m], sc.values[m])[0], 3))
print("candidate p-deciles counts:", np.histogram(pc, bins=np.linspace(0, 1, 11))[0].tolist())
json.dump({"n": len(out), "ge05": int((pc >= .5).sum()), "ge_p95": int((pc >= thr_p95).sum()), "lt01": int((pc < .1).sum()),
           "spearman_koi_score": float(spearmanr(out.planet_probability.values[m], sc.values[m])[0]), "hist": np.histogram(pc, bins=np.linspace(0, 1, 11))[0].tolist()},
          open("../results/candidate_summary.json", "w"))
print("DONE")
