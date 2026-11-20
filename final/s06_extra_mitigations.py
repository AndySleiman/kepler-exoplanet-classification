"""Step 6 (validation only, test set NOT used): extra mitigations and a quantified interpretability trade-off."""
import sys, pickle, json
sys.path.insert(0, ".")
from common import *
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.base import clone
df, lab, cand, tr, te, y, splits = load()
S1 = pickle.load(open("cache/s01_classical.pkl", "rb")); S2 = pickle.load(open("cache/s02_torch.pkl", "rb"))
xgb_n, nn_n = "XGBoost +new features", "PyTorch ResMLP +new features"
p0 = (S1[xgb_n]["oof"] + S2[nn_n]["oof"]) / 2
snr = tr.koi_model_snr.values
out = {}

# ---- A. selective prediction: defer uncertain objects to human vetters --------------------------
rows = []
for band in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    keep = np.abs(p0 - 0.5) >= band
    h = (p0 >= 0.5).astype(int)
    m = keep & (snr >= 10) & (snr < 20)
    rows.append({"defer band |p-0.5|<": band, "coverage": keep.mean(), "error rate on decided": (h[keep] != y[keep]).mean(),
                 "errors caught by deferral": int(((h != y) & ~keep).sum()), "objects deferred": int((~keep).sum()),
                 "error SNR 10-20 (decided)": (h[m] != y[m]).mean(), "coverage SNR 10-20": m.sum() / ((snr >= 10) & (snr < 20)).sum()})
A = pd.DataFrame(rows).set_index("defer band |p-0.5|<"); print("A. SELECTIVE PREDICTION\n", A.round(3).to_string()); A.to_csv("../results/mitigation_deferral.csv")

# ---- B. specialist model for weak signals (SNR<30), blended with the global model ----------------
xg = S1[xgb_n]["model"]; weak = snr < 30
pb = p0.copy()
for a, b in splits:
    a_w = a[weak[a]]; b_w = b[weak[b]]
    sp = clone(xg).fit(tr[FEATS].iloc[a_w], y[a_w])
    pb[b_w] = 0.5 * p0[b_w] + 0.5 * sp.predict_proba(tr[FEATS].iloc[b_w])[:, 1]
def rep(label, p):
    h = (p >= .5).astype(int); m = (snr >= 10) & (snr < 20)
    return {"variant": label, "PR-AUC": average_precision_score(y, p), "Recall@P95": recall_at_precision(y, p), "errors": int((h != y).sum()),
            "error SNR 10-20": float((h[m] != y[m]).mean()), "PR-AUC SNR 10-20": average_precision_score(y[m], p[m])}
B = pd.DataFrame([rep("global ensemble", p0), rep("global + weak-signal specialist (SNR<30)", pb)]).set_index("variant")
print("\nB. WEAK-SIGNAL SPECIALIST\n", B.round(4).to_string()); B.to_csv("../results/mitigation_specialist.csv")

# ---- C. interpretability vs accuracy: how much does readability cost? ---------------------------
rows = []
for d in [2, 3, 4, 6, 8]:
    pipe = Pipeline(preprocessing(False, True) + [("m", DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=RS))])
    p = cross_val_predict(pipe, tr[FEATS], y, cv=splits, method="predict_proba")[:, 1]
    rows.append({"model": f"decision tree depth {d}", "readable size": f"{2 ** d} leaves", "PR-AUC": average_precision_score(y, p), "Recall@P95": recall_at_precision(y, p)})
# Exploratory: the SHAP ranking and imputer see the entire development partition.
# A fold-independent estimate would fit both again inside each training fold.
imp = pd.read_csv("../results/shap_importance.csv", index_col=0).iloc[:, 0]
for k in [3, 5, 8]:
    top = list(imp.index[:k]); raw = [f for f in top if f in FEATS]
    # engineered features need raw columns: keep raw columns required to build them
    Xk = Pipeline(preprocessing(False, True))   # build full features then select columns
    full = Pipeline(preprocessing(False, True)).fit_transform(tr[FEATS]); full = pd.DataFrame(full, columns=ALL_FEATS)
    pk = cross_val_predict(clone(xg.named_steps["m"]), full[top].values, y, cv=splits, method="predict_proba")[:, 1]
    rows.append({"model": f"XGBoost on top-{k} SHAP features", "readable size": f"{k} features", "PR-AUC": average_precision_score(y, pk), "Recall@P95": recall_at_precision(y, pk)})
rows.append({"model": "XGBoost (17 features)", "readable size": "200 trees", "PR-AUC": average_precision_score(y, S1[xgb_n]["oof"]), "Recall@P95": recall_at_precision(y, S1[xgb_n]["oof"])})
rows.append({"model": "Ensemble XGB + ResMLP", "readable size": "black box", "PR-AUC": average_precision_score(y, p0), "Recall@P95": recall_at_precision(y, p0)})
C = pd.DataFrame(rows).set_index("model"); print("\nC. INTERPRETABILITY vs ACCURACY\n", C.round(4).to_string()); C.to_csv("../results/interpretability_tradeoff.csv")
pipe = Pipeline(preprocessing(False, True) + [("m", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RS))]).fit(tr[FEATS], y)
print("\nDepth-3 tree rules:\n", export_text(pipe.named_steps["m"], feature_names=ALL_FEATS, decimals=2))
