"""Step 3b: dataset statistics and leakage ablation (train/validation part only; test set untouched)."""
import sys, pickle, json
sys.path.insert(0, ".")
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
df, lab, cand, tr, te, y, splits = load()
S1 = pickle.load(open("cache/s01_classical.pkl", "rb"))
print("Disposition counts:", df.koi_disposition.value_counts().to_dict(), "| total", len(df), "| columns", df.shape[1])
print("Labelled:", len(lab), "positive rate", round(lab.y.mean(), 4), "| train", len(tr), tr.kepid.nunique(), "stars | test", len(te), te.kepid.nunique(), "stars")
print("Stars with >1 KOI:", int((df.kepid.value_counts() > 1).sum()), "of", df.kepid.nunique())
print("\nMissing rate per base feature (labelled):\n", lab[FEATS].isna().mean().round(3).to_string())
print("Rows with >=1 missing:", round(lab[FEATS].isna().any(axis=1).mean(), 4), "| planet rate among them:", round(lab[lab[FEATS].isna().any(axis=1)].y.mean(), 3))
# leakage ablation
FLAGS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
leaky = FEATS + ["koi_score"] + FLAGS
xg = S1["XGBoost"]["model"]
mk = lambda: clone(xg)
p_base = S1["XGBoost"]["oof"]
pl = cross_val_predict(mk(), tr[leaky], y, cv=splits, method="predict_proba")[:, 1]
print("\nXGB honest 12 feats      PR-AUC", round(average_precision_score(y, p_base), 4))
print("XGB + koi_score + 4 flags PR-AUC", round(average_precision_score(y, pl), 4))
fl = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", LogisticRegression(max_iter=2000))])
pf = cross_val_predict(fl, tr[FLAGS], y, cv=splits, method="predict_proba")[:, 1]
print("Logistic regression on the 4 flags only PR-AUC", round(average_precision_score(y, pf), 4))
ps = cross_val_predict(Pipeline([("imp", SimpleImputer(strategy="median")), ("m", LogisticRegression(max_iter=2000))]), tr[["koi_score"]], y, cv=splits, method="predict_proba")[:, 1]
print("Logistic regression on koi_score only PR-AUC", round(average_precision_score(y, ps), 4))
# process signals
tr2 = tr.copy(); tr2["n_koi_system"] = tr2.kepid.map(df.kepid.value_counts())
pp = cross_val_predict(mk(), tr2[FEATS + ["koi_tce_plnt_num", "n_koi_system"]], y, cv=splits, method="predict_proba")[:, 1]
print("XGB (same hp) + planet number + KOIs per star PR-AUC", round(average_precision_score(y, pp), 4))
multi = df.kepid.map(df.kepid.value_counts()) > 1
print("\nPlanet share by multi-KOI star:\n", pd.crosstab(multi.rename(">1 KOI"), df.koi_disposition, normalize="index").round(3).to_string())
# EDA stats for report
print("\nPlanet rate by SNR bin:", tr.groupby(pd.cut(tr.koi_model_snr, [0, 10, 20, 50, 1e9]), observed=True).y.mean().round(3).to_dict())
print("Planet rate by radius bin:", tr.groupby(pd.cut(tr.koi_prad, [0, 2, 4, 15, 1e9]), observed=True).y.mean().round(3).to_dict())
# ablation figure
fig, ax = plt.subplots(figsize=(6.5, 3.6))
vals = {"12 physical features\n(honest)": average_precision_score(y, p_base), "+ planet number,\nKOIs per star": average_precision_score(y, pp),
        "+ koi_score + 4 flags\n(leaky)": average_precision_score(y, pl)}
ax.bar(vals.keys(), vals.values(), color=["#2a78d6", "#c4c3b8", "#d6402a"]); ax.set_ylim(0.9, 1.0); ax.set_ylabel("Validation PR-AUC (XGBoost)")
for i, v in enumerate(vals.values()): ax.text(i, v + 0.001, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig("../results/figures/fig_leakage.png", dpi=200)
json.dump({"honest": float(vals["12 physical features\n(honest)"]), "process": float(vals["+ planet number,\nKOIs per star"]), "leaky": float(vals["+ koi_score + 4 flags\n(leaky)"]),
           "flags_only": float(average_precision_score(y, pf)), "score_only": float(average_precision_score(y, ps))}, open("../results/leakage_ablation.json", "w"))
