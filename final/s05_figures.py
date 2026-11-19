"""Step 5: report figures from the saved results."""
import sys, pickle, json
sys.path.insert(0, ".")
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG = "../results/figures"
val = pd.read_csv("../results/validation_results_final.csv", index_col=0); test = pd.read_csv("../results/test_results.csv", index_col=0)
S = pickle.load(open("cache/s04_test.pkl", "rb")); tabs = pickle.load(open("cache/s03_subgroups.pkl", "rb"))
# 1. validation vs test PR-AUC
order = ["Logistic regression", "Random forest", "XGBoost", "MLP (sklearn)", "Logistic regression +new features", "Random forest +new features",
         "XGBoost +new features", "MLP (sklearn) +new features", "PyTorch ResMLP +new features", "Ensemble XGB + ResMLP +new features"]
short = ["LR", "RF", "XGB", "MLP", "LR+", "RF+", "XGB+", "MLP+", "ResMLP+", "Ens+"]
fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(order)); w = 0.38
ax.bar(x - w / 2, val.loc[order, "PR-AUC"], w, label="validation (5-fold grouped CV)", color="#c4c3b8")
ax.bar(x + w / 2, test.loc[order, "PR-AUC"], w, label="held-out test (once)", color="#2a78d6")
ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0.7, 1.0); ax.set_ylabel("PR-AUC"); ax.legend(loc="lower right")
ax.set_title("PR-AUC by model ('+' = with the 5 new features)"); plt.tight_layout(); plt.savefig(f"{FIG}/fig_models_pr_auc.png", dpi=200); plt.close()
# 2. PR curve on test (primary)
P, yt = S["P"], S["yt"]; pr, rc, _ = precision_recall_curve(yt, P)
fig, ax = plt.subplots(figsize=(4.6, 4.2)); ax.plot(rc, pr, color="#2a78d6", lw=2); ax.axhline(0.95, ls="--", c="grey", lw=1)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Test precision-recall curve (AP = {average_precision_score(yt, P):.3f})"); ax.set_ylim(0.4, 1.02)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_test_pr_curve.png", dpi=200); plt.close()
# 3. failure analysis: error rate by subgroup (validation)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (col, title) in zip(axs, [("SNR", "Transit SNR"), ("Planet radius (R_earth)", "Planet radius (Earth radii)"), ("Impact parameter", "Impact parameter")]):
    t = tabs[col]; a.bar(t.index.astype(str), t["error rate"], color="#d6402a"); a.set_title(title); a.set_ylabel("error rate"); a.tick_params(axis="x", labelsize=8)
    for i, (e, n) in enumerate(zip(t["error rate"], t["n"])): a.text(i, e + 0.002, f"n={n}", ha="center", fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_failure_subgroups.png", dpi=200); plt.close()
# 4. candidate probabilities
c = pd.read_csv("../results/candidate_ranking.csv")
fig, ax = plt.subplots(figsize=(5, 3.4)); ax.hist(c.planet_probability, bins=20, color="#2a78d6"); ax.set_xlabel("Predicted probability of being a planet"); ax.set_ylabel("CANDIDATE objects")
ax.set_title("Scores of the 1,977 CANDIDATE objects"); plt.tight_layout(); plt.savefig(f"{FIG}/fig_candidates.png", dpi=200); plt.close()
print("figures saved")
